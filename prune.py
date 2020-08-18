import sys
import collections
import torch
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq import options, progress_bar, tasks, utils
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq_cli.train import (
    load_dataset_splits,
    load_checkpoint,
)
from itertools import islice
from fairseq import checkpoint_utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq_cli.interactive import translate_corpus, parse_head_pruning_descriptors, mask_heads
from math import ceil
import sacrebleu
import os


def eval_bleu_score(
        model,
        task,
        eval_data,
        beam=5,
        replace_unk=True,
        lenpen=1.0,
        buffer_size=100,
        use_cuda=True,
        remove_bpe=False,
        max_sentences=32,
        max_tokens=9999,
        stop_early=True,
        normalize_scores=True,
        min_len=2,
):
    print(len(task.target_dictionary))
    # Initialize generator
    translator = SequenceGenerator(
        [model], task.target_dictionary, beam_size=beam, minlen=min_len,
        stop_early=stop_early, normalize_scores=normalize_scores,
        len_penalty=lenpen,
        sampling=False,
    )

    results = translate_corpus(
        translator,
        task,
        input_feed=[eval_data.src.get_original_text(
            i) for i in range(len(eval_data.src))],
        buffer_size=buffer_size,
        replace_unk=replace_unk,
        use_cuda=use_cuda,
        print_directly=False,
        nbest=1,
        remove_bpe=remove_bpe,
        print_alignment=False,
        max_sentences=max_sentences,
        max_tokens=max_tokens,
    )

    out = [result.hypos[0].split("\t")[-1] for result in results]
    ref = [(eval_data.tgt.get_original_text(i) + " ").replace(remove_bpe, "")
           for i in range(len(eval_data.tgt))]

    return sacrebleu.corpus_bleu(out, [ref], force=True, tokenize="none")


def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', "valid"])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {},'.format(
        args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel()
                                               for p in model.parameters())))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    '''
    dummy_batch = task.dataset('train').get_dummy_batch(
        args.max_tokens, max_positions)
'''
    quantizer = None
    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))
    print('| Optimizer {}'.format(trainer.optimizer.__class__.__name__))
'''
    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )
    '''
    # Load the latest checkpoint if one is available
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    prune_meter = StopwatchMeter()
    prune_meter.start()
    # Estimate head importance scores
    head_importance, head_stats = estimate_head_importance(
        args, trainer, task, epoch_itr)
    prune_meter.stop()
    print('| done estimating head importance in {:.1f} seconds'.format(
        prune_meter.sum))
    torch.save(
        head_stats, f"{os.path.dirname(args.restore_file)}/heads_stats.bin")
    # Print
    print("Head importances")
    print("Encoder self attention")
    for layer in range(head_importance["encoder_self"].size(0)):
        print(
            "\t".join(f"{x:.5f}" for x in head_importance["encoder_self"][layer]))
    print("Encoder decoder attention")
    for layer in range(head_importance["encoder_decoder"].size(0)):
        print(
            "\t".join(f"{x:.5f}" for x in head_importance["encoder_decoder"][layer]))
    print("Decoder self attention")
    for layer in range(head_importance["decoder_self"].size(0)):
        print(
            "\t".join(f"{x:.5f}" for x in head_importance["decoder_self"][layer]))
    # Print sorted pruning profile
    encoder_self_profile = get_profile(
        head_importance["encoder_self"], prefix="E")
    encoder_decoder_profile = get_profile(
        head_importance["encoder_decoder"], prefix="A")
    decoder_self_profile = get_profile(
        head_importance["decoder_self"], prefix="D")
    # Join all
    all_profiles = {}
    if not (args.decoder_self_only or args.encoder_decoder_only):
        all_profiles.update(encoder_self_profile)
    if not (args.encoder_self_only or args.decoder_self_only):
        all_profiles.update(encoder_decoder_profile)
    if not (args.encoder_self_only or args.encoder_decoder_only):
        all_profiles.update(decoder_self_profile)
    sorted_profiles = sorted(
        all_profiles.items(),
        key=lambda x: x[1],
        reverse=args.one_minus
    )
    print("Heads sorted by importance:")
    print(" ".join(p for p, _ in sorted_profiles))
    print("Sorted head importance scores:")
    print(" ".join(f"{v.data:.5f}" for _, v in sorted_profiles))
    print("Guy comment -> exit")
    exit()
    if args.only_importance:
        return

    tot_n_heads = len(sorted_profiles)
    # Eval pruning
    if args.one_head:
        kept_layers = set()
        to_prune_profile = []
        for p, _ in reversed(sorted_profiles):
            layer_name = ":".join(p.split(":")[:-1])
            if layer_name not in kept_layers:
                kept_layers.add(layer_name)
                continue
            else:
                to_prune_profile.insert(0, p)
        to_prune = parse_head_pruning_descriptors(
            to_prune_profile, reverse_descriptors=False)
        print(f"Evaluating following profile: \t{' '.join(to_prune_profile)}")
        # Apply pruning
        mask_heads(model, to_prune, args.transformer_mask_rescale)
        bleu = eval_bleu_score(
            model,
            task,
            task.dataset(args.valid_subset),
            beam=args.beam,
            replace_unk=args.replace_unk,
            lenpen=args.lenpen,
            buffer_size=100,
            use_cuda=torch.cuda.is_available() and not args.cpu,
            remove_bpe=args.remove_bpe,
            max_sentences=args.max_sentences,
            max_tokens=args.max_tokens,
            stop_early=not args.no_early_stop,
            normalize_scores=not args.unnormalized,
            min_len=args.min_len,
        )
        print(f"BLEU score: \t{bleu.score:.2f}")
        sys.stdout.flush()
        return

    for i in range(0, 10):
        n_to_prune = int(ceil(tot_n_heads * i / 10))
        to_prune_profile = [p for p, _ in sorted_profiles[:n_to_prune]]
        to_prune = parse_head_pruning_descriptors(
            to_prune_profile,
            reverse_descriptors=False
        )
        print(f"Evaluating following profile: \t{' '.join(to_prune_profile)}")
        # Apply pruning
        mask_heads(model, to_prune, args.transformer_mask_rescale)
        bleu = eval_bleu_score(
            model,
            task,
            task.dataset(args.valid_subset),
            beam=args.beam,
            replace_unk=args.replace_unk,
            lenpen=args.lenpen,
            buffer_size=100,
            use_cuda=torch.cuda.is_available() and not args.cpu,
            remove_bpe=args.remove_bpe,
            max_sentences=args.max_sentences,
            max_tokens=args.max_tokens,
            stop_early=not args.no_early_stop,
            normalize_scores=not args.unnormalized,
            min_len=args.min_len,
        )
        print(f"BLEU score: \t{bleu.score:.2f}")
        sys.stdout.flush()


def get_profile(importances, prefix):
    n_layers, n_heads = importances.size()
    return {
        f"{prefix}:{layer + 1}:{head + 1}": importances[layer, head]
        for layer in range(n_layers)
        for head in range(n_heads)
    }


def batch_head_importance(attn_variables, one_minus=False):
    # Retrieve context (shape bsz x nheads x L x dhead) and mask (shape bsz x L)
    ctx = attn_variables["context"]
    mask = attn_variables["out_mask"]
    # Reverse mask
    if mask is not None:
        mask = torch.eq(mask, 0.0).float()
    else:
        mask = torch.ones(ctx.size(0), ctx.size(2)).to(ctx.device)
    # Context gradient
    d_ctx = ctx.grad
    # Take the absolute dot
    importance = torch.einsum(
        "bhli,bhli->bhl",
        [ctx, d_ctx],
    )
    importance *= mask.unsqueeze(1)
    if one_minus:
        layer_importance = importance.sum(1, keepdim=True)
        importance = layer_importance - importance
    importance = importance.abs().sum(-1).sum(0).detach()
    denom = mask.sum()
    return importance, denom


def batch_head_stats(attn_variables, triu_masking=False):
    # Retrieve context (shape bsz x nheads x L x dhead), mask (shape bsz x L) and weights (shape bsz x nheads x L x l)
    ctx = attn_variables["context"].detach()
    in_mask = attn_variables["in_mask"]
    out_mask = attn_variables["out_mask"]
    p = attn_variables["weights"].detach()
    logp = torch.log(p)
    device = p.device
    # Results
    results = {}
    # Triu mask for self att
    triu_mask = torch.triu(p.new_ones((p.size(2), p.size(3))), 1).byte()
    # Reverse mask
    if in_mask is not None:
        in_mask = torch.eq(in_mask, 0.0).float()
    else:
        in_mask = torch.ones(p.size(0), p.size(3)).to(ctx.device)

    # Reverse mask
    if out_mask is not None:
        out_mask = torch.eq(out_mask, 0.0).float()
    else:
        out_mask = torch.ones(ctx.size(0), ctx.size(2)).to(ctx.device)

    def reduce_head(x):
        return (x * out_mask.unsqueeze(1)).sum(0).sum(-1).detach().cpu()

    def reduce_head_pairs(x):
        return (x * out_mask.unsqueeze(1).unsqueeze(1)).sum(0).sum(-1).detach().cpu()

    # p_mask has shape bsz x -1 x -1 x l
    p_mask = in_mask.unsqueeze(1).unsqueeze(1)
    # Entropy
    plogp = p * logp
    plogp[p == 0] = 0
    if triu_masking:
        plogp.masked_fill_(triu_mask.unsqueeze(0).unsqueeze(0), 0)
        # plogp.masked_fill_(p_mask.eq(0), 0)
    H_p = -plogp.sum(-1)
    results["entropy"] = reduce_head(H_p)
    # Cross entropy
    plogq = torch.einsum("bilk,bjlk->bijlk", [p, logp])
    plogq.masked_fill_((p == 0).unsqueeze(1), 0)
    if triu_masking:
        plogq.masked_fill_(triu_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0)
    H_pq = -plogq.sum(-1)
    # Avg KL (bsz x nhead x L)
    avg_KL = (H_pq - H_p.unsqueeze(2))
    results["kl"] = reduce_head_pairs(avg_KL)
    if (results["kl"] == float('inf')).any():
        print(triu_mask)
        print(p)
        print(avg_KL)
        exit()
    # avg output disagreement
    out = ctx / (torch.sqrt((ctx ** 2).sum(-1, keepdim=True)) + 1e-20)
    out_dis = torch.einsum("bild,bjld->bijl", [out, out]) / out.size(1) ** 2
    results["out_dis"] = reduce_head_pairs(out_dis)
    # avg attn disagreement
    attn_dis = torch.einsum("bilk,bjlk->bijl", [p, p]) / p.size(1) ** 2
    results["attn_dis"] = reduce_head_pairs(attn_dis)
    # Avg attn offset
    self_pos = torch.arange(p.size(2)).to(device).float().view(1, 1, -1)
    if triu_masking:
        masked_p = torch.where(triu_mask.unsqueeze(0).unsqueeze(0), -p.new_ones(p.size()), p)
    else:
        masked_p = p
    attn_pos = masked_p.argmax(dim=-1).float()
    attn_offset = self_pos - attn_pos
    results["attn_pos"] = reduce_head(attn_pos)
    results["attn_offset"] = reduce_head(attn_offset)
    # Avg attn offset
    attn_dist = torch.abs(attn_offset)
    results["attn_dist"] = reduce_head(attn_dist)
    # Avg squared attn offset
    results["attn_offset_sq"] = reduce_head(attn_offset ** 2)
    results["attn_pos_sq"] = reduce_head(attn_pos ** 2)
    # Denominator
    denom = out_mask.sum().detach().cpu().data
    return results, denom


def aggregate_stats(tot_stats, batch_stats):
    keys = [
        "entropy",
        "kl",
        "out_dis",
        "attn_dis",
        "attn_offset",
        "attn_pos",
        "attn_dist",
        "attn_offset_sq",
        "attn_pos_sq",
    ]
    for key in keys:
        if key not in tot_stats:
            tot_stats[key] = batch_stats[key]
        else:
            tot_stats[key] += batch_stats[key]


def estimate_head_importance(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus)
    if args.n_pruning_steps > 0:
        itr = islice(itr, args.n_pruning_steps)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )
    # Inititalize meters
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    # Initialize head importance scores
    encoder_layers = trainer.args.encoder_layers
    decoder_layers = trainer.args.decoder_layers
    encoder_heads = trainer.args.encoder_attention_heads
    decoder_heads = trainer.args.decoder_attention_heads
    device = next(trainer.model.parameters()).device
    head_importance = {
        "encoder_self": torch.zeros(encoder_layers, encoder_heads).to(device),
        "encoder_decoder": torch.zeros(decoder_layers, decoder_heads).to(device),
        "decoder_self": torch.zeros(decoder_layers, decoder_heads).to(device),
    }
    # Denominators to normalize properly
    denoms = {attn_type: val.clone()
              for attn_type, val in head_importance.items()}
    head_stats = {
        attn_type: [{} for _ in range(val.size(0))]
        for attn_type, val in head_importance.items()
    }
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # Compute gradients
        log_output = trainer.prune_step(samples)
        # Retrieve importance scores for the encoder
        for layer in range(encoder_layers):
            self_attn_variables = trainer.model.encoder.layers[layer].self_attn_variables
            importance, denom = batch_head_importance(
                self_attn_variables, one_minus=args.one_minus)
            head_importance["encoder_self"][layer] += importance
            denoms["encoder_self"][layer] += denom
            # Stats
            aggregate_stats(head_stats["encoder_self"][layer],
                            batch_head_stats(self_attn_variables)[0])
        # Retrieve importance scores for the decoder
        for layer in range(decoder_layers):
            # Self attention
            self_attn_variables = trainer.model.decoder.layers[layer].self_attn_variables
            importance, denom = batch_head_importance(
                self_attn_variables, one_minus=args.one_minus)
            head_importance["decoder_self"][layer] += importance
            denoms["decoder_self"][layer] += denom
            aggregate_stats(head_stats["decoder_self"][layer],
                            batch_head_stats(self_attn_variables, triu_masking=True)[0])
            # Encoder attention
            encoder_attn_variables = trainer.model.decoder.layers[layer].encoder_attn_variables
            importance, denom = batch_head_importance(
                encoder_attn_variables, one_minus=args.one_minus)
            head_importance["encoder_decoder"][layer] += importance
            denoms["encoder_decoder"][layer] += denom
            aggregate_stats(head_stats["encoder_decoder"][layer],
                            batch_head_stats(encoder_attn_variables)[0])
        # log mid-epoch stats
        stats = get_pruning_stats(trainer)
        for k, v in log_output.items():
            extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)
        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()
    # log end-of-epoch stats
    stats = get_pruning_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)
    # Normalize by type
    for attn_type in denoms:
        head_importance[attn_type] /= denoms[attn_type]
    # Normalize head stats
    for attn_type in denoms:
        for layer in range(len(head_stats[attn_type])):
            for key in head_stats[attn_type][layer]:
                head_stats[attn_type][layer][key] /= denoms[attn_type].mean().cpu()
    # Normalize by layer
    if args.normalize_by_layer:
        for layer in range(encoder_layers):
            for attn_type, importance in head_importance.items():
                head_importance[attn_type][layer] /= torch.sqrt(
                    torch.sum(importance[layer] ** 2))
    return {k: v.cpu() for k, v in head_importance.items()}, head_stats


def get_pruning_stats(trainer):
    stats = collections.OrderedDict()
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = round(trainer.get_meter('train_wall').sum)
    return stats


def add_pruning_args(parser):
    group = parser.add_argument_group('Pruning')
    group.add_argument('--n-pruning-steps', default=0, type=int, metavar='N',
                       help='Number of steps to estimate the head importance scores')
    group.add_argument("--normalize-by-layer", action="store_true")
    group.add_argument("--only-importance", action="store_true")
    group.add_argument("--one-minus", action="store_true")
    group.add_argument("--one-head", action="store_true")
    group.add_argument("--encoder-self-only", action="store_true", help="Only prune from the encoder self attention")
    group.add_argument("--encoder-decoder-only", action="store_true",
                       help="Only prune from the encoder decoder attention")
    group.add_argument("--decoder-self-only", action="store_true", help="Only prune from the decoder self attention")


if __name__ == '__main__':
    parser = options.get_training_parser()
    add_pruning_args(parser)
    options.add_pruning_args(parser)
    options.add_generation_args(parser)
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        raise NotImplementedError(
            "Pruning doesn't support multiprocessing yet")
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        raise NotImplementedError(
            "Pruning doesn't support multiprocessing yet")
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
