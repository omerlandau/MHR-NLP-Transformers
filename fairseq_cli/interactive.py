#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import numpy as np

import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def parse_head_pruning_descriptors(
        descriptors,
        reverse_descriptors=False,
        n_heads=None
):
    """Returns a dictionary mapping layers to the set of heads to prune in
    this layer (for each kind of attention)"""
    to_prune = {
        "E": {},
        "A": {},
        "D": {},
    }
    for descriptor in descriptors:
        attn_type, layer, heads = descriptor.split(":")
        layer = int(layer) - 1
        heads = set(int(head) - 1 for head in heads.split(","))
        if layer not in to_prune[attn_type]:
            to_prune[attn_type][layer] = set()
        to_prune[attn_type][layer].update(heads)
    # Reverse
    if reverse_descriptors:
        if n_heads is None:
            raise ValueError("You need to specify the total number of heads")
        for attn_type in to_prune:
            for layer, heads in to_prune[attn_type].items():
                to_prune[attn_type][layer] = set([head for head in range(n_heads)
                                                  if head not in heads])
    return to_prune


def get_attn_layer(model, attn_type, layer):
    if attn_type == "E":
        return model.encoder.layers[layer].self_attn
    elif attn_type == "D":
        return model.decoder.layers[layer].self_attn
    elif attn_type == "A":
        return model.decoder.layers[layer].encoder_attn


def mask_heads(model, to_prune, rescale=False):
    for attn_type in to_prune:
        for layer, heads in to_prune[attn_type].items():
            attn_layer = get_attn_layer(model, attn_type, layer)
            attn_layer.mask_heads = heads
            attn_layer.mask_head_rescale = rescale
            attn_layer._head_mask = None

def make_result(src_str, hypos, align_dict, tgt_dict, nbest=1, remove_bpe=False, print_alignment=False):
    result = Translation(
        src_str='O\t{}'.format(src_str),
        hypos=[],
        pos_scores=[],
        alignments=[],
    )
    # Process top predictions
    for i, hypo in enumerate(hypos[:min(len(hypos), nbest)]):
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens=hypo['tokens'].int().cpu(),
            src_str=src_str,
            alignment=hypo['alignment'].int().cpu(
            ) if hypo['alignment'] is not None else None,
            align_dict=align_dict,
            tgt_dict=tgt_dict,
            remove_bpe=remove_bpe,
        )
        # Now all hypos
        result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
        result.pos_scores.append('P\t{}'.format(
            ' '.join(map(
                lambda x: '{:.4f}'.format(x),
                hypo['positional_scores'].tolist(),
            ))
        ))
        result.alignments.append(
            'A\t{}'.format(
                ' '.join(map(lambda x: str(utils.item(x)), alignment)))
            if print_alignment else None
        )
    return result


def process_batch(
    translator,
    batch,
    align_dict,
    tgt_dict,
    use_cuda=False,
    nbest=1,
    remove_bpe=False,
    print_alignment=False,
    max_len_a=0,
    max_len_b=200,
):
    tokens = batch.tokens
    lengths = batch.lengths

    if use_cuda:
        tokens = tokens.cuda()
        lengths = lengths.cuda()

    encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
    translations = translator.generate(
        encoder_input,
        maxlen=int(max_len_a * tokens.size(1) + max_len_b),
    )

    batch_results = [
        make_result(
            batch.srcs[i],
            t,
            align_dict,
            tgt_dict,
            nbest=nbest,
            remove_bpe=remove_bpe,
            print_alignment=print_alignment,
        ) for i, t in enumerate(translations)
    ]
    return batch_results

def translate_corpus(
        translator,
        task,
        input_feed=None,
        buffer_size=1,
        replace_unk=False,
        use_cuda=False,
        print_directly=False,
        nbest=1,
        remove_bpe=False,
        print_alignment=False,
        max_sentences=1,
        max_tokens=9999,
):
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in translator.models]
    )
    if input_feed is None:
        input_feed = sys.stdin

    if buffer_size > 1:
        print('| Sentence buffer size:', buffer_size)
    print('| Type the input sentence and press return:')
    all_results = []
    for inputs in buffered_read(buffer_size, input_feed):
        indices = []
        results = []
        for batch, batch_indices in make_batches(inputs, task, max_positions, max_sentences, max_tokens):
            indices.extend(batch_indices)
            results += process_batch(
                translator,
                batch,
                align_dict,
                copy.deepcopy(task.target_dictionary),
                use_cuda=use_cuda,
                nbest=nbest,
                remove_bpe=remove_bpe,
                print_alignment=print_alignment,
            )
        # Sort results
        results = [results[i] for i in np.argsort(indices)]
        # Print to stdout
        if print_directly:
            for result in results:
                print(result.src_str)
                for hypo, pos_scores, align in zip(result.hypos, result.pos_scores, result.alignments):
                    print(hypo)
                    print(pos_scores)
                    if align is not None:
                        print(align)
        all_results.extend(results)
    return all_results


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = task.inference_step(generator, models, sample)
            print("Guy comment - > inside interactive")
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print('H-{}\t{}\t{}'.format(id, score, hypo_str))
                # detokenized hypothesis
                print('D-{}\t{}\t{}'.format(id, score, detok_hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        # convert from base e to base 2
                        hypo['positional_scores'].div_(math.log(2)).tolist(),
                    ))
                ))
                if args.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(
                        id,
                        alignment_str
                    ))

        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == '__main__':
    cli_main()
