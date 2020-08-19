#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import argparse
import logging
import math
import os
import random
import sys
from typing import Callable, Optional
import time
import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
import json

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(threshold=5000)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(
        args,
        init_distributed=False,
        after_distributed_init_fn: Optional[
            Callable[[argparse.Namespace], argparse.Namespace]
        ] = None,
):
    utils.import_user_module(args)

    assert (
            args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"
    metrics.reset()

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu and not getattr(args, "tpu", False):
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)
        if after_distributed_init_fn:
            args = after_distributed_init_fn(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info(
        "model {}, criterion {}".format(args.arch, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.max_sentences
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    if args.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers
        xm.mark_step()

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    experiment_path = args.mhr_experiment  # path for experiment configuration
    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr, model, experiment_path)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, "data", "")),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


def tpu_data_loader(args, itr):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    xm.rendezvous("tpu_data_loader")  # wait for all workers
    xm.mark_step()
    device = utils.get_tpu_device(args)
    return iterators.CountingIterator(
        pl.ParallelLoader(itr, [device]).per_device_loader(device),
        start=getattr(itr, "n", 0),
        total=len(itr),
    )


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr, model, experiment_path):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = tpu_data_loader(args, itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    num_heads = args.decoder_attention_heads
    head_dim = args.decoder_embed_dim // num_heads
    with open(experiment_path, 'r') as f:
        swaps = json.load(f)

    mhr(model, swaps, head_dim, num_heads, epoch_itr.epoch)

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function("train_step-%d" % i):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:
            stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
            progress.log(stats, tag="train_inner", step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    do_save = (
                      args.save_interval_updates > 0
                      and num_updates > 0
                      and num_updates % args.save_interval_updates == 0
              ) or (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
    do_validate = (
                          (not end_of_epoch and do_save)  # validate during mid-epoch saves
                          or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
                  ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    max_update = args.max_update or math.inf
    should_stop = (
            should_stop_early(args, valid_losses[0])
            or trainer.get_num_updates() >= max_update
    )

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = tpu_data_loader(args, itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def distributed_main(
        i,
        args,
        start_rank=0,
        after_distributed_init_fn: Optional[
            Callable[[argparse.Namespace], argparse.Namespace]
        ] = None,
):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(
        args, init_distributed=True, after_distributed_init_fn=after_distributed_init_fn
    )


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                cli_main_helper(args)
    else:
        cli_main_helper(args)


def cli_main_helper(args):
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        if not getattr(args, "tpu", False):
            # fallback for single node with multiple GPUs
            assert args.distributed_world_size <= torch.cuda.device_count()
            port = random.randint(10000, 20000)
            args.distributed_init_method = "tcp://localhost:{port}".format(port=port)
            args.distributed_rank = None  # set based on device id
            torch.multiprocessing.spawn(
                fn=distributed_main, args=(args,), nprocs=args.distributed_world_size
            )
        else:
            import torch_xla.distributed.xla_multiprocessing as xmp

            torch.multiprocessing.set_sharing_strategy("file_system")
            xmp.spawn(
                fn=distributed_main, args=(args,), nprocs=8  # use all 8 TPU cores
            )
    else:
        # single GPU training
        main(args)


def get_parameter_names(model, src_layer, src_layer_module,
                        src_transformer_module, dst_layer, dst_layer_module,
                        dst_transformer_module):
    '''

    :param model: Our transformer model.
    :param src_layer: Source layer.
    :param src_layer_module: Source layer module. i.e - self_attn,encoder_attn etc.
    :param src_transformer_module: Encoder\Decoder
    :param dst_layer: Destination layer.
    :param dst_layer_module: Destination layer module. i.e - self_attn,encoder_attn etc.
    :param dst_transformer_module:  Encoder\Decoder
    :return: The q,k,v,out weights parameters names.
    '''
    model_parms_list = list(model.state_dict().keys())
    src_param_names = [str for str in model_parms_list if src_layer in str and
                       src_transformer_module in str and src_layer_module in str and "norm" not in str and "out_proj" not in str]
    dst_param_names = [str for str in model_parms_list if dst_layer in str
                       and dst_transformer_module in str
                       and dst_layer_module in str and "norm" not in str and "out_proj" not in str]
    return src_param_names, dst_param_names


def get_parameters(model, src_param_names, dst_param_names):
    '''

    :param model: Our transformer model.
    :param src_param_names: Source parameters names.
    :param dst_param_names: Destination parameters names.
    :return: Two dictionaries that hold the q,k,v,out weights parameters.
    '''
    src_parameters = {src_param_name: model.state_dict()[src_param_name] for src_param_name in src_param_names}
    dst_parameters = {dst_param_name: model.state_dict()[dst_param_name] for dst_param_name in dst_param_names}
    return src_parameters, dst_parameters


def mhr_single_head(model, head_dim, num_heads, src_parameters, dst_parameters, src_head, dst_head, src_layer,
                    dst_layer):
    '''
    :param model: Our transformer model.
    :param head_dim: Head's dimension.
    :param num_heads: Amount of heads in each MHA mechanism.
    :param src_parameters: Source q,k,v,out weights parameters.
    :param dst_parameters: Destination q,k,v,out weights parameters.
    :param src_head: Source head.
    :param dst_head: Destination head.
    :param src_layer: Source layer.
    :param dst_layer: Destination layer.
    :return: Nothing. Performs the parameter swapping between two heads.
    '''

    print(
        "Start swapping parameters of head {} in layer {} and head {} in layer {}".format(src_head, src_layer, dst_head,
                                                                                          dst_layer))

    print(num_heads)
    for s_key, d_key in zip(src_parameters.keys(), dst_parameters.keys()):
        with torch.no_grad():
            # one source parameter(holds all heads)
            # print("######## before #########")
            # print(d_key)
            # print(model.state_dict()[d_key][0:3,0:4])
            # print(model.state_dict()[d_key][0:3, 128:132])
            # print(model.state_dict()[d_key].size())
            # print(s_key)
            # print(model.state_dict()[s_key][0:3,0:4])
            # print(model.state_dict()[s_key][0:3, 128:132])
            # print(model.state_dict()[s_key].size())

            ms = model.state_dict()

            if ("bias" in s_key and "bias" in d_key):
                # all source bias's weights
                src_parameter = ms[s_key]
                # all dst bias's weights
                dst_parameter = ms[d_key]
                # getting only bias's weights which relates to a specific head computation
                src_head_parameter = src_parameter[src_head * head_dim:(src_head + 1) * head_dim].clone()
                dst_head_parameter = dst_parameter[dst_head * head_dim:(dst_head + 1) * head_dim].clone()
                # rotating bias's weights
                dst_parameter[dst_head * head_dim:(dst_head + 1) * head_dim] = src_head_parameter
                src_parameter[src_head * head_dim:(src_head + 1) * head_dim] = dst_head_parameter
                del src_parameter
                del dst_parameter
                torch.cuda.empty_cache()
            else:
                # all source layer heads
                src_parameter = ms[s_key]

                # all dst layer heads
                dst_parameter = ms[d_key]

                # Get specific head parameters

                src_head_parameter = src_parameter[src_head * head_dim:(src_head + 1) * head_dim, :].clone()
                dst_head_parameter = dst_parameter[dst_head * head_dim:(dst_head + 1) * head_dim, :].clone()
                dst_parameter[dst_head * head_dim:(dst_head + 1) * head_dim, :] = src_head_parameter
                src_parameter[src_head * head_dim:(src_head + 1) * head_dim, :] = dst_head_parameter

                del src_parameter
                del dst_parameter
                torch.cuda.empty_cache()

                # print("######## after ########")
                # print(d_key)
                # print(model.state_dict()[d_key][0:3, 0:4])
                # print(model.state_dict()[d_key][0:3, 128:132])
                # print(model.state_dict()[d_key].size())
                # print(s_key)
                # print(model.state_dict()[s_key][0:3, 0:4])
                # print(model.state_dict()[s_key][0:3, 128:132])
                # print(model.state_dict()[s_key].size())

    print(
        "Done swapping parameters for creation of head {} in layer {} and head {} in layer {}".format(src_head,
                                                                                                      src_layer,
                                                                                                      dst_head,
                                                                                                      dst_layer))


def mhr(model, swaps, head_dim, num_heads, num_epoch):
    '''

    :param model: Our transformer model.
    :param swaps: A specific experiment of parameter swapping.
    :param head_dim: Head's dimension.
    :param num_heads: Amount of heads in each MHA mechanism.
    :param num_epoch: string, a key for the epoch on which the swap should be made.
    :return: Nothing. Performs the experiments.
    '''

    try:
        s_epoch = swaps['{0}'.format(num_epoch)]
    except:
        return
    start = time.time()
    for s in s_epoch:
        src_layer = s['s_layer']
        src_head = s['s_head']
        src_layer_module = s['s_layer_module']
        src_transformer_module = s['s_transformer_module']
        dst_layer = s['d_layer']
        dst_head = s['d_head']
        dst_layer_module = s['d_layer_module']
        dst_transformer_module = s['d_transformer_module']
        print(
            "src_layer = {0}, src_head = {1}, src_lm = {2}, src_tm = {3}, dst_layer = {4}, dst_head = {5}, dst_lm = {6}, dst_tm = {7}".format(
                src_layer, src_head, src_layer_module, src_transformer_module, dst_layer, dst_head, dst_layer_module,
                dst_transformer_module))
        src_param_names, dst_param_names = get_parameter_names(model, src_layer, src_layer_module,
                                                               src_transformer_module, dst_layer,
                                                               dst_layer_module, dst_transformer_module)
        src_parameters, dst_parameters = get_parameters(model, src_param_names, dst_param_names)
        mhr_single_head(model, head_dim, num_heads, src_parameters, dst_parameters, src_head, dst_head,
                        src_layer,
                        dst_layer)
    end = time.time()
    print("The experiment swapping took {} minuets".format(str((end - start) / 60)))


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    # keep this last so that it's a symlink
    checkpoint_conds['checkpoint_last.pt'] = True

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn)
                   for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(
            args.save_dir, pattern=r'checkpoint(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    pure_restore_file = os.path.abspath(args.restore_file)
    if os.path.isfile(pure_restore_file):
        checkpoint_path = pure_restore_file
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None and not args.reset_optimizer:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


if __name__ == "__main__":
    cli_main()
