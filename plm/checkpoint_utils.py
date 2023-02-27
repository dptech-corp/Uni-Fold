# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import collections
import contextlib
import logging
import os
import re
import traceback
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch
from .file_io import PathManager


logger = logging.getLogger(__name__)

# async ckp copy
def ckp_copy_fun(src, checkpoints, end_of_epoch, args):
    has_copy = False
    can_delete = args.tmp_save_dir != args.save_dir
    for cp in checkpoints:
        try:
            if src != cp:
                logger.info("copy {} to {}".format(src, cp))
                has_copy = True
                PathManager.copy(src, cp, overwrite=True)
        except:
            logger.info("copy failed, please copy it manaully")

    try:
        if can_delete and has_copy and os.path.lexists(src):
            logger.info("removing temp file {} ...".format(src))
            os.remove(src)
        def remove_ckps(root_path):
            if not end_of_epoch and args.keep_interval_updates > 0:
                # remove old checkpoints; checkpoints are sorted in descending order
                checkpoints = checkpoint_paths(
                    root_path, pattern=r"checkpoint_\d+_(\d+)\.pt"
                )
                for old_chk in checkpoints[args.keep_interval_updates :]:
                    if os.path.lexists(old_chk):
                        os.remove(old_chk)
                        logger.info("removed {}".format(old_chk))

            if args.keep_last_epochs > 0:
                # remove old epoch checkpoints; checkpoints are sorted in descending order
                checkpoints = checkpoint_paths(root_path, pattern=r"checkpoint(\d+)\.pt")
                for old_chk in checkpoints[args.keep_last_epochs :]:
                    if os.path.lexists(old_chk):
                        os.remove(old_chk)
                        logger.info("removed {}".format(old_chk))

            if args.keep_best_checkpoints > 0:
                # only keep the best N checkpoints according to validation metric
                checkpoints = checkpoint_paths(
                    root_path,
                    pattern=r"checkpoint\.best_{}_(\d+\.?\d*)\.pt".format(
                        args.best_checkpoint_metric
                    ),
                )
                if not args.maximize_best_checkpoint_metric:
                    checkpoints = checkpoints[::-1]
                for old_chk in checkpoints[args.keep_best_checkpoints :]:
                    if os.path.lexists(old_chk):
                        os.remove(old_chk)
                        logger.info("removed {}".format(old_chk))
        remove_ckps(args.save_dir)
    except:
        logger.info("remove old ckps error")

    logger.info("finished async ckp saving.")

def save_checkpoint(args, trainer, epoch_itr, val_loss, ckp_copy_thread, do_save=True):
    from unicore import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if args.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if args.no_save or not do_save:
        return

    if not trainer.should_save_checkpoint_on_current_rank:
        return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if args.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch and not args.no_epoch_checkpoints and epoch % args.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        not end_of_epoch
        and args.save_interval_updates > 0
        and updates % args.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and args.keep_best_checkpoints > 0:
        checkpoint_conds[
            "checkpoint.best_{}_{:.2f}.pt".format(args.best_checkpoint_metric, val_loss)
        ] = not hasattr(save_checkpoint, "best") or is_better(
            val_loss, save_checkpoint.best
        )
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not args.no_last_checkpoints

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    tmp_checkpoints = [
        os.path.join(args.tmp_save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(tmp_checkpoints[0], extra_state)
        if ckp_copy_thread is not None:
            ckp_copy_thread.apply_async(ckp_copy_fun, (tmp_checkpoints[0], checkpoints, end_of_epoch, args))
        write_timer.stop()
        logger.info(
            "Saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                tmp_checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )


def load_checkpoint(args, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """

    reset_optimizer = args.reset_optimizer
    reset_lr_scheduler = args.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(args.optimizer_overrides)
    reset_meters = args.reset_meters
    reset_dataloader = args.reset_dataloader

    if args.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer"
            " or reset_lr_scheduler or reset_meters or reset_dataloader"
        )

    suffix = trainer.checkpoint_suffix
    if (
        args.restore_file == "checkpoint_last.pt"
    ):  # default value of restore_file is 'checkpoint_last.pt'
        checkpoint_path = os.path.join(
            args.save_dir, "checkpoint_last{}.pt".format(suffix)
        )
        first_launch = not PathManager.exists(checkpoint_path)
        if args.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint, e.g. restart from last checkpoint and etc.
            if PathManager.exists(args.finetune_from_model):
                checkpoint_path = args.finetune_from_model
                reset_optimizer = True
                reset_lr_scheduler = True
                reset_meters = True
                reset_dataloader = True
                logger.info(
                    f"loading pretrained model from {checkpoint_path}: "
                    "optimizer, lr scheduler, meters, dataloader will be reset"
                )
            else:
                raise ValueError(
                    f"--funetune-from-model {args.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = args.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = args.restore_file

    if args.restore_file != "checkpoint_last.pt" and args.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(args)
        )

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
    )

    if (
        extra_state is not None
        and "best" in extra_state
        and not reset_optimizer
        and not reset_meters
    ):
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], load_dataset=True, **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(
            epoch=1, load_dataset=True, **passthrough_args
        )

    trainer.init_total_train_steps(epoch_itr)
    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=True):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).
    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    local_path = PathManager.get_local_path(path)
    # The locally cached file returned by get_local_path() may be stale for
    # remote files that are periodically updated/overwritten (ex:
    # checkpoint_last.pt) - so we remove the local copy, sync across processes
    # (if needed), and then download a fresh copy.
    if local_path != path and PathManager.path_requires_pathmanager(path):
        try:
            os.remove(local_path)
        except FileNotFoundError:
            # With potentially multiple processes removing the same file, the
            # file being missing is benign (missing_ok isn't available until
            # Python 3.8).
            pass
        if load_on_all_ranks:
            torch.distributed.barrier()
        local_path = PathManager.get_local_path(path)

    with open(local_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)


    return state


def load_model_ensemble(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training
        task (dpaie.tasks.DpaieTask, optional): task to use for loading
    """
    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble, args, _task = load_model_ensemble_and_task(
        filenames,
        arg_overrides,
        task,
        strict,
        suffix,
        num_shards,
        state,
    )
    return ensemble, args


def load_model_ensemble_and_task(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    assert state is None or len(filenames) == 1

    from unicore import tasks

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    args = None
    for filename in filenames:
        orig_filename = filename
        assert num_shards > 0
        for shard_idx in range(num_shards):
            if num_shards == 1:
                filename = filename.replace(".pt", suffix + ".pt")
            else:
                filename = orig_filename[:-3] + f"_part{shard_idx}.pt"

            if not PathManager.exists(filename):
                raise IOError("Model file nost found: {}".format(filename))
            if state is None:
                state = load_checkpoint_to_cpu(filename, arg_overrides)
                # state_2 = load_checkpoint_to_cpu("/mnt/vepfs/projects/msa_pretrain/checkpoints/ESM_3B_multimer_finetune_multimer02_ppi04/checkpoint_21_80000.pt", arg_overrides)
            if "args" in state and state["args"] is not None:
                args = state["args"]
            # elif 'args' in state_2:
            #     args = state_2["args"]
            #     del state_2
            else:
                raise RuntimeError(
                    f"Neither args nor args exist in state keys = {state.keys()}"
                )

            if task is None:
                task = tasks.setup_task(args)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            # build model for ensemble
            model = task.build_model(args)
            def upgrade_state_dict(state_dict):
                """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
                prefixes = ["encoder."]# ["encoder.sentence_encoder.", "encoder."]
                pattern = re.compile("^" + "|".join(prefixes))
                state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
                
                prefixes = ["sentence_encoder.embed_tokens"]# ["encoder.sentence_encoder.", "encoder."]
                pattern = re.compile("^" + "|".join(prefixes))
                state_dict = {pattern.sub("embed_tokens", name): param for name, param in state_dict.items()}
                return state_dict
            
            state['model'] = upgrade_state_dict(state['model'])
            # print(state['model'])
            model.load_state_dict(state["model"], strict=strict, model_args=args)

            # reset state so it gets loaded for the next model in ensemble
            state = None

        ensemble.append(model)
    return ensemble, args, task


def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt"):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(obj, filename):
    if PathManager.supports_rename(filename):
        # do atomic save
        with PathManager.open(filename + ".tmp", "wb") as f:
            _torch_persistent_save(obj, f)
        PathManager.rename(filename + ".tmp", filename)
    else:
        # fallback to non-atomic save
        with PathManager.open(filename, "wb") as f:
            _torch_persistent_save(obj, f)


def _torch_persistent_save(obj, f):
    if isinstance(f, str):
        with PathManager.open(f, "wb") as h:
            torch_persistent_save(obj, h)
        return
    for i in range(3):
        try:
            return torch.save(obj, f)
        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())


def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning(
            "Unable to access checkpoint save directory: {}".format(save_dir)
        )
        raise e
    else:
        os.remove(temp_file_path)
