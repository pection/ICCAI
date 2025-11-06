"""
vqa.py modified for CLEVR.

max clevr question length exposed, instead of fixed to 20.

NOTE: Creating entirely new linear layer from scratch, rather than adjusting
an existing linear layer
"""

# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from tqdm import tqdm


from argparse import Namespace
from pretrain.qa_answer_table import load_lxmert_qa
from clevr_tasks.clevr_model import CLEVRModel
from vqa_framework.data_modules.generic_clevr_loader import (
    GenericCLEVRDataModule,
    ClevrFeature,
    ClevrDataLoader,
)
import json
import yaml
import math
from typing import Tuple
import numpy as np

# Must be last to overwrite any previous loading
from param import args

DataTuple = collections.namedtuple("DataTuple", "dataset loader evaluator")


#############################################################
# Datamodules
#############################################################


def create_clevr_atom_ho_exist(
    ho_idx: int, config: dict, dl_kwargs=None
) -> GenericCLEVRDataModule:
    """
    Create dataloaders for the minimal-iid and minimal-ood splits.

    - ho_idx is the index of the HOP to load
    - config is a dictionary containing paths to the .h5 data files
    (see ./src/clevr_tasks/symlink_config.yaml)
    - dl_kwargs are additional arguments for the dataloader
    """
    if ho_idx < 0 or ho_idx > 28:
        raise NotImplementedError

    if dl_kwargs is None:
        dl_kwargs = Namespace()

    # Mismatch, match, FILTERED Mismatch
    # NOTE: Only match (i.e., minimal-ood) and
    # FILTERED Mismatch (i.e., minimal-iid) should be used.
    val_files = [
        f"atom_ho_exist_mismatch_ho{ho_idx}.json",
        f"atom_ho_exist_match_ho{ho_idx}.json",
        f"atom_ho_exist_mismatch_ho{ho_idx}_filtered.json",
    ]

    ques_path = config["clevr_atom_ho_exists_ques_path"]
    img_features_path = config["clevr_atom_v1_img_features_path"]
    img_features = "user_%s_vg_frcnn.h5"

    return GenericCLEVRDataModule(
        data_dir=ques_path,
        dm_questions_name="val_v1.0_ques_dir",
        dm_images_name=None,
        dm_train_images=None,
        dm_train_scenes=None,
        dm_val_questions=val_files,
        dm_test_questions=None,
        user_image_features_val=[
            (
                os.path.join(img_features_path, img_features % "val"),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(img_features_path, img_features % "val"),
                "roi_features",
                "roi_features",
            ),
        ],
        image_features=ClevrFeature.USER_PROVIDED,
        pin_memory=True,
        drop_last_train=True,
        as_tokens=True,
        loader_num_workers=0 if args.debug else 2,
        **vars(dl_kwargs),
    )


def create_clevr_ho(
    ho_idx: int, config: dict, dl_kwargs=None, max_questions=None
) -> GenericCLEVRDataModule:
    """
    Create dataloaders for the train, complex-iid and complex-ood splits.

    - ho_idx is the index of the HOP to load
    - config is a dictionary containing paths to the .h5 data files
    (see ./src/clevr_tasks/symlink_config.yaml)
    - dl_kwargs are additional arguments for the dataloader
    - max_questions is the maximum number of questions to load in the train set
    """
    if ho_idx < 0 or ho_idx > 28:
        raise ValueError("Can only accept values of ho_idx from 0 to 28")

    if ho_idx <= 5:
        return _create_original_clevr_ho(
            ho_idx=ho_idx,
            config=config,
            dl_kwargs=dl_kwargs,
            max_questions=max_questions,
        )
    elif ho_idx <= 28:
        return _create_clevr_ho_additional(
            ho_idx=ho_idx,
            config=config,
            dl_kwargs=dl_kwargs,
            max_questions=max_questions,
        )
    else:
        raise NotImplementedError()


def _create_clevr_ho_additional(
    ho_idx: int, config: dict, dl_kwargs=None, max_questions=None
) -> GenericCLEVRDataModule:
    """
    NOTE: For the sake of *safety*, this function expects ho_idx between 6 and 28
    """
    if dl_kwargs is None:
        dl_kwargs = Namespace()

    assert 6 <= ho_idx <= 28
    ho_idx -= 6

    img_features_path = config["clevr_ho_add_img_features_path"]
    ques_path = config["clevr_ho_add_ques_path"]

    ho_choices = (
        (None, None, "sphere", "rubber"),
        (None, "brown", None, "rubber"),
        ("small", None, None, "rubber"),
        (None, "brown", "sphere", None),
        ("small", None, "sphere", None),
        ("small", "brown", None, None),
        (None, None, "cylinder", "metal"),
        (None, "red", None, "metal"),
        ("small", None, None, "metal"),
        (None, "red", "cylinder", None),
        ("small", None, "cylinder", None),
        ("small", "red", None, None),
        (None, None, "cube", "metal"),
        (None, "gray", None, "metal"),
        ("large", None, None, "metal"),
        (None, "gray", "cube", None),
        ("large", None, "cube", None),
        ("large", "gray", None, None),
        (None, None, "cube", "rubber"),
        (None, "purple", None, "rubber"),
        (None, "purple", "sphere", None),
        ("small", None, "cube", None),
        ("small", "purple", None, None),
    )

    # double check uniqueness
    assert len(ho_choices) == len(set(ho_choices))

    ho = ho_choices[ho_idx]
    e_type = args.experiment_name
    img_features = "user_%s_vg_frcnn.h5"

    # Note that unlike the original CLEVR-HO, this step is not required
    # as we did not created paired data for CLEVR-HO-Additional
    # print("NOTE: Limiting CLEVR-HO validation set to first 120K questions only -- i.e., first 80%")

    return GenericCLEVRDataModule(
        # Take just the first this many questions (all if it's None)
        dm_num_train_samples=1000 if args.debug else max_questions,
        dm_batch_size=args.batch_size,
        # 'data_dir' is the root directory where you've downloaded the
        # hdf5 files for the question data.
        # Specifically, for each possible ho+test_type, there is a corresponding
        # folder in s3://systematicity/wip_datasets_dir/held_out_CLEVR/hdf5_held_out_CLEVR/
        # called f'{test_type}_{str(ho)}_v1.0_ques'
        # All of the question data is only 8GB, so it's simplest to download the
        # entire s3://systematicity/wip_datasets_dir/held_out_CLEVR/hdf5_held_out_CLEVR/
        # folder, and point 'data_dir' to it.
        data_dir=ques_path,
        # This is a hard-coded value corresponding to the question folder name.
        # please do not change it.
        dm_questions_name=f"{e_type}_{str(ho)}_v1.0_ques",
        # Don't need scenes
        dm_train_scenes=None,
        dm_val_scenes=None,
        dm_test_scenes=None,
        # Paths to ResNet feature .hdf5 files
        image_features=ClevrFeature.USER_PROVIDED,
        user_image_features_train=[
            (
                os.path.join(
                    img_features_path, "train_v1.0_img", img_features % "train"
                ),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(
                    img_features_path, "train_v1.0_img", img_features % "train"
                ),
                "roi_features",
                "roi_features",
            ),
        ],
        user_image_features_val=[
            (
                os.path.join(
                    img_features_path, "val_iid_v1.0_img", img_features % "val"
                ),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(
                    img_features_path, "val_iid_v1.0_img", img_features % "val"
                ),
                "roi_features",
                "roi_features",
            ),
        ],
        user_image_features_test=[
            (
                os.path.join(img_features_path, "test_v1.0_img", img_features % "test"),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(img_features_path, "test_v1.0_img", img_features % "test"),
                "roi_features",
                "roi_features",
            ),
        ],
        # These would normally be the paths to question .json files.
        # Once the .json files are converted to .hdf5 files, the .json
        # files are no longer required. We still however need to pass in the
        # names of the .json files, as that determines the names of the .hdf5
        # files we should be reading in & using.
        # Do not change these values.
        dm_train_questions=f"CLEVR_ho_train_{str(ho)}_{e_type}_questions.json",
        dm_val_questions=[f"CLEVR_ho_val_iid_{str(ho)}_{e_type}_questions.json"],
        dm_test_questions=[f"CLEVR_held_out_questions_test_{str(ho)}.json"],
        # Not needed b/c we used user-provided features
        dm_images_name=None,
        pin_memory=True,
        drop_last_train=True,
        loader_num_workers=0 if args.debug else 2,  # Only for debugging
        dm_num_val_samples=1000 if args.debug else None,
        as_tokens=True,
        **vars(dl_kwargs),
    )


def _create_original_clevr_ho(
    ho_idx: int, config: dict, dl_kwargs=None, max_questions=None
) -> GenericCLEVRDataModule:
    assert 0 <= ho_idx <= 5
    if dl_kwargs is None:
        dl_kwargs = Namespace()

    img_features_path = config["clevr_ho_img_features_path"]
    ques_path = config["clevr_ho_ques_path"]

    ho_choices = (
        (None, None, "cylinder", "rubber"),
        (None, "cyan", None, "rubber"),
        ("large", None, None, "rubber"),
        (None, "cyan", "cylinder", None),
        ("large", None, "cylinder", None),
        ("large", "cyan", None, None),
    )

    assert len(ho_choices) == len(set(ho_choices))

    ho = ho_choices[ho_idx]
    e_type = args.experiment_name
    img_features = "user_%s_vg_frcnn.h5"

    print(
        "NOTE: Limiting CLEVR-HO validation set to first 120K questions only -- i.e., first 80%"
    )

    return GenericCLEVRDataModule(
        # Take just the first this many questions (all if it's None)
        dm_num_train_samples=1000 if args.debug else max_questions,
        dm_batch_size=args.batch_size,
        # 'data_dir' is the root directory where you've downloaded the
        # hdf5 files for the question data.
        # Specifically, for each possible ho+test_type, there is a corresponding
        # folder in s3://systematicity/wip_datasets_dir/held_out_CLEVR/hdf5_held_out_CLEVR/
        # called f'{test_type}_{str(ho)}_v1.0_ques'
        # All of the question data is only 8GB, so it's simplest to download the
        # entire s3://systematicity/wip_datasets_dir/held_out_CLEVR/hdf5_held_out_CLEVR/
        # folder, and point 'data_dir' to it.
        data_dir=ques_path,
        # This is a hard-coded value corresponding to the question folder name.
        # please do not change it.
        dm_questions_name=f"{e_type}_{str(ho)}_v1.0_ques",
        # Don't need scenes
        dm_train_scenes=None,
        dm_val_scenes=None,
        dm_test_scenes=None,
        # Paths to ResNet feature .hdf5 files
        image_features=ClevrFeature.USER_PROVIDED,
        user_image_features_train=[
            (
                os.path.join(
                    img_features_path, "train_v1.0_img", img_features % "train"
                ),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(
                    img_features_path, "train_v1.0_img", img_features % "train"
                ),
                "roi_features",
                "roi_features",
            ),
        ],
        user_image_features_val=[
            (
                os.path.join(
                    img_features_path, "val_iid_v1.0_img", img_features % "val"
                ),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(
                    img_features_path, "val_iid_v1.0_img", img_features % "val"
                ),
                "roi_features",
                "roi_features",
            ),
        ],
        user_image_features_test=[
            (
                os.path.join(img_features_path, "test_v1.0_img", img_features % "test"),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(img_features_path, "test_v1.0_img", img_features % "test"),
                "roi_features",
                "roi_features",
            ),
        ],
        # These would normally be the paths to question .json files.
        # Once the .json files are converted to .hdf5 files, the .json
        # files are no longer required. We still however need to pass in the
        # names of the .json files, as that determines the names of the .hdf5
        # files we should be reading in & using.
        # Do not change these values.
        dm_train_questions=f"CLEVR_ho_train_{str(ho)}_{e_type}_questions.json",
        dm_val_questions=[f"CLEVR_ho_val_iid_{str(ho)}_{e_type}_questions.json"],
        dm_test_questions=[f"CLEVR_held_out_questions_test_{str(ho)}.json"],
        # Not needed b/c we used user-provided features
        dm_images_name=None,
        pin_memory=True,
        drop_last_train=True,
        loader_num_workers=0 if args.debug else 2,  # Only for debugging
        dm_num_val_samples=(
            1000 if args.debug else 120000
        ),  # 120k to avoid paired examples
        as_tokens=True,
        **vars(dl_kwargs),
    )


def create_clevr(config: dict, dl_kwargs=None) -> GenericCLEVRDataModule:
    """
    Create dataloaders for the original CLEVR dataset.

    - config is a dictionary containing paths to the .h5 data files
    (see ./src/clevr_tasks/symlink_config.yaml)
    """
    if dl_kwargs is None:
        dl_kwargs = Namespace()

    img_features_path = config["clevr_img_features_path"]
    # ques_path is the parent directory of the img features path; see config.yaml for details
    # As to why this is, it has to do with passing CLEVR dataloader h5 files into
    # the GenericCLEVRLoader
    ques_path = os.path.abspath(os.path.join(img_features_path, os.pardir))
    img_features = "user_%s_vg_frcnn.h5"

    return GenericCLEVRDataModule(
        dm_batch_size=args.batch_size,
        # 'data_dir' is the root directory where you've downloaded the
        # hdf5 files for the question data.
        data_dir=ques_path,
        # This is a hard-coded value corresponding to the question folder name.
        # please do not change it.
        dm_questions_name=f"hdf5_clevr",
        # Don't need scenes
        dm_train_scenes=None,
        dm_val_scenes=None,
        dm_test_scenes=None,
        # Paths to ResNet feature .hdf5 files
        image_features=ClevrFeature.USER_PROVIDED,
        user_image_features_train=[
            (
                os.path.join(img_features_path, img_features % "train"),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(img_features_path, img_features % "train"),
                "roi_features",
                "roi_features",
            ),
        ],
        user_image_features_val=[
            (
                os.path.join(img_features_path, img_features % "val"),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(img_features_path, img_features % "val"),
                "roi_features",
                "roi_features",
            ),
        ],
        user_image_features_test=[
            (
                os.path.join(img_features_path, img_features % "test"),
                "normalized_boxes",
                "normalized_boxes",
            ),
            (
                os.path.join(img_features_path, img_features % "test"),
                "roi_features",
                "roi_features",
            ),
        ],
        # These would normally be the paths to question .json files.
        # Once the .json files are converted to .hdf5 files, the .json
        # files are no longer required. We still however need to pass in the
        # names of the .json files, as that determines the names of the .hdf5
        # files we should be reading in & using.
        # Do not change these values.
        dm_train_questions=f"questions.json",
        dm_val_questions=[f"questions.json"],
        dm_test_questions=None,
        # Not needed b/c we used user-provided features
        dm_images_name=None,
        pin_memory=True,
        drop_last_train=True,
        loader_num_workers=0 if args.debug else 2,  # Only for debugging
        dm_num_train_samples=1000 if args.debug else None,  # Only for debugging
        dm_num_val_samples=1000 if args.debug else None,  # Only for debugging
        as_tokens=True,
        **vars(dl_kwargs),
    )


# Figure out what the *index* of the last epoch will be
def get_last_epoch_idx(args, train_dl) -> int:
    last_epoch_index = args.epochs - 1
    if args.train_steps is not None:
        last_epoch_index = min(
            last_epoch_index, math.ceil(args.train_steps / len(train_dl)) - 1
        )
        last_epoch_index = max(last_epoch_index, 0)
    return last_epoch_index


#############################################################
# Model, Training, & Eval code
#############################################################


class CLEVR:
    """
    class containing LXMERT model, as well as training and evaluation procedures
    for CLEVR variants.
    """

    def __init__(
        self,
        max_clevr_length: int,
        dm: GenericCLEVRDataModule,
        device: str = "cuda",
        total_steps=None,
        ignore_num_loaders: bool = False,
    ):
        """
        PRECONDITION: datamodule {dm} has pin_memory=True, train_drop_last=True

        NOTE: the args.load_lxmert_qa seems to load a pretrained QA model, including
              answer head, and it seems to intelligently reuse those answers it can

        :param max_clevr_length:
        :param dm:
        """
        self.device = device

        if not dm.drop_last_train:
            raise ValueError()
        if not dm.pin_memory:
            raise ValueError()

        # Get information about the dataset required for creating the model
        dm.prepare_data()
        dm.setup()
        validation_dataloaders = dm.val_dataloader()
        if len(validation_dataloaders) != 1 and not ignore_num_loaders:
            raise NotImplementedError()
        self.vocab = dm.get_vocab()
        num_answers = len(self.vocab.vocab["answer_token_to_idx"])
        label2ans = self.vocab.vocab[
            "answer_idx_to_token"
        ]  # Dict int -> str should be OK

        # Model
        self.model = CLEVRModel(num_answers, max_clevr_length)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(
                args.load_lxmert_qa,
                self.model,
                label2ans=label2ans,
                use_clevr_answer_table=args.load_clevr_pretrained,
            )

        # Transfer model to GPU before apex.
        self.model = self.model.cuda()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if "bert" in args.optim:
            train_dataloader = dm.train_dataloader()
            batches_per_epoch = len(train_dataloader)
            t_total = int(batches_per_epoch * args.epochs)
            if total_steps is not None:
                t_total = min(total_steps, t_total)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam

            self.optim = BertAdam(
                list(self.model.parameters()), lr=args.lr, warmup=0.1, t_total=t_total
            )
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Half Precision
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level="O2"
            )

        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(
        self,
        train_dl: ClevrDataLoader,
        val_dl: ClevrDataLoader,
        start_step: int = 0,
        start_epoch: int = 0,
        best_valid: float = 0.0,
    ):
        iter_wrapper = (
            (lambda x: tqdm(x, total=len(train_dl))) if args.tqdm else (lambda x: x)
        )
        curr_step = start_step

        # Figure out what the *index* of the last epoch will be
        last_epoch_index = get_last_epoch_idx(args, train_dl)
        val_epochs = {}

        # If we're not validating each epoch, then figure out at which of these epochs we should be performing validation
        if args.num_val_evals > 0:
            val_epochs = {0, last_epoch_index}
            val_epochs.update(
                max(0, round((last_epoch_index + 1) * x / (args.num_val_evals - 1)) - 1)
                for x in range(1, args.num_val_evals)
            )

            # If there are more epochs than the number of times that we need
            # to run validation, then make sure we are actually validating
            # the required number of times (as the above code may have a collision
            # where two entries end up in the same place)
            # Add missing values by finding the largest gaps between consecutive epochs
            # where we run validation
            # Note this process is (and must be) deterministic as it had to be
            # reconstructed in the same way if you interrupt & resume the job.
            if last_epoch_index + 1 > args.num_val_evals:
                while len(val_epochs) < args.num_val_evals:
                    sorted_val_epochs = sorted(list(val_epochs))
                    gaps = [
                        sorted_val_epochs[i + 1] - sorted_val_epochs[i]
                        for i in range(len(val_epochs) - 1)
                    ]
                    assert max(gaps) >= 2
                    idx_to_fill = gaps.index(max(gaps))
                    val_epochs.add(sorted_val_epochs[idx_to_fill] + max(gaps) // 2)
                assert len(val_epochs) == args.num_val_evals

            # If there are less epochs than the number of times that we need to
            # run validation, then make sure we validate every epoch
            elif last_epoch_index + 1 <= args.num_val_evals:
                # In low-epoch  conditions, make sure we hit every epoch
                val_epochs.update(range(last_epoch_index + 1))
                assert len(val_epochs) == last_epoch_index + 1, val_epochs
            print(f"Validation epochs: {val_epochs}")

        interrupted_epoch = False
        for epoch in range(start_epoch, args.epochs):
            if args.train_steps is not None and curr_step >= args.train_steps:
                interrupted_epoch = True
                break

            quesid2ans = {}
            for batch in iter_wrapper(train_dl):
                if args.train_steps is not None and curr_step >= args.train_steps:
                    break

                # print('a')
                # i, (ques_id, feats, boxes, sent, target)
                sent = batch[0]
                ques_id = batch[1]
                all_img_feats = batch[2]
                feats = all_img_feats["roi_features"]
                boxes = all_img_feats["normalized_boxes"]
                feats, boxes = feats.to(self.device), boxes.to(self.device)
                target_strs = batch[4]

                # Convert target strings into required tensor on the correct device
                target = torch.zeros(
                    (len(target_strs), len(self.vocab.vocab["answer_token_to_idx"]))
                )
                for tar_i, target_str in enumerate(target_strs):
                    target[tar_i, self.vocab.answer_token_to_idx(target_str)] = 1
                target = target.to(self.device)

                # ques_id: batch_size long list of string question IDs
                # feats: batch_size X Num_objs X 2048 float Tensor
                # boxes: batch_size X Num_objs X 4 float Tensor (normalized to 0-1)
                # sent: List of string questions (no processing)
                # target: batch_size X answer_vocab_len float Tensor

                self.model.train()
                self.optim.zero_grad()

                # feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                if args.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                        )
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), 5.0)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                self.optim.step()

                score, label = logit.max(1)
                for qid, l, gt_tar in zip(ques_id, label.cpu().numpy(), target_strs):
                    ans = self.vocab.answer_idx_to_token(l)
                    # ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = {
                        "pred": ans,
                        "target": gt_tar,
                    }  # NOTE: Tweaked from original code (there it was just ans)

                curr_step += 1

            log_str = "\nEpoch %d: Train %0.2f\n" % (
                epoch,
                sum(x["pred"] == x["target"] for x in quesid2ans.values())
                / len(quesid2ans)
                * 100.0,
            )

            if args.num_val_evals < 0 or epoch in val_epochs or interrupted_epoch:
                valid_score, _ = self.evaluate(dataloader=val_dl)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.0)
            else:
                log_str += "Epoch %d: Valid SKIPPED\n" % epoch
            log_str += "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.0)

            print(log_str, end="")

            with open(self.output + "/log.log", "a") as f:
                f.write(log_str)
                f.flush()

            # If the job may be preempted, save a checkpoint that would allow us to
            # resume after preemption
            if args.preemptable:
                self.save_preempt(
                    next_epoch=epoch + 1, start_step=curr_step, best_val=best_valid
                )

        self.save("LAST")

    def predict(self, dataloader: ClevrDataLoader, dump=None) -> Tuple[dict, float]:
        """
        Predict the answers to questions in a data split.

        :param dataloader: ClevrDataLoader to evaluate on
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()

        quesid2ans = {}
        total_loss = 0
        for batch in dataloader:
            sent = batch[0]
            ques_id = batch[1]
            all_img_feats = batch[2]
            feats = all_img_feats["roi_features"]
            boxes = all_img_feats["normalized_boxes"]
            targets = batch[4]
            # ques_id, feats, boxes, sent = datum_tuple[:4]
            with torch.no_grad():
                # feats, boxes = feats.cuda(), boxes.cuda()
                feats, boxes = feats.to(self.device), boxes.to(self.device)
                logits = self.model(feats, boxes, sent).detach().cpu()

                # Convert target strings into required tensor on the correct device
                target_tensor = torch.zeros(
                    (len(targets), len(self.vocab.vocab["answer_token_to_idx"]))
                )
                for tar_i, target_str in enumerate(targets):
                    target_tensor[tar_i, self.vocab.answer_token_to_idx(target_str)] = 1

                assert logits.dim() == target_tensor.dim() == 2
                loss = self.bce_loss(logits, target_tensor)
                loss = loss * logits.size(1)
                total_loss += loss.cpu().item()

                score, label = logits.max(1)
                for qid, l, target, logit_arr in zip(
                    ques_id, label.cpu().numpy(), targets, logits
                ):
                    ans = self.vocab.answer_idx_to_token(l)
                    logit_arr = logit_arr.detach().cpu().numpy()
                    assert logit_arr.shape == (
                        len(self.vocab.vocab["answer_token_to_idx"]),
                    )
                    logit_arr = logit_arr.tolist()
                    quesid2ans[qid.item()] = {"pred": ans, "target": target}
                    # , 'logits': logit_arr} # NOTE: Tweaked from original code (there it was just ans)
                    #  Note: removed logits saving b/c it was too space consuming  -- 140MB for each CLEVR-HO val
        if dump is not None:
            # evaluator.dump_result(quesid2ans, dump)
            with open(dump, "w") as outfile:
                json.dump(quesid2ans, outfile, indent=2, sort_keys=True)
        return quesid2ans, total_loss / len(dataloader)

    def evaluate(self, dataloader: ClevrDataLoader, dump=None) -> Tuple[float, float]:
        """Evaluate all data in data_tuple."""
        quesid2answers, total_loss = self.predict(dataloader, dump)
        return (
            sum(x["pred"] == x["target"] for x in quesid2answers.values())
            / len(quesid2answers),
            total_loss,
        )

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def save_preempt(self, next_epoch: int, start_step: int, best_val: float):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "next_epoch": next_epoch,
                "start_step": start_step,
                "best_val": best_val,
            },
            os.path.join(self.output, "tmp_preempt_ckpt.pth"),
        )

        # Atomic swap in case of preemption during saving
        os.replace(
            os.path.join(self.output, "tmp_preempt_ckpt.pth"),
            os.path.join(self.output, "preempt_ckpt.pth"),
        )

    def load_preempt(self, path, terminate_if_done=False) -> Tuple[int, int, float]:
        print("Load model from %s" % path)
        preempt_dict = torch.load(path)

        if (
            terminate_if_done
            and args.train_steps is not None
            and preempt_dict["start_step"] >= args.train_steps
        ):
            print(
                f"Loaded checkpoint is already complete: {preempt_dict['start_step']} out of {args.train_steps} training steps. Terminating",
                flush=True,
            )
            exit(0)

        state_dict = preempt_dict["model_state_dict"]
        self.model.load_state_dict(state_dict)
        self.optim.load_state_dict(preempt_dict["optimizer_state_dict"])
        return (
            preempt_dict["next_epoch"],
            preempt_dict["start_step"],
            preempt_dict["best_val"],
        )


def main(
    dm,
    test_pred_name: str,
    test_perf_name: str,
    test_loss_name: str,
    val_pred_name: str,
    val_perf_name: str,
    val_loss_name: str,
):
    """
    :param dm:
    :return:
    """

    # Build Class
    # Determine correct max len for CLEVR; LXMERT just uses 20 for VQA, GQA & NLVR2
    # Based on quick check; ~50% of CLEVR are within this length; max is 49
    max_clevr_length = args.max_seq_length
    clevr = CLEVR(
        max_clevr_length=max_clevr_length,
        dm=dm,
        total_steps=args.train_steps,
        ignore_num_loaders=args.dataset == "clevr_atom_ho_exist",
    )

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        clevr.load(args.load)

    if args.load_lang_lxmert is not None:
        lang_ckpt = torch.load(args.load_lang_lxmert)
        clevr.model.lxrt_encoder.model.bert.embeddings.load_state_dict(
            lang_ckpt["model.bert.embeddings"]
        )
        clevr.model.lxrt_encoder.model.bert.encoder.layer.load_state_dict(
            lang_ckpt["model.bert.encoder.layer"]
        )
        print(f"Loaded language-encoder from {args.load_lang_lxmert}")
    if args.load_vis_lxmert is not None:
        vis_ckpt = torch.load(args.load_vis_lxmert)
        clevr.model.lxrt_encoder.model.bert.encoder.r_layers.load_state_dict(
            vis_ckpt["model.bert.encoder.r_layers"]
        )
        clevr.model.lxrt_encoder.model.bert.encoder.visn_fc.load_state_dict(
            vis_ckpt["model.bert.encoder.visn_fc"]
        )
        print(f"Loaded vision-encoder from {args.load_vis_lxmert}")

    # If the preemption flag is on, and we have an in-progress checkpoint, then
    # load it and continue training from there.
    preempt_ckpt_path = os.path.join(clevr.output, "preempt_ckpt.pth")
    if args.preemptable and os.path.exists(preempt_ckpt_path):
        next_epoch, start_step, best_val = clevr.load_preempt(
            preempt_ckpt_path, terminate_if_done=True
        )
        print(
            f"Found preempted checkpoint. Resuming from {preempt_ckpt_path}; next epoch: {next_epoch}, best_val: {best_val}"
        )

    else:
        next_epoch = 0
        start_step = 0
        best_val = 0.0

    # Also load the start epoch & best/worst.

    # Test or Train
    if args.test is not None:
        print("Start evaluation")
        args.fast = args.tiny = False  # Always loading all data in test
        if "test" in args.test:
            if args.dataset != "clevr_atom_ho_exist":
                result, total_loss = clevr.evaluate(
                    dataloader=dm.test_dataloader()[0],
                    dump=os.path.join(args.output, test_pred_name),
                )
            else:  # For clevr_atom_ho_exist, treat 2nd validation (atomic test of ho) as the test set.
                result, total_loss = clevr.evaluate(
                    dataloader=dm.val_dataloader()[1],
                    dump=os.path.join(args.output, test_pred_name),
                )
            print(f"Test Acc: {result}\nTest Loss: {total_loss}")
            with open(os.path.join(args.output, test_perf_name), "w") as outfile:
                print(result, file=outfile)
            with open(os.path.join(args.output, test_loss_name), "w") as outfile:
                print(total_loss, file=outfile)
        if "val" in args.test:
            # Extra evaluation for CLEVR-ATOM-HO to also eval filtered.
            if args.dataset == "clevr_atom_ho_exist":
                result, total_loss = clevr.evaluate(
                    dataloader=dm.val_dataloader()[2],  # Number 2 is filtered.
                    dump=os.path.join(args.output, "filtered_" + val_pred_name),
                )
                print(f"Filtered Val Acc: {result}\nFiltered Val Loss: {total_loss}")
                with open(
                    os.path.join(args.output, "filtered_" + val_perf_name), "w"
                ) as outfile:
                    print(result, file=outfile)
                with open(
                    os.path.join(args.output, "filtered_" + val_loss_name), "w"
                ) as outfile:
                    print(total_loss, file=outfile)
            # Normal evaluation unchanged
            result, total_loss = clevr.evaluate(
                dataloader=dm.val_dataloader()[0],
                dump=os.path.join(args.output, val_pred_name),
            )
            print(f"Val Acc: {result}\nVal Loss: {total_loss}")
            with open(os.path.join(args.output, val_perf_name), "w") as outfile:
                print(result, file=outfile)
            with open(os.path.join(args.output, val_loss_name), "w") as outfile:
                print(total_loss, file=outfile)
        elif "test" not in args.test:
            assert False, "No such test option for %s" % args.test
    else:
        print("Start training:")
        clevr.train(
            dm.train_dataloader(),
            dm.val_dataloader()[0],
            start_epoch=next_epoch,
            start_step=start_step,
            best_valid=best_val,
        )


if __name__ == "__main__":
    print(args)

    if args.train is not None:
        raise ValueError("--train is not used by CLEVR; please remove flag")
    if args.valid is not None:
        raise ValueError("--valid is not used by CLEVR; please remove flag")

    config = yaml.load(open(args.clevr_config, "r"), Loader=yaml.Loader)

    # Create datamodule
    # clevr_dm =
    if args.dataset == "clevr_ho":
        in_dm = create_clevr_ho(
            config=config, ho_idx=args.ho_idx, max_questions=args.max_questions
        )
        tpn = f"clevr_ho{args.ho_idx}_test_predict.json"
        tpern = f"clevr_ho{args.ho_idx}_test_perf.txt"
        tlossn = f"clevr_ho{args.ho_idx}_test_loss.txt"
        vpn = f"clevr_ho{args.ho_idx}_val_predict.json"
        vpern = f"clevr_ho{args.ho_idx}_val_perf.txt"
        vlossn = f"clevr_ho{args.ho_idx}_val_loss.txt"

    elif args.dataset == "clevr":
        if args.max_questions is not None:
            raise NotImplementedError
        in_dm = create_clevr(config=config)
        tpn = "clevr_test_predict.json"
        tpern = f"clevr_test_perf.txt"
        tlossn = f"clevr_test_loss.txt"
        vpn = f"clevr_val_predict_{args.ho_idx}.json"
        vpern = f"clevr_val_perf.txt"
        vlossn = f"clevr_val_loss.txt"

    elif args.dataset == "clevr_ho_v2":
        if args.max_questions is not None:
            raise NotImplementedError
        in_dm = create_clevr_ho_v2(config=config, ho_idx=args.ho_idx)
        tpn = f"clevr_ho{args.ho_idx}_v2_test_predict.json"
        tpern = f"clevr_ho{args.ho_idx}_v2_test_perf.txt"
        tlossn = f"clevr_ho{args.ho_idx}_v2_test_loss.txt"
        vpn = f"clevr_ho{args.ho_idx}_v2_val_predict.json"
        vpern = f"clevr_ho{args.ho_idx}_v2_val_perf.txt"
        vlossn = f"clevr_ho{args.ho_idx}_v2_val_loss.txt"

    elif args.dataset == "clevr_atom_ho_exist":
        if args.max_questions is not None:
            raise NotImplementedError
        if args.test is None:
            raise ValueError(
                "Can't train on CLEVR-ATOM-HO-exists. Dataset has no train set."
            )
        in_dm = create_clevr_atom_ho_exist(config=config, ho_idx=args.ho_idx)
        # test set is matching HO
        tpn = f"clevr_atom_ho{args.ho_idx}_exists_match_ho_predict.json"
        tpern = f"clevr_atom_ho{args.ho_idx}_exists_match_ho_perf.txt"
        tlossn = f"clevr_atom_ho{args.ho_idx}_exists_match_ho_loss.txt"
        # "val" set is no matching ho
        vpn = f"clevr_atom_ho{args.ho_idx}_exists_mismatch_ho_predict"
        vpern = f"clevr_atom_ho{args.ho_idx}_exists_mismatch_ho_perf.txt"
        vlossn = f"clevr_atom_ho{args.ho_idx}_exists_mismatch_ho_loss.txt"

    else:
        raise NotImplementedError()

    main(
        in_dm,
        test_pred_name=tpn,
        test_perf_name=tpern,
        test_loss_name=tlossn,
        val_pred_name=vpn,
        val_perf_name=vpern,
        val_loss_name=vlossn,
    )
