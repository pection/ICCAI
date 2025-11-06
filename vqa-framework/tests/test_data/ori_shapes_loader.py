import pytorch_lightning as pl
import os
from vqa_framework.utils.misc import checkout_repo
import shutil
import vqa_framework.data_modules.shapes_scripts as scripts
from argparse import Namespace
from typing import Optional, List, Union
import logging
from vqa_framework.utils.misc import add_init_args
from typing import Dict, Any

# Code re-used from CLOSURE paper
from vqa_framework.vr.utils import load_vocab
from vqa_framework.vr.data import ClevrDataLoader

from vqa_framework.utils.vocab import ClosureVocab

"""
Original SHAPES data loader, back from the dead.
"""


class OriSHAPESDataModule(pl.LightningDataModule):
    """
    Data module for downloading the SHAPES dataset, adding ground-truth symbolic
    representations of the images, and storing images into HDF5 format.
    ==== Public Attributes ====
    <self.vocab>: Only initialized after setup.
    """

    vocab: Optional[ClosureVocab]

    # https://openaccess.thecvf.com/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf
    # SHAPES dataset from Andreas NMN paper  (2016)

    GIT_URL: str = "https://github.com/ronghanghu/n2nmn.git"
    GIT_COMMIT: str = "ccbca1c623f3e208e7cde8d57b00e32dfb7c3c14"
    LOCAL_NAME = "hdf5_shapes"
    ORI_NAME = "ori_shapes"

    @staticmethod
    def add_model_specific_args(
        parent_parser,
        prefix: str = "",
        postfix: str = "",
        defaults: Dict[str, Any] = {},
    ):
        parser = parent_parser.add_argument_group("SHAPES_DM")
        add_init_args(
            parser,
            initializer=OriSHAPESDataModule.__init__,
            exclusions=["data_dir", "tmp_dir"],
            prefix=prefix,
            postfix=postfix,
            defaults=defaults,
        )
        return parent_parser

    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        tmp_dir: Union[str, os.PathLike],
        test_disable_download_process: bool = True,
        dm_batch_size: int = 64,
        dm_shuffle_train_data: bool = True,
        dm_question_families: Optional[List[int]] = None,
        dm_num_train_samples: Optional[int] = None,
        loader_num_workers: int = 4,
        dm_percent_of_data_for_training: float = 1,
        val_batch_size: int = 512,
        dm_num_val_samples: Optional[int] = None,
        dm_num_test_samples: Optional[int] = None,
        **kwargs,
    ):
        """
        <data_dir>: Where to store original SHAPES dataset, and processed
                    equivalent
        <tmp_dir>: Where to clone repo containing SHAPES (repo is deleted after)
        <dm_batch_size>: Train-time batch size
        <dm_question_families>: List of what question indices should be included in training/validation
        <dm_percent_of_data_for_training>: Reduce the total amount of training data
                                        to a random subset containing this
                                        percent of the data
        <dm_num_train_samples>: Truncate training dataset to at most this many
                             samples or None for all. NOTE: can be combined with
                             <dm_percent_of_data_for_training>; will truncate this
                             random subset to at most <dm_num_train_samples>
                             examples.
        <loader_num_workers>: Same argument for PyTorch DataLoader; 0 for main
                              process loading data
        Preconditions:
        tmp dir cannot contain file/dir called 'n2nmn'
        Following N2NMN code, all SHAPES train splits (large, med, small, tiny)
        are merged into a single train split.
        """
        super().__init__()
        # Record data loader details as hyperparameters, excluding data
        # directory paths
        self.save_hyperparameters(
            ignore=["data_dir", "tmp_dir", "loader_num_workers", "val_batch_size"]
        )

        self.test_disable_download_process = test_disable_download_process

        self.data_dir = data_dir
        self.tmp_dir = tmp_dir
        self.batch_size = dm_batch_size
        self.shuffle_train_data = dm_shuffle_train_data
        self.question_families = dm_question_families
        self.num_train_samples = dm_num_train_samples
        self.loader_num_workers = loader_num_workers
        self.percent_of_data_for_training = dm_percent_of_data_for_training
        self.val_batch_size = val_batch_size
        self.num_val_samples = dm_num_val_samples
        self.num_test_samples = dm_num_test_samples
        self.vocab = None

    def get_vocab(self):
        self.prepare_data()
        self.setup()
        return self.vocab

    def prepare_data(self) -> None:
        # prepare_data is called from a single process (e.g. GPU 0).
        # Do not use it to assign state (self.x = y).

        # Download the raw SHAPES dataset and copy to
        # <self.data_dir/ori_shapes
        # if not already present

        if self.test_disable_download_process:  # Refuse to download anythin
            return

        ori_dir = os.path.join(self.data_dir, self.ORI_NAME)
        if not os.path.exists(ori_dir):
            logging.info("SHAPES DATALOADER: Downloading SHAPES")
            checkout_repo(self.GIT_URL, self.tmp_dir, "n2nmn", self.GIT_COMMIT)
            shutil.move(
                os.path.join(self.tmp_dir, "n2nmn", "exp_shapes", "shapes_dataset"),
                ori_dir,
            )
            shutil.rmtree(os.path.join(self.tmp_dir, "n2nmn"))
        else:
            logging.warning(
                "SHAPES DATALOADER: Repository already downloaded, skipping"
            )

        # Determine symbolic representations/convert images to HDF5 file if not
        # already present
        target_dir = os.path.join(self.data_dir, self.LOCAL_NAME)
        if not os.path.exists(target_dir):
            logging.info("SHAPES DATALOADER: Processing SHAPES")
            os.makedirs(target_dir)

            # Create HDF5 files for images; note files have no additional processing.
            scripts.extract_feats(
                Namespace(input_dir=ori_dir, output_dir=target_dir, max_images=None)
            )

            # Convert SHAPES questions to the CLEVR .json format
            # also records question family #, puts program into reversh polish notation
            scripts.q_to_json(
                Namespace(input_dir=ori_dir, output_dir=target_dir, max_images=None)
            )

            # Preprocess train questions, create vocabulary
            scripts.preprocess_questions(
                Namespace(
                    input_questions_json=[
                        os.path.join(target_dir, f"SHAPES_train_questions.json")
                    ],
                    output_h5_file=os.path.join(target_dir, f"train_questions.h5"),
                    output_vocab_json=os.path.join(target_dir, "vocab.json"),
                    # defaults
                    mode="prefix",
                    input_vocab_json="",
                    expand_vocab=0,
                    unk_threshold=1,
                    encode_unk=0,
                )
            )

            # Preprocess val/test questions using train vocab.
            for split in ["val", "test"]:
                scripts.preprocess_questions(
                    Namespace(
                        input_questions_json=[
                            os.path.join(target_dir, f"SHAPES_{split}_questions.json")
                        ],
                        output_h5_file=os.path.join(
                            target_dir, f"{split}_questions.h5"
                        ),
                        input_vocab_json=os.path.join(target_dir, "vocab.json"),
                        # defaults
                        mode="prefix",
                        output_vocab_json="",
                        expand_vocab=0,
                        unk_threshold=1,
                        encode_unk=0,
                    )
                )

            # Generate scene graphs for symbolic execution during testing:
            # python3 shapes/shapes_im_to_json.py \
            #   --h5_dir ../data2/SHAPES/exp_shapes/shapes_h5/
            scripts.im_to_json(Namespace(h5_dir=target_dir, max_images=None))

        else:
            logging.warning(
                "SHAPES DATALOADER: Processed SHAPES dataset already exists, skipping"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        # stage is fit, validate, test, or None (equal to all)
        # setup is called from every process. Setting state here is okay.
        # ? 'fit' = train + val splits

        # Based on deprecation warning, modify setup so it only runs if it
        # hasn't alreaady
        if self.vocab is None:
            vocab_json = os.path.join(self.data_dir, self.LOCAL_NAME, "vocab.json")
            self.vocab = ClosureVocab(load_vocab(vocab_json))

    def train_dataloader(self):
        train_question_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "train_questions.h5"
        )
        train_features_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "train_features.h5"
        )
        train_scenes = os.path.join(self.data_dir, self.LOCAL_NAME, "train_scenes.json")
        train_loader_kwargs = {
            "question_h5": train_question_h5,
            "feature_h5": train_features_h5,  # if features_needed else None,
            "scene_path": train_scenes,  # if scenes_needed else None,
            "load_features": True,  # args.load_features,
            "vocab": self.vocab,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle_train_data,
            "question_families": self.question_families,
            "max_samples": self.num_train_samples,
            "num_workers": self.loader_num_workers,
            "percent_of_data": self.percent_of_data_for_training,
            "oversample": None,  # args.oversample,
            "oversample_shift": None,  # args.oversample_shift
        }
        return ClevrDataLoader(**train_loader_kwargs)

    def val_dataloader(self):
        val_question_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_questions.h5"
        )
        val_features_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_features.h5"
        )
        val_scenes = os.path.join(self.data_dir, self.LOCAL_NAME, "val_scenes.json")
        val_loader_kwargs = {
            "question_h5": val_question_h5,
            "feature_h5": val_features_h5,  # if features_needed else None,
            "scene_path": val_scenes,  # if scenes_needed else None,
            "load_features": True,  # args.load_features,
            "vocab": self.vocab,
            "batch_size": self.val_batch_size,
            "question_families": self.question_families,
            "max_samples": self.num_val_samples,
            "num_workers": self.loader_num_workers,
        }
        return ClevrDataLoader(**val_loader_kwargs)

    def test_dataloader(self):
        input_question_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "test_questions.h5"
        )
        input_features_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "test_features.h5"
        )
        input_scenes = os.path.join(self.data_dir, self.LOCAL_NAME, "test_scenes.json")

        loader_kwargs = {
            "question_h5": input_question_h5,
            "feature_h5": input_features_h5,
            "scene_path": input_scenes,
            # if isinstance(ee, ClevrExecutor) else None,
            "vocab": self.vocab,
            "batch_size": self.val_batch_size,
        }
        if self.num_test_samples is not None and self.num_test_samples > 0:
            loader_kwargs["max_samples"] = self.num_test_samples
        if self.question_families:
            loader_kwargs["question_families"] = self.question_families
        return ClevrDataLoader(**loader_kwargs)
