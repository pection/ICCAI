import pytorch_lightning as pl
import os
from vqa_framework.utils.misc import checkout_repo
import shutil
from vqa_framework.data_modules.shapes_scripts.extract_SHAPES_features import (
    process_shapes,
)
from vqa_framework.data_modules.shapes_scripts.syget_q_to_json import (
    main2 as main2_q_to_json,
)
from vqa_framework.data_modules.shapes_scripts.shapes_im_to_json import (
    main2 as main2_im_to_json,
)
from argparse import Namespace
from typing import Optional, List, Union
import logging
from vqa_framework.utils.misc import add_init_args
from typing import Dict, Any
import json

# Code re-used from CLOSURE paper
from vqa_framework.vr.utils import load_vocab
from vqa_framework.vr.data import ClevrDataLoader
from vqa_framework.utils.vocab import ClosureVocab
import vqa_framework.data_modules.shapes_scripts as scripts
from vqa_framework.resources.vocabs import SHAPES_VOCAB as vocab_SyGeT


class SHAPESSyGeTDataModule(pl.LightningDataModule):
    # https://openreview.net/pdf?id=Pd_oMxH8IlF
    # SHAPES-SyGeT from Iterated Learning paper (2021)
    # SHAPES *Sy*stematic*Ge*neralization*T*est
    vocab: Optional[ClosureVocab]

    # https://openaccess.thecvf.com/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf
    # SHAPES dataset from Andreas NMN paper  (2016)

    GIT_URL: str = "https://github.com/ankitkv/SHAPES-SyGeT"
    GIT_COMMIT: str = "97d9e3ab51b1a65903d542e26d8772283347512e"
    LOCAL_NAME = "hdf5_syget"
    ORI_NAME = "ori_syget"

    TRAIN_QUESTION_FAMILIES = (1, 2, 3, 4, 5, 6, 7)
    OOD_QUESTION_FAMILIES = (8, 9, 10, 11, 12)

    @staticmethod
    def add_model_specific_args(
        parent_parser,
        prefix: str = "",
        postfix: str = "",
        defaults: Dict[str, Any] = None,
    ):
        if defaults is None:
            defaults = {}
        parser = parent_parser.add_argument_group("SHAPES-SyGeT_DM")
        add_init_args(
            parser,
            initializer=SHAPESSyGeTDataModule.__init__,
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
        dm_batch_size: int = 64,
        dm_shuffle_train_data: bool = True,
        dm_question_families: Optional[List[int]] = None,
        dm_num_train_samples: Optional[int] = None,
        loader_num_workers: int = os.cpu_count(),
        dm_percent_of_data_for_training: float = 1,
        val_batch_size: int = 512,
        dm_num_val_samples: Optional[int] = None,
        dm_num_test_samples: Optional[int] = None,
        train_fam: List[int] = None,
        val_fams: List[List[int]] = None,
        test_fams: List[List[int]] = None,
        **kwargs,
    ):
        """
        <data_dir>: Where to store original SyGeT dataset, and processed
                    equivalent
        <tmp_dir>: Where to clone repo containing SyGeT (repo is deleted after)
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


        <train_fam>: Family for training; must be subset of 1-7. Data
                     from original train split is used. None for all families.
        <val_fams>: Families for each validation dataloader. Each sublist must
                   be subset of 1-7. Data from original VAL-IID split is used.
                   None for all families.
        <test_fams>: Families for each test dataloader. Each sublist must be
                    subset of 8-12. Data from original Val OOD split is used.
                    None for all families.

        Preconditions:
        tmp dir cannot contain file/dir called 'n2nmn'
        """
        super().__init__()
        # Record data loader details as hyperparameters, excluding data
        # directory paths
        self.save_hyperparameters(
            ignore=["data_dir", "tmp_dir", "loader_num_workers", "val_batch_size"]
        )

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

        # Check indices are OK
        assert train_fam is None or set(train_fam).issubset(
            set(self.TRAIN_QUESTION_FAMILIES)
        )
        self.train_fam = train_fam

        if val_fams is None:
            self.val_fams = [None]
        else:
            for fam_group in val_fams:
                assert set(fam_group).issubset(set(self.TRAIN_QUESTION_FAMILIES))
            self.val_fams = val_fams

        if test_fams is None:
            self.test_fams = [None]
        else:
            for fam_group in test_fams:
                assert set(fam_group).issubset(set(self.OOD_QUESTION_FAMILIES))
            self.test_fams = test_fams

    def get_vocab(self):
        self.prepare_data()
        self.setup()
        return self.vocab

    def prepare_data(self) -> None:
        # prepare_data is called from a single process (e.g. GPU 0).
        # Do not use it to assign state (self.x = y).

        # Download the raw SyGeT dataset and copy to
        # <self.data_dir>/ori_syget
        # if not already present
        ori_dir = os.path.join(self.data_dir, self.ORI_NAME)
        if not os.path.exists(ori_dir):
            logging.warning("SyGeT DATALOADER: Downloading SyGeT")
            checkout_repo(self.GIT_URL, self.tmp_dir, "syget", self.GIT_COMMIT)
            shutil.move(
                os.path.join(self.tmp_dir, "syget", "shapes_syget"),
                ori_dir,
            )
            shutil.rmtree(os.path.join(self.tmp_dir, "syget"))
        else:
            logging.info("SyGeT DATALOADER: Repository already downloaded, skipping")

        # Determine symbolic representations/convert images to HDF5 file if not
        # already present
        target_dir = os.path.join(self.data_dir, self.LOCAL_NAME)
        if not os.path.exists(target_dir):
            logging.warning("SyGeT DATALOADER: Processing SyGeT")
            os.makedirs(target_dir)

            # Create HDF5 files for images; note files have no additional processing.
            for name, components in [
                ("train", ["train"]),
                ("val_iid", ["val_iid"]),
                ("val_ood", ["val_ood"]),
            ]:
                args = Namespace(
                    input_dir=ori_dir, output_dir=target_dir, max_images=None
                )
                process_shapes(name=name, component_files=components, args=args)

            # Convert SHAPES questions to the CLEVR .json format
            # also records question family #, puts program into reversh polish notation
            for name, components in [
                ("train", ["train"]),
                ("val_iid", ["val_iid"]),
                ("val_ood", ["val_ood"]),
            ]:
                main2_q_to_json(
                    name=name,
                    component_files=components,
                    in_dir=ori_dir,
                    out_dir=target_dir,
                    max_ims=None,
                    dset_name="SyGeT",
                )

            # Save the vocab stored in utils:
            with open(os.path.join(target_dir, "vocab.json"), "w") as outfile:
                json.dump(vocab_SyGeT, outfile)

            # Preprocess train/val_iid/val_ood questions using fixed vocab from utils
            for split in ["train", "val_iid", "val_ood"]:
                scripts.preprocess_questions(
                    Namespace(
                        input_questions_json=[
                            os.path.join(target_dir, f"SyGeT_{split}_questions.json")
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
            for name in ["train", "val_ood", "val_iid"]:
                args = Namespace(h5_dir=target_dir, max_images=None)
                main2_im_to_json(name, args)

        else:
            logging.info(
                "SyGeT DATALOADER: Processed SyGeT dataset already exists, skipping"
            )

            # Check the vocab file matches what we're expecting
            with open(os.path.join(target_dir, "vocab.json"), "r") as infile:
                assert json.load(infile) == vocab_SyGeT

    def setup(self, stage: Optional[str] = None) -> None:
        # stage is fit, validate, test, or None (equal to all)
        # setup is called from every process. Setting state here is okay.
        # ? 'fit' = train + val splits

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
            "question_families": self.train_fam,
            "max_samples": self.num_train_samples,
            "num_workers": self.loader_num_workers,
            "percent_of_data": self.percent_of_data_for_training,
            "oversample": None,  # args.oversample,
            "oversample_shift": None,  # args.oversample_shift
        }
        return ClevrDataLoader(**train_loader_kwargs)

    def val_dataloader(self):
        val_loaders = []
        val_question_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_iid_questions.h5"
        )
        val_features_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_iid_features.h5"
        )
        val_scenes = os.path.join(self.data_dir, self.LOCAL_NAME, "val_iid_scenes.json")
        for val_fam in self.val_fams:
            val_loader_kwargs = {
                "question_h5": val_question_h5,
                "feature_h5": val_features_h5,  # if features_needed else None,
                "scene_path": val_scenes,  # if scenes_needed else None,
                "load_features": True,  # args.load_features,
                "vocab": self.vocab,
                "batch_size": self.val_batch_size,
                "question_families": val_fam,
                "max_samples": self.num_val_samples,
                "num_workers": self.loader_num_workers,
            }
            val_loaders.append(ClevrDataLoader(**val_loader_kwargs))

        return val_loaders

    def test_dataloader(self):
        test_loaders = []

        input_question_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_ood_questions.h5"
        )
        input_features_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_ood_features.h5"
        )
        input_scenes = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_ood_scenes.json"
        )

        for test_fam in self.test_fams:
            loader_kwargs = {
                "question_h5": input_question_h5,
                "feature_h5": input_features_h5,
                "scene_path": input_scenes,
                # if isinstance(ee, ClevrExecutor) else None,
                "vocab": self.vocab,
                "batch_size": self.val_batch_size,
                "num_workers": self.loader_num_workers,
                "question_families": test_fam,
            }
            if self.num_test_samples is not None and self.num_test_samples > 0:
                loader_kwargs["max_samples"] = self.num_test_samples
            test_loaders.append(ClevrDataLoader(**loader_kwargs))

        return test_loaders


# Quick main method that doesn't do anything other than setup CLEVR;
# should download the dataset and process it. This shouldn't need to be
# repeated in future.
if __name__ == "__main__":
    from vqa_framework.global_settings import comp_env_vars

    logging.basicConfig(level=20)
    dm = SHAPESSyGeTDataModule(
        tmp_dir=comp_env_vars.TMP_DIR,
        data_dir=comp_env_vars.DATA_DIR,
        val_batch_size=2,
        dm_batch_size=2,
        train_fam=[1, 2],
        val_fams=[[1], [2, 3]],
        test_fams=[[8, 9], [9]],
    )
    dm.prepare_data()
    dm.setup()
    vocab = dm.get_vocab()

    names = [
        "question",
        "program_id",
        "image",
        "scene_graph",
        "answers",
        "program",
        "ques_fam",
        "question_len",
        "program_len",
    ]

    for j, loader in enumerate(dm.val_dataloader()):
        print(f"VAL #{j}", "=" * 30)
        print(f"Len: {len(loader)}")
        for item in loader:
            for i in range(len(item)):
                print(names[i])
                print(item[i])

            program = item[5][0]
            program = " ".join(
                [
                    vocab.program_idx_to_token(x)
                    for x in program.cpu().detach().numpy().tolist()
                ]
            )
            print(program)

            break

    for j, loader in enumerate(dm.test_dataloader()):
        print(f"TEST #{j}", "=" * 30)
        print(f"Len: {len(loader)}")
        for item in loader:
            for i in range(len(item)):
                print(names[i])
                print(item[i])
            break

    print("TRAIN", "=" * 30)
    loader = dm.train_dataloader()
    print(f"Len: {len(loader)}")
    for item in loader:
        for i in range(len(item)):
            print(names[i])
            print(item[i])
        break
