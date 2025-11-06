import pytorch_lightning as pl
import os
import wget
import shutil
from argparse import Namespace
from typing import Optional, List, Union, Dict, Any, Tuple
import logging
from vqa_framework.utils.misc import add_init_args
from vqa_framework.utils.vocab import ClosureVocab

from vqa_framework.resources.vocabs import CLEVR_VOCAB

# Code re-used from CLOSURE paper
from vqa_framework.vr.utils import load_vocab
from vqa_framework.vr.data import ClevrDataLoader
import vqa_framework.data_modules.clevr_scripts as scripts
from vqa_framework.data_modules.clevr_im_scripts.CLEVR_im_to_hdf5 import (
    main as clevr_ims_to_h5,
)
from enum import Enum
import json


class ClevrFeature(Enum):
    IEP_RESNET = 1
    IMAGES = 2
    USER_PROVIDED = 3


class CLEVRDataModule(pl.LightningDataModule):
    """
    Data module for downloading the CLEVR dataset,
    and storing images into HDF5 format.

    ==== Public Attributes ====
    <self.vocab>: Only initialized after setup.

    """

    vocab: Optional[ClosureVocab]

    # https://cs.stanford.edu/people/jcjohns/clevr/
    # CLEVR dataset

    LOCAL_NAME = "hdf5_clevr"
    ORI_NAME = "ori_clevr"
    ZIP_FILENAME = "CLEVR_v1.0"  # "CLEVR_v1.0_no_images" for no images -- debugging.

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
            initializer=CLEVRDataModule.__init__,
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
        overwrite_CLEVR_loc: Optional[str] = None,
        image_features: Optional[ClevrFeature] = ClevrFeature.IEP_RESNET,
        as_tokens: bool = False,
        resize_images: tuple = None,
        user_image_features_train: Optional[List[Tuple[str, str, str]]] = None,
        user_image_features_val: Optional[List[Tuple[str, str, str]]] = None,
        user_image_features_test: Optional[List[Tuple[str, str, str]]] = None,
        **kwargs,
    ):
        """
        <data_dir>: Where to store original CLEVR dataset, and processed
                    equivalent. NOTE: if <overwrite_CLEVR_loc> is not None,
                    then we will NOT download CLEVR & save to tmp, we'll instead
                    look for an existing copy of the downloaded dataset at
                    <overwrite_CLEVR_loc>
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
        <overwrite_CLEVR_loc>: If not None, then we should look for an existing
                               copy of the original CLEVR dataset with images
                               at this path. Assuming the dataset hasn't been
                               renamed, then the final part of the path should
                               be "CLEVR_v1.0"
        <image_features>: What kind of image feature this dataloader should
                          provide. By default, it provides the ResNet features
                          used by IEP (i.e., TensorNMN).
                          Note that ClevrFeature.IEP_RESNET are the ResNet
                          feature used by IEP/Tensor-NMN, and ClevrFeature.IMAGES
                          are just the raw images.
                          Make <None> to disable loading of image features entirely
        <user_image_features_train>: (also _val and _test)
                                        Should be non-None if <image_features>
                                        is the value ClevrFeature.USER_PROVIDED;
                                        in this case, it should be a list
                                        of triples of strings [(file, file_key, name),].
                                        First is the
                                        path to an hdf5 file, second is the key
                                        within the hdf5 file, and last is the name
                                        of the feature. It should be the case
                                        that hdf5_file[{hdf5 key}][i] is the feature
                                        for the ith image. Indexing should be the
                                        same as the hdf5 files automatically created
                                        by this dataloader (easiest way is probably
                                        to read in the generated hdf5 file, extract
                                        features, and save into the new hdf5 file
                                        in the same order).

        Preconditions:
        tmp dir cannot contain file/dir called "CLEVR_v1.0.zip"

        Note: downloading the 18gb CLEVR may be _very_ slow (e.g., 2h).
        If you would rather do it manually, then create the directory
        <self.data_dir>/<self.ORI_NAME> and download the .zip file there with

        wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O CLEVR_v1.0.zip

        NOTE: Need 18 GB to download CLEVR .zip file (in tmp),
              Need 101GB to store CLEVR dataset (in data)
                 - additional 20 GB to store decompressed full dataset (in data),
                 - 81GB to storge processed

              To provide the images directly, we're currently using the same hdf5
              backend storage. This seems to really suck
                32GB for train, 7GB for val & test ea. Therefore:
                Additional 46GB to store just the images
        """
        super().__init__()
        # Record data loader details as hyperparameters, excluding data
        # directory paths
        self.save_hyperparameters(
            ignore=[
                "data_dir",
                "tmp_dir",
                "loader_num_workers",
                "val_batch_size",
                "overwrite_CLEVR_loc",
            ]
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
        self.overwrite_CLEVR_loc = overwrite_CLEVR_loc
        self.vocab = None
        self.ori_clevr_path = None
        self.type_of_image_features = image_features
        if image_features == ClevrFeature.IEP_RESNET:
            self.feature_suffix = "_features.h5"
        elif image_features == ClevrFeature.IMAGES:
            self.feature_suffix = "_ims.h5"
        else:
            assert (
                image_features is None or image_features == ClevrFeature.USER_PROVIDED
            )
            self.feature_suffix = None
        self.as_tokens = as_tokens
        self.resize_images = resize_images

        self.user_im_feat_train = user_image_features_train
        self.user_im_feat_val = user_image_features_val
        self.user_im_feat_test = user_image_features_test

    def get_vocab(self):
        self.prepare_data()
        self.setup()
        return self.vocab

    def prepare_data(self) -> None:
        # prepare_data is called from a single process (e.g. GPU 0).
        # Do not use it to assign state (self.x = y).

        # If temp dir already has the clevr zip then use it, and don't delete it.
        pre_existing_clevr_zip = False

        # Download the raw CLEVR dataset and extract to
        # <self.data_dir>/ori_clevr
        # if not already present, and if <overwrite_CLEVR_loc> is None
        # By the end of this block, ori_dir will be a path to root of decompressed
        # CLEVR dataset.
        if self.overwrite_CLEVR_loc is None:
            # Where the extracted dataset should ultimately be (note: unzipping should create 'CLEVR_v1.0'
            ori_dir = os.path.join(self.data_dir, self.ORI_NAME, "CLEVR_v1.0")
            ori_dir_parent = os.path.join(self.data_dir, self.ORI_NAME)

            # where to temporarily save .zip file
            zip_path = os.path.join(self.tmp_dir, f"{self.ZIP_FILENAME}.zip")
            if not os.path.exists(ori_dir):

                if not os.path.exists(zip_path):
                    logging.warning(
                        "CLEVR DATALOADER: Downloading CLEVR. This may be very slow (e.g., 2h)"
                    )
                    wget.download(
                        f"https://dl.fbaipublicfiles.com/clevr/{self.ZIP_FILENAME}.zip",
                        zip_path,
                    )
                else:
                    logging.info(
                        "CLEVR DATALOADER: CLEVR .zip file already exists, not re-downloading, and not deleting"
                    )
                    pre_existing_clevr_zip = True

                # unzip from the tmp location into the data directory
                logging.info("CLEVR DATALOADER: Extracting. This may take some time.")
                shutil.unpack_archive(filename=zip_path, extract_dir=ori_dir_parent)

                assert os.path.exists(ori_dir)

                # Delete zip file from tmp directory
                if not pre_existing_clevr_zip:
                    os.remove(zip_path)
            else:
                logging.info(
                    "CLEVR DATALOADER: Extracted CLEVR dataset already exists, not re-downloading and re-extracting"
                )
        else:  # user is claiming CLEVR already downloaded elsewhere
            ori_dir = self.overwrite_CLEVR_loc

            if os.path.split(ori_dir)[1] != self.ZIP_FILENAME:
                logging.warning(
                    f"CLEVR DATALOADER: Warning! <overwrite_CLEVR_loc> ends with something other than {self.ZIP_FILENAME}"
                )

            if not os.path.exists(ori_dir):
                logging.critical(
                    f"CLEVR DATALOADER: Aborting!!! No directory at {ori_dir}"
                )
                exit(-1)

            for subdir in ["images", "scenes", "questions"]:
                if not os.path.exists(os.path.join(ori_dir, subdir)):
                    logging.critical(
                        f"CLEVR DATALOADER: Aborting!!! No `{subdir}` subdirectory at {ori_dir}"
                    )
                    exit(-1)

        # Record where the original CLEVR dataset is
        self.ori_clevr_path = ori_dir

        # Extract image features & process text if not already done
        target_dir = os.path.join(self.data_dir, self.LOCAL_NAME)
        if not os.path.exists(target_dir):
            logging.warning("CLEVR DATALOADER: Processing CLEVR")
            os.makedirs(target_dir)

        for split in ["train", "val", "test"]:
            if self.feature_suffix is None:
                continue
            target_name = f"{split}{self.feature_suffix}"

            if not os.path.exists(os.path.join(target_dir, target_name)):
                # If we need IEP ResNet features, then extract them
                if self.type_of_image_features == ClevrFeature.IEP_RESNET:
                    logging.warning(
                        f"CLEVR DATALOADER: Extracting {split} image features"
                    )
                    # Create HDF5 files for images; note files have are extracted ResNet features.
                    scripts.extract_feats(
                        Namespace(
                            input_image_dir=os.path.join(ori_dir, "images", split),
                            output_h5_file=os.path.join(target_dir, target_name),
                            # defaults
                            max_images=None,
                            image_height=224,
                            image_width=224,
                            model="resnet101",
                            model_stage=3,
                            batch_size=128,
                        )
                    )
                else:  # Otherwise just dump the images into a hdf5 file directly
                    assert self.type_of_image_features == ClevrFeature.IMAGES
                    logging.warning(
                        f"CLEVR DATALOADER: Saving {split} images as hdf5 file"
                    )
                    clevr_ims_to_h5(
                        Namespace(
                            input_image_dir=os.path.join(ori_dir, "images", split),
                            output_h5_file=os.path.join(target_dir, target_name),
                            max_images=None,
                        )
                    )
            else:
                logging.info(
                    f"CLEVR DATALOADER: Skipping generation of {target_name}; file already exists"
                )

        if not (os.path.exists(os.path.join(target_dir, "vocab.json"))):
            # Save training vocab
            with open(os.path.join(target_dir, "vocab.json"), "w") as outfile:
                json.dump(CLEVR_VOCAB, outfile)
        else:
            # Confirm the vocab is OK
            with open(os.path.join(target_dir, "vocab.json"), "r") as infile:
                assert CLEVR_VOCAB == json.load(infile)

        # Preprocess val/test questions using train vocab.
        for split in ["val", "test", "train"]:
            if not os.path.exists(os.path.join(target_dir, f"{split}_questions.h5")):
                logging.warning(f"CLEVR DATALOADER: Processing {split} questions")
                scripts.preprocess_questions(
                    Namespace(
                        input_questions_json=[
                            os.path.join(
                                ori_dir, "questions", f"CLEVR_{split}_questions.json"
                            )
                        ],
                        output_h5_file=os.path.join(
                            target_dir, f"{split}_questions.h5"
                        ),
                        input_vocab_json=os.path.join(target_dir, "vocab.json"),
                        # defaults
                        mode="prefix",
                        q_family_shift=[],
                        output_vocab_json="",
                        expand_vocab=0,
                        unk_threshold=1,
                        encode_unk=0,
                    )
                )
            else:
                logging.info(
                    f"CLEVR DATALOADER: Skipping generation of {split}_questions.h5; file already exists"
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
        train_features_h5 = None
        if self.feature_suffix is not None:
            train_features_h5 = os.path.join(
                self.data_dir, self.LOCAL_NAME, f"train{self.feature_suffix}"
            )
        elif (
            self.user_im_feat_train is not None
        ):  # Either user-provided image features, or don't load them at all
            train_features_h5 = self.user_im_feat_train

        train_scenes = os.path.join(
            self.ori_clevr_path, "scenes", "CLEVR_train_scenes.json"
        )
        train_loader_kwargs = {
            "question_h5": train_question_h5,
            "feature_h5": train_features_h5,  # if features_needed else None,
            "scene_path": train_scenes,  # if scenes_needed else None,
            "load_features": False,  # args.load_features,
            "vocab": self.vocab,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle_train_data,
            "question_families": self.question_families,
            "max_samples": self.num_train_samples,
            "num_workers": self.loader_num_workers,
            "percent_of_data": self.percent_of_data_for_training,
            "oversample": None,  # args.oversample,
            "oversample_shift": None,  # args.oversample_shift
            "as_tokens": self.as_tokens,
            "resize_images": self.resize_images,
        }
        return ClevrDataLoader(**train_loader_kwargs)

    def val_dataloader(self):
        val_question_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "val_questions.h5"
        )
        val_features_h5 = None
        if self.feature_suffix is not None:
            val_features_h5 = os.path.join(
                self.data_dir, self.LOCAL_NAME, f"val{self.feature_suffix}"
            )
        elif (
            self.user_im_feat_val is not None
        ):  # Either user-provided image features, or don't load them at all
            val_features_h5 = self.user_im_feat_val

        val_scenes = os.path.join(
            self.ori_clevr_path, "scenes", "CLEVR_val_scenes.json"
        )
        val_loader_kwargs = {
            "question_h5": val_question_h5,
            "feature_h5": val_features_h5,  # if features_needed else None,
            "scene_path": val_scenes,  # if scenes_needed else None,
            "load_features": False,  # args.load_features,
            "vocab": self.vocab,
            "batch_size": self.val_batch_size,
            "question_families": self.question_families,
            "max_samples": self.num_val_samples,
            "num_workers": self.loader_num_workers,
            "as_tokens": self.as_tokens,
            "resize_images": self.resize_images,
        }
        return ClevrDataLoader(**val_loader_kwargs)

    def test_dataloader(self):
        input_question_h5 = os.path.join(
            self.data_dir, self.LOCAL_NAME, "test_questions.h5"
        )
        input_features_h5 = None
        if self.feature_suffix is not None:
            input_features_h5 = os.path.join(
                self.data_dir, self.LOCAL_NAME, f"test{self.feature_suffix}"
            )
        elif (
            self.user_im_feat_test is not None
        ):  # Either user-provided image features, or don't load them at all
            input_features_h5 = self.user_im_feat_test
        input_scenes = None  # Don't exist for CLEVR

        loader_kwargs = {
            "question_h5": input_question_h5,
            "feature_h5": input_features_h5,
            "scene_path": input_scenes,
            # if isinstance(ee, ClevrExecutor) else None,
            "vocab": self.vocab,
            "batch_size": self.val_batch_size,
            "as_tokens": self.as_tokens,
            "resize_images": self.resize_images,
            "num_workers": self.loader_num_workers,
        }
        if self.num_test_samples is not None and self.num_test_samples > 0:
            loader_kwargs["max_samples"] = self.num_test_samples
        if self.question_families:
            loader_kwargs["question_families"] = self.question_families
        return ClevrDataLoader(**loader_kwargs)


class CLEVRCoGenTDataModule(pl.LightningDataModule):
    # CoGenT also part of original CLEVR https://cs.stanford.edu/people/jcjohns/clevr/
    pass

    def __init__(self):
        super().__init__()
        raise NotImplementedError


# Quick main method that doesn't do anything other than setup CLEVR;
# should download the dataset and process it. This shouldn't need to be
# repeated in future.
if __name__ == "__main__":
    from vqa_framework.global_settings import comp_env_vars

    logging.basicConfig(level=20)
    dm = CLEVRDataModule(
        tmp_dir=comp_env_vars.TMP_DIR,
        data_dir=comp_env_vars.DATA_DIR,
        val_batch_size=2,
        dm_batch_size=2,
        image_features=ClevrFeature.IMAGES,
        as_tokens=True,
        resize_images=(320, 320),
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

    print("VAL", "=" * 30)
    loader = dm.val_dataloader()
    for item in loader:
        for i in range(len(item)):
            print(names[i])
            print(item[i])
        break

    print("TEST", "=" * 30)
    loader = dm.test_dataloader()
    for item in loader:
        for i in range(len(item)):
            print(names[i])
            print(item[i])
        break

    print("TRAIN", "=" * 30)
    loader = dm.train_dataloader()
    for item in loader:
        for i in range(len(item)):
            print(names[i])
            print(item[i])
        break
