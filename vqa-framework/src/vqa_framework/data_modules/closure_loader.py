import pytorch_lightning as pl
import os
import wget
import shutil
from argparse import Namespace
from typing import Optional, List, Union, Dict, Any, Tuple
import logging
from vqa_framework.utils.misc import add_init_args
from vqa_framework.utils.vocab import ClosureVocab

# Code re-used from CLOSURE paper
from vqa_framework.vr.data import ClevrDataLoader
import vqa_framework.data_modules.clevr_scripts as scripts
from enum import Enum

# Recycle CLEVR processing
from vqa_framework.data_modules.clevr_loader import CLEVRDataModule, ClevrFeature

import json


class ClosureFamily(Enum):
    """
    Which question family of the CLOSURE test set to use.
    """

    # <A> and <Q> are properties (e.g., shape, size)
    # <Z>,<C>,<M>,<S> are properties
    # <R> are spatial words

    # What is the <Q> of the XXX that is <R> the YYY and is the same <A> as the ZZZ
    AND_MAT_SPA = 1

    # There is another XXX that is the same <A> as the YYYY. Does it have the same <Q> as the ZZZ?
    COMPARE_MAT = 2

    # There is another XXX that is the same <A> as the YYY; does is have the same <Q> as the ZZZ [that is] <R> the WWW
    COMPARE_MAT_SPA = 3

    # Is there a XXX <R> the YYY that is the same <A> as ZZZ?
    EMBED_MAT_SPA = 4

    # Is there a XXX that is the same <A> as the YYY <R> the ZZZ
    EMBED_SPA_MAT = 5

    # How many things are [either] XXX or YYY that are the same <A> as the ZZZ
    OR_MAT = 6

    # How many things are [either] XXX [that are] <R> the YYY or ZZZ that are the same <A> as the WWW.
    OR_MAT_SPA = 7


class ClosureTestType(Enum):
    """
    Whether to provide the "test" questions (i.e., CLOSURE test set),
    or the "baseline" questions (the CLEVR questions most similar to the CLOSURE
    family)
    """

    BASELINE = 1
    TEST = 2


def check_ims_from_val(file: str):
    with open(file, "r") as infile:
        questions = json.load(infile)

    questions = questions["questions"]
    for q in questions:
        assert q["image_filename"] == f"CLEVR_val_{str(q['image_index']).zfill(6)}.png"


class ClosureDataModule(pl.LightningDataModule):
    """
    Data module for downloading the closure dataset.

    WARNING!!! Ignores the provided closure vocab file. Instead uses the same
    vocab as data_modules/clevr_loader

    ==== Public Attributes ====
    <self.vocab>: Only initialized after setup.

    """

    vocab: Optional[ClosureVocab]

    # https://cs.stanford.edu/people/jcjohns/clevr/
    # CLEVR dataset

    LOCAL_NAME = "hdf5_closure"
    CLEVR_NAME = "hdf5_clevr"  # NOTE: depends on CLEVR dataloader!
    ORI_NAME = "ori_closure"
    ZIP_FILENAME = "closure"

    @staticmethod
    def add_model_specific_args(
        parent_parser,
        prefix: str = "",
        postfix: str = "",
        defaults: Dict[str, Any] = None,
    ):
        if defaults is None:
            defaults = {}
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
        closure_test_question_fam: ClosureFamily,
        closure_test_type: ClosureTestType,
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
        overwrite_closure_loc: Optional[str] = None,
        image_features: Optional[ClevrFeature] = ClevrFeature.IEP_RESNET,
        as_tokens: bool = False,
        resize_images: tuple = None,
        user_image_features: Optional[List[Tuple[str, str, str]]] = None,
        **kwargs,
    ):
        """
        <closure_question_fam> What CLOSURE question family to provide in
                               the test set. NOTE: validation returns over all val families (CLOSURE is small so why not)
        <closure_test_type> What CLOSURE test type to provide at test time.
                            Either test (the true test set) or baseline (the most
                            similar CLEVR questions)
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
                               NOTE: *We NEED* CLEVR b/c closure dataset uses
                               images from CLEVR and does not come with them.
        <overwrite_closure_loc>: If not None, then we look for existing copy of
                                 CLOSURE dataset at this path.
        <image_features>: What kind of image feature this dataloader should
                          provide. By default, it provides the ResNet features
                          used by IEP (i.e., TensorNMN).
                          Note that ClevrFeature.IEP_RESNET are the ResNet
                          feature used by IEP/Tensor-NMN, and ClevrFeature.IMAGES
                          are just the raw images.
                          Make <None> to disable loading of image features entirely
        <as_tokens>: If true, return strings instead of token indices.
        <resize_images>: If provided, attempt to resize images to these dimensions (NOTE: not to be used with features)
        user_image_features>:
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
                                        by the CLEVR dataloader for the CLEVR
                                        validation split (as CLOSURE only uses
                                        images from the CLEVR validation split)
                                        (the easiest way is probably
                                        to read in the generated hdf5 file, extract
                                        features, and save into the new hdf5 file
                                        in the same order).

        Preconditions:
        tmp dir cannot contain file/dir called "CLEVR_v1.0.zip"

        Note: downloading the 18gb CLEVR may be _very_ slow (e.g., 2h).
        If you would rather do it manually, then create the directory
        <self.data_dir>/<self.ORI_NAME> and download the .zip file there with

        wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O CLEVR_v1.0.zip

        Need 18 GB to download CLEVR .zip file (in tmp),
        Need 101GB to store CLEVR dataset (in data)
         - additional 20 GB to store decompressed full dataset (in data),
         - 81GB to storge extracted ResNet features
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

        self.clevr_datamodule = CLEVRDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            overwrite_CLEVR_loc=overwrite_CLEVR_loc,
            image_features=image_features,
        )
        self.overwrite_closure_loc = overwrite_closure_loc

        self.as_tokens = as_tokens
        self.resize_images = resize_images

        self.closure_test_fam = closure_test_question_fam
        self.closure_test_type = closure_test_type

        self.user_im_feat_val = user_image_features

    def get_vocab(self):
        # Use CLEVR Vocab (need to be consistent to apply same models to both!)
        return self.clevr_datamodule.get_vocab()

    def prepare_data(self) -> None:
        self.clevr_datamodule.prepare_data()
        # prepare_data is called from a single process (e.g. GPU 0).
        # Do not use it to assign state (self.x = y).

        # If temp dir already has the clevr zip then use it, and don't delete it.
        pre_existing_closure_zip = False

        # Download the raw closure dataset and extract to
        # <self.data_dir>/ori_closure
        # if not already present, and if <overwrite_closure_loc> is None
        # By the end of this block, ori_dir will be a path to root of decompressed
        # closure dataset.
        if self.overwrite_closure_loc is None:
            # Where the extracted dataset should ultimately be
            ori_dir = os.path.join(self.data_dir, self.ORI_NAME, "closure")

            # where to temporarily save .zip file
            zip_path = os.path.join(self.tmp_dir, f"{self.ZIP_FILENAME}.zip")
            if not os.path.exists(ori_dir):

                # NOTE: Not totally sure this is working
                if not os.path.exists(zip_path):
                    logging.warning("CLOSURE DATALOADER: Downloading CLOSURE")
                    wget.download(
                        f"https://zenodo.org/record/3634090/files/closure.zip",
                        zip_path,
                    )
                else:
                    logging.info(
                        "CLOSURE DATALOADER: CLOSURE .zip file already exists, not re-downloading, and not deleting"
                    )
                    pre_existing_closure_zip = True

                # unzip from the tmp location into the data directory
                logging.info("CLOSURE DATALOADER: Extracting.")
                os.makedirs(ori_dir)
                shutil.unpack_archive(filename=zip_path, extract_dir=ori_dir)

                # Delete zip file from tmp directory
                if not pre_existing_closure_zip:
                    os.remove(zip_path)
            else:
                logging.info(
                    "CLOSURE DATALOADER: Extracted closure dataset already exists, not re-downloading and re-extracting"
                )
        else:  # user is claiming closure already downloaded elsewhere
            ori_dir = self.overwrite_closure_loc

            if not os.path.exists(ori_dir):
                logging.critical(
                    f"Closure DATALOADER: Aborting!!! No directory at {ori_dir}"
                )
                exit(-1)

            if not len(os.listdir(ori_dir)) == 23:
                logging.critical(
                    f"Closure DATALOADER: Aborting!!! Closure should contain 23 files, {len(os.listdir(ori_dir))} found."
                )
                exit(-1)

        # Record where the original closure dataset is
        self.ori_closure_path = ori_dir

        # process text if not already done
        target_dir = os.path.join(self.data_dir, self.LOCAL_NAME)
        if not os.path.exists(target_dir):
            logging.warning("Closure DATALOADER: Processing closure")
            os.makedirs(target_dir)

        vocab_path = os.path.join(
            self.data_dir, self.CLEVR_NAME, "vocab.json"
        )  # Use CLEVR vocab

        # Preprocess val/test questions using train vocab.
        for split in [
            "baseline",
            "test",
            "val",
        ]:
            for family in ClosureFamily:
                family = family.name.lower()
                if not os.path.exists(os.path.join(target_dir, f"{family}_{split}.h5")):

                    # Sanity check -- makes sure all images are from validation set
                    check_ims_from_val(os.path.join(ori_dir, f"{family}_{split}.json"))

                    logging.warning(
                        f"Closure DATALOADER: Processing {family} {split} questions"
                    )
                    scripts.preprocess_questions(
                        Namespace(
                            input_questions_json=[
                                os.path.join(ori_dir, f"{family}_{split}.json")
                            ],
                            output_h5_file=os.path.join(
                                target_dir, f"{family}_{split}.h5"
                            ),
                            input_vocab_json=vocab_path,
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
                        f"CLOSURE DATALOADER: Skipping generation of {family}_{split}.h5; file already exists"
                    )

    def setup(self, stage: Optional[str] = None) -> None:
        self.clevr_datamodule.setup()
        # stage is fit, validate, test, or None (equal to all)
        # setup is called from every process. Setting state here is okay.
        # ? 'fit' = train + val splits

        # Based on deprecation warning, modify setup so it only runs if it
        # hasn't alreaady
        if self.vocab is None:
            # Get vocab from CLEVR datamodule to ensure same models work on both
            self.vocab = self.get_vocab()

    def train_dataloader(self):
        raise NotImplementedError(
            "CLOSURE dataset released seems to be missing the fine tune set?"
        )

    def val_dataloader(self):

        validation_dataloaders = []

        for family in ClosureFamily:
            family = family.name.lower()
            input_question_h5 = os.path.join(
                self.data_dir,
                self.LOCAL_NAME,
                f"{family}_val.h5",
            )

            # Images are from CLEVR validation set.
            clevr_val_features_h5 = None
            if self.feature_suffix is not None:
                clevr_val_features_h5 = os.path.join(
                    self.data_dir, self.CLEVR_NAME, f"val{self.feature_suffix}"
                )
            elif (
                self.user_im_feat_val is not None
            ):  # Either user-provided image features, or don't load them at all
                clevr_val_features_h5 = self.user_im_feat_val

            clevr_val_scenes = os.path.join(
                self.clevr_datamodule.ori_clevr_path, "scenes", "CLEVR_val_scenes.json"
            )

            loader_kwargs = {
                "question_h5": input_question_h5,
                "feature_h5": clevr_val_features_h5,
                "scene_path": clevr_val_scenes,
                # if isinstance(ee, ClevrExecutor) else None,
                "vocab": self.vocab,
                "batch_size": self.val_batch_size,
                "num_workers": self.loader_num_workers,
                "as_tokens": self.as_tokens,
                "resize_images": self.resize_images,
            }
            if self.num_val_samples is not None and self.num_val_samples > 0:
                loader_kwargs["max_samples"] = self.num_val_samples
            if self.question_families:
                loader_kwargs["question_families"] = self.question_families

            validation_dataloaders.append(ClevrDataLoader(**loader_kwargs))

        return validation_dataloaders

    def test_dataloader(self):
        split = self.closure_test_type.name.lower()
        input_question_h5 = os.path.join(
            self.data_dir,
            self.LOCAL_NAME,
            f"{self.closure_test_fam.name.lower()}_{split}.h5",
        )

        # Images are from CLEVR validation set.
        clevr_val_features_h5 = None
        if self.feature_suffix is not None:
            clevr_val_features_h5 = os.path.join(
                self.data_dir, self.CLEVR_NAME, f"val{self.feature_suffix}"
            )
        elif (
            self.user_im_feat_val is not None
        ):  # Either user-provided image features, or don't load them at all
            clevr_val_features_h5 = self.user_im_feat_val

        clevr_val_scenes = os.path.join(
            self.clevr_datamodule.ori_clevr_path, "scenes", "CLEVR_val_scenes.json"
        )

        loader_kwargs = {
            "question_h5": input_question_h5,
            "feature_h5": clevr_val_features_h5,
            "scene_path": clevr_val_scenes,
            # if isinstance(ee, ClevrExecutor) else None,
            "vocab": self.vocab,
            "batch_size": self.val_batch_size,
            "num_workers": self.loader_num_workers,
            "as_tokens": self.as_tokens,
            "resize_images": self.resize_images,
        }
        if self.num_test_samples is not None and self.num_test_samples > 0:
            loader_kwargs["max_samples"] = self.num_test_samples
        if self.question_families:
            loader_kwargs["question_families"] = self.question_families

        return ClevrDataLoader(**loader_kwargs)


# Quick main method that doesn't do anything other than setup CLEVR;
# should download the dataset and process it. This shouldn't need to be
# repeated in future.
if __name__ == "__main__":
    from vqa_framework.global_settings import comp_env_vars

    logging.basicConfig(level=20)
    dm = ClosureDataModule(
        closure_test_question_fam=ClosureFamily.AND_MAT_SPA,
        closure_test_type=ClosureTestType.TEST,
        tmp_dir=comp_env_vars.TMP_DIR,
        data_dir=comp_env_vars.DATA_DIR,
        val_batch_size=2,
        dm_batch_size=2,
    )
    dm.prepare_data()
    dm.setup()

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
