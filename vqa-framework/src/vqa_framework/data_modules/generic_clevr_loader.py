# CLEVR dataloader Modified to accept fixed images directory and a list of
# questions .json
# NOTE: Current question processing script saves original indices into h5 file;
#       therefore that can abe used to retrieve the original question indices.

# NOTE: Not sure how nicely the vocab can be extended to new words if the
#       different splits use different vocabs. For now, assume default vocab
#       or provided.

import pytorch_lightning as pl
import os
from argparse import Namespace
from typing import Optional, List, Union, Dict, Any, Tuple
import logging
from vqa_framework.utils.misc import add_init_args
from vqa_framework.utils.vocab import ClosureVocab

from vqa_framework.resources.vocabs import CLEVR_VOCAB

from vqa_framework.data_modules.clevr_loader import ClevrFeature

# Code re-used from CLOSURE paper
from vqa_framework.vr.utils import load_vocab
from vqa_framework.vr.data import ClevrDataLoader
import vqa_framework.data_modules.clevr_scripts as scripts
from vqa_framework.data_modules.clevr_im_scripts.CLEVR_im_to_hdf5 import (
    main as clevr_ims_to_h5,
)
from enum import Enum
import json


def path_to_h5_filename(json_path: str, split: str):
    json_path = os.path.split(json_path)[1]
    return f"{split}_{json_path[:-4]}h5"


class GenericCLEVRDataModule(pl.LightningDataModule):
    """
    Data module for processing a dataset in a CLEVR-like format,
    and storing images into HDF5 format.

    ==== Public Attributes ====
    <self.vocab>: Only initialized after setup.

    """

    vocab: Optional[ClosureVocab]

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
            initializer=GenericCLEVRDataModule.__init__,
            exclusions=["data_dir", "tmp_dir"],
            prefix=prefix,
            postfix=postfix,
            defaults=defaults,
        )
        return parent_parser

    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        dm_images_name: Optional[str],  # must be unique; used for target dir name
        dm_questions_name: str,  # must be unique; used for target dir name
        dm_train_images: Optional[Union[str, os.PathLike]] = None,
        dm_val_images: Optional[Union[str, os.PathLike]] = None,
        dm_test_images: Optional[Union[str, os.PathLike]] = None,
        dm_train_scenes: Optional[Union[str, os.PathLike]] = None,
        dm_val_scenes: Optional[Union[str, os.PathLike]] = None,
        dm_test_scenes: Optional[Union[str, os.PathLike]] = None,
        dm_train_questions: Optional[Union[str, os.PathLike]] = None,
        dm_val_questions: Optional[List[Union[str, os.PathLike]]] = None,
        dm_test_questions: Optional[List[Union[str, os.PathLike]]] = None,
        dm_vocab: str = "CLEVR_DEFAULT",
        dm_batch_size: int = 64,
        dm_shuffle_train_data: bool = True,
        dm_question_families: Optional[List[int]] = None,
        dm_num_train_samples: Optional[int] = None,
        loader_num_workers: int = os.cpu_count(),
        dm_percent_of_data_for_training: float = 1,
        val_batch_size: int = 512,
        dm_num_val_samples: Optional[int] = None,
        dm_num_test_samples: Optional[int] = None,
        image_features: Optional[ClevrFeature] = ClevrFeature.IEP_RESNET,
        as_tokens: bool = False,
        resize_images: tuple = None,
        user_image_features_train: Optional[
            Union[str, List[Tuple[str, str, str]]]
        ] = None,
        user_image_features_val: Optional[
            Union[str, List[Tuple[str, str, str]]]
        ] = None,
        user_image_features_test: Optional[
            Union[str, List[Tuple[str, str, str]]]
        ] = None,
        pin_memory: bool = False,
        drop_last_train: bool = False,
        **kwargs,
    ):
        """
        <data_dir>: Where to store original CLEVR dataset, and processed
                    equivalent. NOTE: if <overwrite_CLEVR_loc> is not None,
                    then we will NOT download CLEVR & save to tmp, we'll instead
                    look for an existing copy of the downloaded dataset at
                    <overwrite_CLEVR_loc>
        <dm_images_name>: A *unique* name for the *images* this dataloader is
                          using. The images will be saved under a
                          <data_dir>/<dm_images_name> folder, so this folder
                          must either not exist, or already contain the
                          processed images.
                          Can be None if image_features is None
        <dm_questions_name>: A *unique* name for the *questions* this dataloader
                          is using. The questions will be saved under a
                          <data_dir>/<dm_questions_name> folder, so this folder
                          must either not exist, or already contain the
                          processed questions.
        <dm_train_images>: (also _val and _test). Path to a directory of .png files.
                           They should be indexed as per CLEVR standard (i.e.,
                           filename ending with "_{idx}.png")
                           Furthermore, the image indices should start at 0,
                           and incremement by 1. You cannot skip or
                           duplicate indices.
                           Can be None if image_features is None
        <dm_train_scenes>: (also _val and _test). Path to .json file in CLEVR
                           format specifying the scene graphs of the images.
                           NOTE!!! internal list of scenes must be sorted to
                           match the image indices (i.e., ith scene in list
                           is for image i)
                           Can be None if there are no scene graphs, or they
                           aren't required.
        <dm_train_questions>: path to a .json file containing the CLEVR questions.
        <dm_val_questions>: (also _test). List of paths, each to a .json file
                            of CLEVR questions. Each filename should be unique,
                            as we will process each of these files and
                            create a corresponding file at <data_dir>/<dm_questions_name>/<val_{json_filename}.h5>
        <dm_vocab>: path to a CLEVR .json file for the vocab, or "CLEVR_DEFAULT";
                    ("CLEVR_DEFAULT" being to instead use the default CLEVR vocab
                    as the vocab)
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
                                        in this case, it should be either a
                                        path to an .hdf5 feature file in exactly
                                        the same format as the CLEVR dataloader
                                        would create, or else a list
                                        of triples of strings [(file, file_key, name),].
                                        In the case of a path to an .hdf5 file
                                        exactly as the CLEVR dataloader would
                                        create, then GenericClevrDataloader
                                        will behave exactly the same as the
                                        standard CLEVR dataloader (i.e., returning
                                        a tensor of image features).
                                        In the case of triples, first is the
                                        path to an hdf5 file, second is the key
                                        within the hdf5 file, and last is the name
                                        of the feature. It should be the case
                                        that hdf5_file[{hdf5 key}][i] is the feature
                                        for the ith image. Indexing should be the
                                        same as the hdf5 files automatically created
                                        by this dataloader (easiest way is probably
                                        to read in the generated hdf5 file, extract
                                        features, and save into the new hdf5 file
                                        in the same order). The dataloader
                                        will then return a dictionary from {name} to
                                        {hdf5_file[{hdf5 key}][i]} for the proper
                                        index i.
        <pin_memory>: If true then copy tensors into CUDA pinned memory before
                      returning them. (i.e., set corresponding flag in pytorch dataloader).
                      Note that due to a bug in pytorch, in all of the dataloaders
                      created it will be the case that values that would normally
                      be returned as tuples will instead be returned as lists.
                      https://github.com/pytorch/pytorch/issues/48419
        <drop_last_train>: If true then ensure that all train batches exactly
                           match the requested batch size (i.e., if the final
                           batch of remaining values is too small, this batch
                           will be dropped).

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

        if dm_images_name is None:
            # We don't provide the images_name iff we don't need it!
            # i.e., if we're not providing any images, or if the user is
            # pointing us directly at the desired features.
            assert (
                image_features is None or image_features == ClevrFeature.USER_PROVIDED
            )

        self.data_dir = data_dir
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
        self.vocab_path = dm_vocab
        self.IMAGES_LOCAL_NAME = dm_images_name
        self.QUESTIONS_LOCAL_NAME = dm_questions_name
        self.train_questions_path = dm_train_questions
        self.val_questions_paths = dm_val_questions
        self.test_questions_paths = dm_test_questions
        self.train_img_path = dm_train_images
        self.val_img_path = dm_val_images
        self.test_img_path = dm_test_images
        self.train_scene_path = dm_train_scenes
        self.val_scene_path = dm_val_scenes
        self.test_scene_path = dm_test_scenes
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

        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train

        paths = []
        if dm_train_questions is not None:
            paths = [dm_train_questions]
        if dm_val_questions is not None:
            assert len(dm_val_questions) == len(
                set([os.path.split(x)[1] for x in dm_val_questions])
            )
            paths += dm_val_questions
        if dm_test_questions is not None:
            assert len(dm_test_questions) == len(
                set([os.path.split(x)[1] for x in dm_test_questions])
            )
            paths += dm_test_questions

        for path in paths:
            assert path.endswith(".json")

        for path in [dm_train_scenes, dm_val_scenes, dm_test_scenes]:
            if path is not None:
                with open(path, "r") as infile:
                    scenes = json.load(infile)
                for i, scene in enumerate(scenes["scenes"]):
                    if i != scene["image_index"]:
                        raise ValueError(
                            "Scene .json's must be sorted w.r.t to"
                            "image index, and indices must start at 0. However,"
                            f"{path} violates this requriement"
                        )

    def get_vocab(self):
        self.prepare_data()
        self.setup()
        return self.vocab

    def prepare_data(self) -> None:
        # prepare_data is called from a single process (e.g. GPU 0).
        # Do not use it to assign state (self.x = y).

        if self.IMAGES_LOCAL_NAME is None:
            assert (
                self.type_of_image_features is None
                or self.type_of_image_features == ClevrFeature.USER_PROVIDED
            )
        else:
            # Extract image features & process text if not already done
            # Get directory where processed images should go
            img_target_dir = os.path.join(self.data_dir, self.IMAGES_LOCAL_NAME)
            # Iterate over each possible split
            for split, img_path in [
                ("train", self.train_img_path),
                ("val", self.val_img_path),
                ("test", self.test_img_path),
            ]:
                # Figure out the file we should be expecting to see; skip if
                # there isn't one.
                if self.feature_suffix is None or img_path is None:
                    continue
                target_name = f"{split}{self.feature_suffix}"

                if not os.path.exists(os.path.join(img_target_dir, target_name)):
                    logging.warning("Generic CLEVR DATALOADER: Processing images")
                    os.makedirs(img_target_dir, exist_ok=True)

                    # If we need IEP ResNet features, then extract them
                    if self.type_of_image_features == ClevrFeature.IEP_RESNET:
                        logging.warning(
                            f"Generic CLEVR DATALOADER: Extracting {split} image Resnet features"
                        )
                        # Create HDF5 files for images; note files have are extracted ResNet features.
                        scripts.extract_feats(
                            Namespace(
                                input_image_dir=os.path.join(img_path),
                                output_h5_file=os.path.join(
                                    img_target_dir, target_name
                                ),
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
                                input_image_dir=os.path.join(img_path),
                                output_h5_file=os.path.join(
                                    img_target_dir, target_name
                                ),
                                max_images=None,
                            )
                        )
                else:
                    logging.info(
                        f"CLEVR DATALOADER: Skipping generation of {target_name}; file already exists"
                    )

        question_target_dir = os.path.join(self.data_dir, self.QUESTIONS_LOCAL_NAME)
        os.makedirs(question_target_dir, exist_ok=True)

        if self.vocab_path == "CLEVR_DEFAULT":
            clevr_vocab_dict = CLEVR_VOCAB
        else:
            with open(self.vocab_path, "r") as infile:
                clevr_vocab_dict = json.load(infile)

        if not (os.path.exists(os.path.join(question_target_dir, "vocab.json"))):
            # Save training vocab
            with open(os.path.join(question_target_dir, "vocab.json"), "w") as outfile:
                json.dump(clevr_vocab_dict, outfile)
        else:
            # Confirm the vocab is OK
            with open(os.path.join(question_target_dir, "vocab.json"), "r") as infile:
                assert clevr_vocab_dict == json.load(infile)

        # Preprocess questions using train vocab.
        all_question_jsons = []
        if self.train_questions_path is not None:
            all_question_jsons += [("train", self.train_questions_path)]
        if self.val_questions_paths is not None:
            all_question_jsons += [("val", x) for x in self.val_questions_paths]
        if self.test_questions_paths is not None:
            all_question_jsons += [("test", x) for x in self.test_questions_paths]

        for split, json_path in all_question_jsons:  # ["val", "test", "train"]:
            assert json_path.endswith(".json")
            target_filename = path_to_h5_filename(json_path, split)

            if not os.path.exists(os.path.join(question_target_dir, target_filename)):
                logging.warning(f"CLEVR DATALOADER: Processing {split} questions")
                scripts.preprocess_questions(
                    Namespace(
                        input_questions_json=[json_path],
                        output_h5_file=os.path.join(
                            question_target_dir, target_filename
                        ),
                        input_vocab_json=os.path.join(
                            question_target_dir, "vocab.json"
                        ),
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
                    f"CLEVR DATALOADER: Skipping generation of {target_filename}; file already exists"
                )

    def setup(self, stage: Optional[str] = None) -> None:
        # stage is fit, validate, test, or None (equal to all)
        # setup is called from every process. Setting state here is okay.
        # ? 'fit' = train + val splits

        # Based on deprecation warning, modify setup so it only runs if it
        # hasn't alreaady
        if self.vocab is None:
            vocab_json = os.path.join(
                self.data_dir, self.QUESTIONS_LOCAL_NAME, "vocab.json"
            )
            self.vocab = ClosureVocab(load_vocab(vocab_json))

    def train_dataloader(self):
        if self.train_questions_path is None:
            raise ValueError(
                "Cannot create train loader because `train_questions` was None when this GenericCLEVRDataModule was created."
            )

        train_question_h5 = os.path.join(
            self.data_dir,
            self.QUESTIONS_LOCAL_NAME,
            path_to_h5_filename(self.train_questions_path, "train"),
        )
        train_features_h5 = None
        if self.feature_suffix is not None:
            train_features_h5 = os.path.join(
                self.data_dir, self.IMAGES_LOCAL_NAME, f"train{self.feature_suffix}"
            )
        elif (
            self.user_im_feat_train is not None
        ):  # Either user-provided image features, or don't load them at all
            train_features_h5 = self.user_im_feat_train

        train_loader_kwargs = {
            "question_h5": train_question_h5,
            "feature_h5": train_features_h5,  # if features_needed else None,
            "scene_path": self.train_scene_path,  # if scenes_needed else None,
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
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last_train,
        }
        return ClevrDataLoader(**train_loader_kwargs)

    def val_dataloader(self):
        if self.val_questions_paths is None:
            raise ValueError(
                "Cannot create val dataloader because `val_questions` was None when this GenericCLEVRDataModule was created."
            )

        val_dataloaders = []
        for val_question_json in self.val_questions_paths:
            val_question_h5 = os.path.join(
                self.data_dir,
                self.QUESTIONS_LOCAL_NAME,
                path_to_h5_filename(val_question_json, "val"),
            )
            val_features_h5 = None
            if self.feature_suffix is not None:
                val_features_h5 = os.path.join(
                    self.data_dir, self.IMAGES_LOCAL_NAME, f"val{self.feature_suffix}"
                )
            elif (
                self.user_im_feat_val is not None
            ):  # Either user-provided image features, or don't load them at all
                val_features_h5 = self.user_im_feat_val

            val_loader_kwargs = {
                "question_h5": val_question_h5,
                "feature_h5": val_features_h5,  # if features_needed else None,
                "scene_path": self.val_scene_path,  # if scenes_needed else None,
                "load_features": False,  # args.load_features,
                "vocab": self.vocab,
                "batch_size": self.val_batch_size,
                "question_families": self.question_families,
                "max_samples": self.num_val_samples,
                "num_workers": self.loader_num_workers,
                "as_tokens": self.as_tokens,
                "resize_images": self.resize_images,
                "pin_memory": self.pin_memory,
            }
            val_dataloaders.append(ClevrDataLoader(**val_loader_kwargs))
        return val_dataloaders

    def test_dataloader(self):
        if self.test_questions_paths is None:
            raise ValueError(
                "Cannot create test dataloader because `test_questions` was None when this GenericCLEVRDataModule was created."
            )

        test_loaders = []

        for test_json in self.test_questions_paths:
            input_question_h5 = os.path.join(
                self.data_dir,
                self.QUESTIONS_LOCAL_NAME,
                path_to_h5_filename(test_json, "test"),
            )
            input_features_h5 = None
            if self.feature_suffix is not None:
                input_features_h5 = os.path.join(
                    self.data_dir, self.IMAGES_LOCAL_NAME, f"test{self.feature_suffix}"
                )
            elif (
                self.user_im_feat_test is not None
            ):  # Either user-provided image features, or don't load them at all
                input_features_h5 = self.user_im_feat_test

            loader_kwargs = {
                "question_h5": input_question_h5,
                "feature_h5": input_features_h5,
                "scene_path": self.test_scene_path,
                # if isinstance(ee, ClevrExecutor) else None,
                "vocab": self.vocab,
                "batch_size": self.val_batch_size,
                "as_tokens": self.as_tokens,
                "resize_images": self.resize_images,
                "num_workers": self.loader_num_workers,
                "pin_memory": self.pin_memory,
            }
            if self.num_test_samples is not None and self.num_test_samples > 0:
                loader_kwargs["max_samples"] = self.num_test_samples
            if self.question_families:
                loader_kwargs["question_families"] = self.question_families
            test_loaders.append(ClevrDataLoader(**loader_kwargs))
        return test_loaders


# See tests for example usage.
