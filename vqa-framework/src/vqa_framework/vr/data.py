#!/usr/bin/env python3

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import numpy as np
import PIL.Image
import h5py
import io
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import random, math
import json
from vqa_framework.vr.programs import ProgramConverter
import logging
from typing import Union, List, Tuple


def load_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_dict = json.load(f)["scenes"]
    scenes = []
    for s in scenes_dict:
        table = []
        for i, o in enumerate(s["objects"]):
            item = {}
            item["id"] = "%d-%d" % (s["image_index"], i)
            if "3d_coords" in o:
                item["position"] = [
                    np.dot(o["3d_coords"], s["directions"]["right"]),
                    np.dot(o["3d_coords"], s["directions"]["front"]),
                    o["3d_coords"][2],
                ]
            else:
                item["position"] = o["position"]
            item["color"] = o["color"]
            item["material"] = o["material"]
            item["shape"] = o["shape"]
            item["size"] = o["size"]
            table.append(item)
        scenes.append(table)
    return scenes


def _dataset_to_tensor(dset, mask=None, dtype=None):
    arr = np.asarray(dset, dtype=np.int64 if dtype is None else dtype)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor


def _gen_subsample_mask(num, percent=1.0):
    chosen_num = math.floor(num * percent)
    mask = np.full((num,), False)
    selected_ids = np.asarray(random.sample(range(num), chosen_num), dtype="int32")
    mask[selected_ids] = True
    return mask


class ClevrDataset(Dataset):
    def __init__(
        self,
        question_h5,
        feature_h5_path: Union[str, List[Tuple[str, str, str]]],
        scene_path,
        vocab,
        mode="prefix",
        load_features=False,
        max_samples=None,
        question_families=None,
        percent_of_data=1.0,
        oversample=None,
        oversample_shift=None,
        as_tokens=False,
        resize_images=None,
    ):
        """

        :param question_h5:
        :param feature_h5_path: Either string path to hdf5 file of image features,
                                or else a list of string triples of the form
                                [(hdf5 file path, hdf5 file key, name),]
                                such that the image feature for the ith image
                                is {hdf5 file}[{hdf5 key}][i].
                                In place of the usual image feature tensor,
                                a dictionary from {name} to the stacked
                                feature tensor will be provided.
        :param scene_path:
        :param vocab:
        :param mode:
        :param load_features:
        :param max_samples:
        :param question_families:
        :param percent_of_data:
        :param oversample:
        :param oversample_shift:
        :param as_tokens:
        :param resize_images:
        """
        logging.info("CLEVR DATASET Object")
        mode_choices = ["prefix", "postfix"]
        if mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % mode)
        self.vocab = vocab
        self.program_converter = ProgramConverter(vocab)
        self.feature_h5_path = feature_h5_path
        self.feature_h5 = None
        self.all_features = None
        self.load_features = load_features
        self.mode = mode
        self.max_samples = max_samples
        self.as_tokens = as_tokens
        self.resize_images = resize_images

        # Compute the mask
        num_mask_options_chosen = (
            int(percent_of_data < 1.0)
            + int(question_families is not None)
            + int(oversample is not None)
        )
        if num_mask_options_chosen > 1:
            raise ValueError()

        mask = None
        if oversample is not None:
            all_families = question_h5["question_families"][()]
            regular_indices = (all_families < oversample_shift).nonzero()[0]
            oversampled_indices = (all_families >= oversample_shift).nonzero()[0]
            mask = np.hstack([regular_indices] + [oversampled_indices] * oversample)
        if question_families is not None:
            # Use only the specified families
            all_families = np.asarray(question_h5["question_families"])
            N = all_families.shape[0]
            logging.info(question_families)
            target_families = np.asarray(question_families)[:, None]
            mask = (all_families == target_families).any(axis=0)
        if percent_of_data < 1.0:
            num_example = np.asarray(question_h5["image_idxs"]).shape[0]
            mask = _gen_subsample_mask(num_example, percent_of_data)
        self.mask = mask

        # Data from the question file is small, so read it all into memory
        logging.info("Reading question data into memory")
        self.all_types = None
        if "types" in question_h5:
            self.all_types = _dataset_to_tensor(question_h5["types"], mask)
        self.all_question_families = None
        if "question_families" in question_h5:
            self.all_question_families = _dataset_to_tensor(
                question_h5["question_families"], mask
            )
        self.all_questions = _dataset_to_tensor(question_h5["questions"], mask)
        self.all_questions_len = _dataset_to_tensor(question_h5["questions_len"], mask)
        self.all_image_idxs = _dataset_to_tensor(question_h5["image_idxs"], mask)
        self.all_programs = None
        if "programs" in question_h5:
            self.all_programs = _dataset_to_tensor(question_h5["programs"], mask)
            self.all_programs_len = _dataset_to_tensor(
                question_h5["programs_len"], mask
            )
        self.all_answers = None
        if "answers" in question_h5:
            self.all_answers = _dataset_to_tensor(question_h5["answers"], mask)
        if scene_path:
            self.all_scenes = load_scenes(scene_path)
        else:
            self.all_scenes = None

    def __getitem__(self, index):
        # Open the feature or load them if requested
        if self.feature_h5_path and not self.feature_h5:
            if isinstance(self.feature_h5_path, str):
                self.feature_h5 = h5py.File(self.feature_h5_path, "r")
                if self.load_features:
                    self.features = self.feature_h5["features"][()]
            else:
                assert isinstance(self.feature_h5_path, list)
                self.feature_h5 = {}
                for tmp_features_file in self.feature_h5_path:
                    # self.features_h5 at NAME is equal to the hdf5 file at the corresponding path,
                    # and also record the key in the hdf5 file that we want to read
                    self.feature_h5[tmp_features_file[2]] = (
                        h5py.File(tmp_features_file[0]),
                        tmp_features_file[1],
                    )
                if self.load_features:
                    raise NotImplementedError(
                        "Haven't implemented feature loading with user provided features."
                    )

        if self.all_question_families is not None:
            question_family = self.all_question_families[index]
        q_type = None if self.all_types is None else self.all_types[index]
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        image_idx = self.all_image_idxs[index]
        answer = None
        if self.all_answers is not None:
            answer = self.all_answers[index]
        program_seq = None
        program_len = None
        if self.all_programs is not None:
            program_seq = self.all_programs[index]
            program_len = self.all_programs_len[index]

        if self.all_scenes:
            scene = self.all_scenes[image_idx]
        else:
            scene = None

        if self.feature_h5_path and isinstance(
            self.feature_h5_path, str
        ):  # Normal load from a single file
            if self.load_features:
                feats = self.features[image_idx]
            else:
                feats = self.feature_h5["features"][image_idx]
            if feats.ndim == 1:
                feats = (
                    np.array(PIL.Image.open(io.BytesIO(feats))).transpose(2, 0, 1)
                    / 255.0
                )
            feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
        elif self.feature_h5_path and isinstance(
            self.feature_h5_path, list
        ):  # Load from several specified files
            feats = {}
            for name in self.feature_h5:
                feature_hdf5, hdf5_key = self.feature_h5[name]
                feats[name] = feature_hdf5[hdf5_key][image_idx]
        else:
            feats = [0]
            feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))

        if self.resize_images:
            feats = self._resize_image(feats)

        if self.as_tokens:
            question, answer, program_seq = self._as_tokens(
                question, answer, program_seq
            )

        # NOTE: Following changed for VQA-FRAMEWORK
        """
        if q_type is None:
            return (question, index, feats, scene, answer, program_seq)
        return ([question, q_type], index, feats, scene, answer, program_seq)
        """
        return (
            question,
            index,
            feats,
            scene,
            answer,
            program_seq,
            q_type,
            question_len,
            program_len,
        )

    def __len__(self):
        if self.max_samples is None:
            return self.all_questions.size(0)
        else:
            return min(self.max_samples, self.all_questions.size(0))

    def _resize_image(self, image):
        image = cv2.resize(
            image.numpy().transpose(1, 2, 0),
            dsize=(384, 384),
            interpolation=cv2.INTER_CUBIC,
        )
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image)
        return image

    def _remove_pad_start_eos(self, str):
        str = str.replace("<START>", "")
        str = str.replace("<END>", "")
        str = str.replace("<NULL>", "")
        return str.rstrip()

    def _as_tokens(self, question, answer, program_seq):
        if answer:
            answer = self.vocab.answer_idx_to_token(answer.item())
        question = " ".join(
            [self.vocab.question_idx_to_token(q.item()) for q in question]
        )
        question = self._remove_pad_start_eos(question)
        if isinstance(program_seq, torch.Tensor):
            program_seq = " ".join(
                [
                    self._remove_pad_start_eos(
                        self.vocab.program_idx_to_token(p.item())
                    )
                    for p in program_seq
                ]
            )
        return question, answer, program_seq


class ClevrDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if "question_h5" not in kwargs:
            raise ValueError("Must give question_h5")
        if "feature_h5" not in kwargs:
            raise ValueError("Must give feature_h5")
        if "vocab" not in kwargs:
            raise ValueError("Must give vocab")

        scene_path = kwargs.pop("scene_path")
        logging.info(f"Reading scenes from {scene_path}")
        feature_h5_path = kwargs.pop("feature_h5")
        logging.info(f"Reading features from {feature_h5_path}")
        question_h5_path = kwargs.pop("question_h5")
        logging.info(f"Reading questions from {question_h5_path}")

        vocab = kwargs.pop("vocab")
        mode = kwargs.pop("mode", "prefix")
        load_features = kwargs.pop("load_features", False)
        percent_of_data = kwargs.pop("percent_of_data", 1.0)
        oversample = kwargs.pop("oversample", None)
        oversample_shift = kwargs.pop("oversample_shift", None)
        question_families = kwargs.pop("question_families", None)
        max_samples = kwargs.pop("max_samples", None)
        as_tokens = kwargs.pop("as_tokens", False)
        resize_images = kwargs.pop("resize_images", False)
        with h5py.File(question_h5_path, "r") as question_h5:
            self.dataset = ClevrDataset(
                question_h5,
                feature_h5_path,
                scene_path,
                vocab,
                mode,
                load_features=load_features,
                max_samples=max_samples,
                question_families=question_families,
                percent_of_data=percent_of_data,
                oversample=oversample,
                oversample_shift=oversample_shift,
                as_tokens=as_tokens,
                resize_images=resize_images,
            )
        kwargs["collate_fn"] = clevr_collate
        super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def clevr_collate(batch):
    transposed = list(zip(*batch))
    question_batch = default_collate(transposed[0])

    index_batch = default_collate(transposed[1])

    feat_batch = transposed[2]
    if isinstance(
        feat_batch[0], dict
    ):  # merge features when we have a dict of distinct features
        tmp = {}
        for name in feat_batch[0]:
            tmp[name] = default_collate([fb[name] for fb in feat_batch])
        feat_batch = tmp
    elif all(f is not None for f in feat_batch):  # merge when we have a single feature
        feat_batch = default_collate(feat_batch)

    scene_batch = transposed[3]

    answer_batch = transposed[4]
    if transposed[4][0] is not None:
        answer_batch = default_collate(answer_batch)

    program_seq_batch = transposed[5]
    if transposed[5][0] is not None:
        program_seq_batch = default_collate(program_seq_batch)

    # Added for VQA-FRAMEWORK
    # collate question family
    q_fam = transposed[6]
    if transposed[6][0] is not None:
        q_fam = default_collate(q_fam)
    # collate question length (must be present)
    q_len = default_collate(transposed[7])
    # collate program length (may be None)
    p_len = transposed[8]
    if transposed[8][0] is not None:
        p_len = default_collate(p_len)

    return [
        question_batch,
        index_batch,
        feat_batch,
        scene_batch,
        answer_batch,
        program_seq_batch,
        q_fam,
        q_len,
        p_len,
    ]
