import pytest
from vqa_framework.data_modules.clevr_loader import CLEVRDataModule, ClevrFeature
from vqa_framework.data_modules.closure_loader import (
    ClosureDataModule,
    ClosureFamily,
    ClosureTestType,
)
from vqa_framework.data_modules.generic_clevr_loader import GenericCLEVRDataModule

import os
import shutil
import torch
from vqa_framework.vr.preprocess import decode
from typing import Dict, Any
import logging
import json
from vqa_framework.models.symbolic_exec import SymbolicClevrExecutor
from vqa_framework.global_settings import comp_env_vars

ORI_CLEVR_EXISTS_MSG = "Extracted CLEVR dataset already exists"
PROCESSED_EXISTS_MSGS = "Skipping generation"


@pytest.fixture
def clean_h5(tmpdir):
    """Clean processed data b/f and after test"""
    # Setup: fill with any logic you want
    if os.path.exists(os.path.join(".", "resources", "r2", "hdf5_clevr")):
        shutil.rmtree(os.path.join(".", "resources", "r2", "hdf5_clevr"))
    if os.path.exists(os.path.join(".", "resources", "r3", "hdf5_closure")):
        shutil.rmtree(os.path.join(".", "resources", "r3", "hdf5_closure"))
    if os.path.exists(os.path.join(".", "resources", "r6", "gen_img_dir")):
        shutil.rmtree(os.path.join(".", "resources", "r6", "gen_img_dir"))
    if os.path.exists(os.path.join(".", "resources", "r6", "gen_ques_dir")):
        shutil.rmtree(os.path.join(".", "resources", "r6", "gen_ques_dir"))
    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    if os.path.exists(os.path.join(".", "resources", "r2", "hdf5_clevr")):
        shutil.rmtree(os.path.join(".", "resources", "r2", "hdf5_clevr"))
    if os.path.exists(os.path.join(".", "resources", "r3", "hdf5_closure")):
        shutil.rmtree(os.path.join(".", "resources", "r3", "hdf5_closure"))
    if os.path.exists(os.path.join(".", "resources", "r6", "gen_img_dir")):
        shutil.rmtree(os.path.join(".", "resources", "r6", "gen_img_dir"))
    if os.path.exists(os.path.join(".", "resources", "r6", "gen_ques_dir")):
        shutil.rmtree(os.path.join(".", "resources", "r6", "gen_ques_dir"))


def fixed_closure_init(data_dir, tmp_dir, image_features, dm_shuffle_train_data=False):
    return ClosureDataModule(
        closure_test_type=ClosureTestType.TEST,
        closure_test_question_fam=ClosureFamily.AND_MAT_SPA,
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=image_features,
        dm_shuffle_train_data=dm_shuffle_train_data,
    )


def fixed_generic_clevr_init(
    data_dir, tmp_dir, image_features, dm_shuffle_train_data=False
):
    return GenericCLEVRDataModule(
        dm_images_name="gen_img_dir",
        dm_questions_name="gen_ques_dir",  # And the rest! Here on Gilligan's Isle!
        dm_train_images=os.path.join(".", "resources", "r6", "ims", "train"),
        dm_val_images=os.path.join(".", "resources", "r6", "ims", "val"),
        dm_test_images=os.path.join(".", "resources", "r6", "ims", "test"),
        dm_train_scenes=os.path.join(
            ".", "resources", "r6", "sgs", "train_scenes.json"
        ),
        dm_val_scenes=os.path.join(".", "resources", "r6", "sgs", "val_scenes.json"),
        dm_test_scenes=os.path.join(".", "resources", "r6", "sgs", "test_scenes.json"),
        dm_train_questions=os.path.join(
            ".", "resources", "r6", "questions", "train", "and_mat_spa_baseline.json"
        ),
        dm_val_questions=[
            os.path.join(".", "resources", "r6", "questions", "val", x)
            for x in [
                "and_mat_spa_val.json",
                "embed_mat_spa_val.json",
                "or_mat_val.json",
                "compare_mat_spa_val.json",
                "embed_spa_mat_val.json",
                "compare_mat_val.json",
                "or_mat_spa_val.json",
            ]
        ],
        dm_test_questions=[
            os.path.join(".", "resources", "r6", "questions", "test", x)
            for x in [
                "and_mat_spa_test.json",
                "embed_mat_spa_test.json",
                "or_mat_test.json",
                "compare_mat_spa_test.json",
                "embed_spa_mat_test.json",
                "compare_mat_test.json",
                "or_mat_spa_test.json",
            ]
        ],
        data_dir=data_dir,
        image_features=image_features,
        dm_shuffle_train_data=dm_shuffle_train_data,
    )


@pytest.mark.human
@pytest.mark.parametrize(
    "data_module,dm_name",
    [
        (fixed_generic_clevr_init, "generic"),
        (fixed_closure_init, "closure"),
        (CLEVRDataModule, "CLEVR"),
    ],
)
def test_by_visualizing_sample_test(tmp_path_factory, clean_h5, data_module, dm_name):
    import matplotlib.pyplot as plt
    import numpy as np

    if dm_name == "CLEVR":
        logging.critical("CLEVR Should be: brown cylinder, grey sphere, blue cube")
        data_dir = "./resources/r2/"
    elif dm_name == "closure":
        logging.critical(
            "CLOSURE Should be: dull brown cylinder, red metal cylinder, red cube, gray sphere, shiny metal cube (val 11)"
        )
        data_dir = "./resources/r3/"
    else:
        assert dm_name == "generic"
        logging.critical(
            "Generic CLEVR should be: Big green metal cube, small purple metal cube, small rubber purple cylinder, two more big green metal cubes (test_000009.png)"
        )
        data_dir = "./resources/r6/"

    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #
    dm = data_module(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IMAGES,
    )
    dm.prepare_data()
    dm.setup("test")
    test_loader = dm.test_dataloader()

    if dm_name == "generic":
        test_loader = test_loader[4]  # Get just the fourth dataloader then.

    for batch in test_loader:
        features = batch[2]
        for i in range(features.size(0)):
            plt.imshow(
                features[i].cpu().numpy().transpose(1, 2, 0).astype(np.int32)
            )  # CxHxW, RGB order -> HxWxC, RGB order
            plt.show()
            break
        break
        # brown cylinder, grey sphere, blue cube


@pytest.mark.human
@pytest.mark.parametrize("shuffle", [True, False])
def test_by_human_train(tmp_path_factory, clean_h5, shuffle):
    logging.critical("HUMAN TEST STARTING")
    import matplotlib.pyplot as plt
    import numpy as np

    CHECK_EVERY = 10

    data_dir = "./resources/r2/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #
    dm = CLEVRDataModule(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IMAGES,
        dm_shuffle_train_data=shuffle,
    )
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    vocab = dm.get_vocab()

    total_seen = 0
    for batch in train_loader:
        features = batch[2]
        answers = batch[4]
        programs = batch[5]
        questions = batch[0]
        for i in range(features.size(0)):
            total_seen += 1
            if total_seen % CHECK_EVERY != 0:
                continue

            logging.critical("============================================")
            logging.critical("QUESTION:")
            logging.critical(
                " ".join(
                    [
                        vocab.question_idx_to_token(x)
                        for x in questions[i].numpy().tolist()
                    ]
                )
            )
            logging.critical("ANSWER:")
            logging.critical(vocab.answer_idx_to_token(answers[i].item()))
            logging.critical("PROGRAM:")
            logging.critical(
                " ".join(
                    [
                        vocab.program_idx_to_token(x)
                        for x in programs[i].numpy().tolist()
                    ]
                )
            )

            plt.imshow(features[i].cpu().numpy().transpose(1, 2, 0).astype(np.int32))
            plt.show()


@pytest.mark.parametrize(
    "im_feat", [ClevrFeature.IMAGES, ClevrFeature.IEP_RESNET, None]
)
def test_dummy_CLEVR_prepare_exists(tmp_path_factory, clean_h5, caplog, im_feat):
    """
    Minimal test, check just that CLEVR Data module can processed
    _something_, and that it doesn't crash in the subsequent steps.

    Also checks resulting dataloaders contain the correct number of items, and
    that if you run the DataModule again, it won't re-process the data.

    This test assumes the repo has already been downloaded, and tests
    accordingly.
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r2/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()
        dm = CLEVRDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            image_features=im_feat,
        )

        assert ORI_CLEVR_EXISTS_MSG not in caplog.text
        assert PROCESSED_EXISTS_MSGS not in caplog.text

        dm.prepare_data()

        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (PROCESSED_EXISTS_MSGS in caplog.text)
        assert ORI_CLEVR_EXISTS_MSG in caplog.text

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is again empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        assert len(test_loader.dataset) == 10

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert len(val_loader.dataset) == 7
        train_loader = dm.train_dataloader()
        assert len(train_loader.dataset) == 10 * 4


def test_dummy_CLEVR_prepare_exists_both_feats(tmp_path_factory, clean_h5, caplog):
    """
    Make sure we can switch between loading img features, images, and none
    without any weird things happening.
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r2/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(6):
        if repeat % 3 == 0:
            im_feat = ClevrFeature.IEP_RESNET
        elif repeat % 3 == 1:
            im_feat = ClevrFeature.IMAGES
        else:
            im_feat = None

        caplog.clear()
        dm = CLEVRDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            image_features=im_feat,
        )

        assert ORI_CLEVR_EXISTS_MSG not in caplog.text
        assert PROCESSED_EXISTS_MSGS not in caplog.text

        dm.prepare_data()

        # check dataset reuses stuff after first go
        if repeat == 0:
            assert PROCESSED_EXISTS_MSGS not in caplog.text
        else:
            assert PROCESSED_EXISTS_MSGS in caplog.text

        if repeat == 1:
            for split in ["train", "val", "test"]:
                assert (
                    f"CLEVR DATALOADER: Saving {split} images as hdf5 file"
                    in caplog.text
                )

        assert ORI_CLEVR_EXISTS_MSG in caplog.text

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is again empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        assert len(test_loader.dataset) == 10

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert len(val_loader.dataset) == 7
        train_loader = dm.train_dataloader()
        assert len(train_loader.dataset) == 10 * 4

        for x in train_loader:
            images = x[2]
            N = 40
            if im_feat is None:
                assert images.size() == (N, 1)
                assert torch.all(images == 0)
            elif im_feat == ClevrFeature.IEP_RESNET:
                C = 1024
                assert images.size() == (N, C, 14, 14)
            else:
                C = 3
                assert images.size() == (N, C, 320, 480)


def check_scene_graph_obj(sg_obj: Dict[str, Any]) -> None:
    assert sg_obj["id"].count("-") == 1
    (
        im_id,
        obj_id,
    ) = sg_obj[
        "id"
    ].split("-")
    assert im_id.isnumeric()
    assert obj_id.isnumeric()

    pos = sg_obj["position"]
    assert len(pos) == 3

    assert sg_obj["color"] in [
        "blue",
        "brown",
        "cyan",
        "gray",
        "green",
        "purple",
        "red",
        "yellow",
    ]
    assert sg_obj["material"] in ["rubber", "metal"]
    assert sg_obj["shape"] in ["cube", "cylinder", "sphere"]
    assert sg_obj["size"] in ["large", "small"]


@pytest.mark.parametrize(
    "im_feat", [ClevrFeature.IMAGES, ClevrFeature.IEP_RESNET, None]
)
@pytest.mark.parametrize("shuffle", [True, False])
def test_dummy_CLEVR_vals(tmp_path_factory, clean_h5, caplog, im_feat, shuffle):
    """
    Quick test, check dataloaders have plausible data. Also check correctness
    against expected return values.
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r2/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    with open("./resources/r2/groundtruth/answer_key.json", "r") as infile:
        ANSWER_KEY = json.load(infile)

        for spl in ANSWER_KEY:
            for id in ANSWER_KEY[spl]["q"]:
                ANSWER_KEY[spl]["q"][id] = ANSWER_KEY[spl]["q"][id].replace(";", " ;")

        ANSWER_MAX_Q_LEN = {
            spl: max(
                [
                    len(ANSWER_KEY[spl]["q"][id].split(" "))
                    for id in ANSWER_KEY[spl]["q"]
                ]
            )
            for spl in ["train", "test", "val"]
        }
        ANSWER_MAX_P_LEN = {
            spl: max([ANSWER_KEY[spl]["p"][id] for id in ANSWER_KEY[spl]["p"]])
            for spl in ["train", "val"]
        }

    for repeat in range(3):
        caplog.clear()
        dm = CLEVRDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            dm_batch_size=2,
            val_batch_size=4,
            image_features=im_feat,
            dm_shuffle_train_data=shuffle,
        )

        assert ORI_CLEVR_EXISTS_MSG not in caplog.text
        assert PROCESSED_EXISTS_MSGS not in caplog.text

        dm.prepare_data()

        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (PROCESSED_EXISTS_MSGS in caplog.text)
        assert ORI_CLEVR_EXISTS_MSG in caplog.text

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is still empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        # check_dataloader(test_loader, dm)

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        train_loader = dm.train_dataloader()

        for dl, name in [
            (test_loader, "test"),
            (val_loader, "val"),
            (train_loader, "train"),
        ]:
            assert len(dl.dataset) == 10 * (4 if name == "train" else 1) - 3 * (
                name == "val"
            )
            vocab = dm.vocab
            assert vocab is not None

            symbolic_executor = SymbolicClevrExecutor(vocab)

            remaining_items = len(dl.dataset)

            for x in dl:
                if name == "train":  # Train batch size
                    N = 2
                else:  # Val/Test batch size
                    N = 4

                # last batch will be smaller
                N = min(remaining_items, N)
                remaining_items -= N

                if im_feat == ClevrFeature.IEP_RESNET:
                    C = 1024
                else:
                    C = 3

                indices = x[1]
                assert isinstance(indices, torch.Tensor)
                assert indices.size() == (N,)
                # logging.warning(indices)
                assert torch.all(
                    indices <= 10 * (4 if name == "train" else 1) - 3 * (name == "val")
                )

                questions = x[0]
                assert isinstance(questions, torch.Tensor)
                assert questions.dim() == 2
                assert (
                    questions.size(0) == N
                )  # Note: dataset length goes into batch size
                assert (
                    questions.size(1) == ANSWER_MAX_Q_LEN[name] + 2
                )  # words, plus start & end tokens
                for i in range(N):
                    qu = questions[i]
                    id = str(indices[i].item())
                    # logging.warning(ANSWER_KEY[name]['q'])
                    expected = f"<START> {ANSWER_KEY[name]['q'][id].strip('?,')} <END>"
                    assert (
                        decode(
                            qu.numpy().tolist(),
                            vocab["question_idx_to_token"],
                            delim=" ",
                        )
                        == expected
                    )

                images = x[2]
                assert isinstance(images, torch.Tensor)

                if im_feat is None:
                    assert images.size() == (N, 1)
                    assert torch.all(images == 0)
                elif im_feat == ClevrFeature.IEP_RESNET:
                    assert images.size() == (N, C, 14, 14)
                else:
                    assert images.size() == (N, C, 320, 480)

                scene_graphs = x[3]
                if name != "test":
                    assert isinstance(scene_graphs, tuple)
                    assert len(scene_graphs) == N
                    for sg in scene_graphs:
                        assert isinstance(sg, list)
                        for obj in sg:
                            assert isinstance(obj, dict)
                            for key in [
                                "id",
                                "position",
                                "color",
                                "material",
                                "shape",
                                "size",
                            ]:
                                assert key in obj
                            check_scene_graph_obj(obj)
                else:
                    assert scene_graphs == (None,) * N

                answers = x[4]
                if name != "test":
                    assert isinstance(answers, torch.Tensor)
                    assert len(answers) == N
                else:  # NOTE: test split has no answers.
                    assert answers == (None,) * N

                programs = x[5]
                if name in ["val", "train"]:
                    assert isinstance(programs, torch.Tensor)
                    assert programs.dim() == 2
                    assert (
                        programs.size(0) == N
                    )  # Note: dataset length goes into batch size
                    assert (
                        programs.size(1) == ANSWER_MAX_P_LEN[name] + 2
                    )  # primitives, plus start & end tokens
                    for pr in programs:
                        decoded = decode(
                            pr.numpy().tolist(),
                            vocab["program_idx_to_token"],
                            delim=" ",
                        )
                        assert decoded.startswith("<START>")
                        assert decoded.endswith("<END>")
                else:  # Test split
                    assert programs == (None,) * N

                q_families = x[6]
                if name in ["train", "val"]:
                    assert isinstance(q_families, torch.Tensor)
                    assert q_families.dim() == 1
                    assert q_families.size(0) == N
                    #       NOTE!!! This is actually "type"
                    #       which is *different* from CLEVR question family.
                    #       See data_modules README for details.
                else:
                    assert q_families == (None,) * N

                q_len = x[7]
                assert isinstance(q_len, torch.Tensor)
                assert q_len.dim() == 1
                assert q_len.size(0) == N

                for q, l in zip(questions.numpy().tolist(), q_len.numpy().tolist()):
                    assert vocab["question_idx_to_token"][q[l - 1]] == "<END>"

                for i in range(N):
                    l = q_len[i].item()
                    id = str(indices[i].item())
                    q = len(ANSWER_KEY[name]["q"][id].split(" ")) + 2
                    assert l == q

                p_len = x[8]
                if name != "test":
                    assert isinstance(p_len, torch.Tensor)
                    assert p_len.dim() == 1
                    assert p_len.size(0) == N
                    for pr, l in zip(programs.numpy().tolist(), p_len.numpy().tolist()):
                        assert vocab["program_idx_to_token"][pr[l - 1]] == "<END>"
                else:
                    assert p_len == (None,) * N

                # Check execution works:
                if name != "test":
                    assert torch.all(
                        symbolic_executor(scene_graphs, programs) == answers
                    )


# @pytest.mark.slow
@pytest.mark.skip
def test_CLEVR_prepare_exists():
    """
    WARNING!!! This test may attempt to download & store the entire
    CLEVR dataset!!! So don't run it unless you want to/have it already.

    NOTE: For *already* downloaded CLEVR, this thing took me 3 hours.

    More specifically, it looks for you CLEVR in the *default* storage location.
    If you've already run this test, or downloaded CLEVR before, then this test
    will *only* check the reading of the already previously processed data.
    """

    dm = CLEVRDataModule(data_dir=comp_env_vars.DATA_DIR, tmp_dir=comp_env_vars.TMP_DIR)

    dm.prepare_data()

    # check this doesn't crash
    dm.setup("test")
    test_loader = dm.test_dataloader()
    # check_dataloader(test_loader, dm)

    dm.setup("fit")
    val_loader = dm.val_dataloader()
    train_loader = dm.train_dataloader()

    for dl, name in [
        (test_loader, "test"),
        (val_loader, "val"),
        (train_loader, "train"),
    ]:
        assert len(dl.dataset) == {"test": 149988, "val": 149991, "train": 699989}[name]
        # NOTE: "officially" the test set is of size 14,988; maybe that's the subset with the answers they have?
        vocab = dm.vocab
        assert vocab is not None

        symbolic_executor = SymbolicClevrExecutor(vocab)

        for x in dl:
            indices = x[1]
            assert isinstance(indices, torch.Tensor)

            N = indices.size(0)
            C = 1024

            questions = x[0]
            assert isinstance(questions, torch.Tensor)
            assert questions.dim() == 2
            assert questions.size(0) == N  # Note: dataset length goes into batch size
            for qu in questions:
                decoded = decode(
                    qu.numpy().tolist(),
                    vocab["question_idx_to_token"],
                    delim=" ",
                )
                assert decoded.startswith("<START>")
                assert decoded.endswith("<END>")

            images = x[2]
            assert isinstance(images, torch.Tensor)
            assert images.dim() == 4
            assert images.size() == (N, C, 14, 14)

            scene_graphs = x[3]
            if name != "test":
                assert isinstance(scene_graphs, tuple)
                assert len(scene_graphs) == N
                for sg in scene_graphs:
                    assert isinstance(sg, list)
                    for obj in sg:
                        assert isinstance(obj, dict)
                        for key in [
                            "id",
                            "position",
                            "color",
                            "material",
                            "shape",
                            "size",
                        ]:
                            assert key in obj
                        check_scene_graph_obj(obj)
            else:
                assert scene_graphs == (None,) * N

            answers = x[4]
            if name != "test":
                assert isinstance(answers, torch.Tensor)
                assert len(answers) == N
            else:  # NOTE: test split has no answers.
                assert answers == (None,) * N

            programs = x[5]
            if name in ["val", "train"]:
                assert isinstance(programs, torch.Tensor)
                assert programs.dim() == 2
                assert (
                    programs.size(0) == N
                )  # Note: dataset length goes into batch size
                for pr in programs:
                    decoded = decode(
                        pr.numpy().tolist(),
                        vocab["program_idx_to_token"],
                        delim=" ",
                    )
                    assert decoded.startswith("<START>")
                    assert decoded.endswith("<END>")
            else:  # Test split
                assert programs == (None,) * N

            q_families = x[6]
            if name in ["train", "val"]:
                assert isinstance(q_families, torch.Tensor)
                assert q_families.dim() == 1
                assert q_families.size(0) == N
                #       NOTE!!! This is actually "type"
                #       which is *different* from CLEVR question family. GAH.
                #       See data_modules README for details.
            else:
                assert q_families == (None,) * N

            q_len = x[7]
            assert isinstance(q_len, torch.Tensor)
            assert q_len.dim() == 1
            assert q_len.size(0) == N

            for q, l in zip(questions.numpy().tolist(), q_len.numpy().tolist()):
                assert vocab["question_idx_to_token"][q[l - 1]] == "<END>"

            p_len = x[8]
            if name != "test":
                assert isinstance(p_len, torch.Tensor)
                assert p_len.dim() == 1
                assert p_len.size(0) == N
                for pr, l in zip(programs.numpy().tolist(), p_len.numpy().tolist()):
                    assert vocab["program_idx_to_token"][pr[l - 1]] == "<END>"
            else:
                assert p_len == (None,) * N

            # Check execution works:
            if name != "test":
                assert torch.all(symbolic_executor(scene_graphs, programs) == answers)


def test_clevr_user_provided_feats(tmp_path_factory, clean_h5, caplog):
    """
    Quick and ugly test of user-provided hdf5 files of features.
    Uses the hdf5 files produced by the dataloader
    itself under normal operation
    """
    data_dir = "./resources/r2/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    dm_img = CLEVRDataModule(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IMAGES,
        dm_shuffle_train_data=False,
        dm_batch_size=3,
    )

    dm_feat = CLEVRDataModule(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IEP_RESNET,
        dm_shuffle_train_data=False,
        dm_batch_size=3,
    )

    # Generate the HDF5 files of features
    dm_img.prepare_data()
    dm_feat.prepare_data()

    # Create new datamodule using the two created hdf5 files as "user provided" features.
    dm_user = CLEVRDataModule(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.USER_PROVIDED,
        dm_shuffle_train_data=False,
        dm_batch_size=3,
        user_image_features_train=[
            ("./resources/r2/hdf5_clevr/train_ims.h5", "features", "img"),
            ("./resources/r2/hdf5_clevr/train_features.h5", "features", "resnet"),
        ],
        user_image_features_val=[
            ("./resources/r2/hdf5_clevr/val_ims.h5", "features", "img"),
            ("./resources/r2/hdf5_clevr/val_features.h5", "features", "resnet"),
        ],
        user_image_features_test=[
            ("./resources/r2/hdf5_clevr/test_ims.h5", "features", "img"),
            ("./resources/r2/hdf5_clevr/test_features.h5", "features", "resnet"),
        ],
    )
    dm_user.prepare_data()

    test_loaders = []
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("test")
        test_loaders.append(dm.test_dataloader())

    for test_loader in test_loaders:
        assert len(test_loader.dataset) == 10

    for batch_user, batch_img, batch_feat in zip(*test_loaders):
        for i in range(9):
            if i not in [
                2,
                3,
                4,
                6,
                5,
                8,
            ]:  # should exactly match outside of image features
                assert torch.all(batch_user[i] == batch_img[i])
                assert torch.all(batch_user[i] == batch_feat[i])
            elif i in [3, 6, 4, 5, 8]:
                assert batch_user[i] == batch_img[i]
                assert batch_user[i] == batch_feat[i]
            else:  # Check image features match
                assert isinstance(batch_user[i], dict)
                assert torch.all(batch_user[i]["img"] == batch_img[i])
                assert torch.all(batch_user[i]["resnet"] == batch_feat[i])

    val_loaders = []
    train_loaders = []
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("fit")
        val_loaders.append(dm.val_dataloader())
        train_loaders.append(dm.train_dataloader())

    for val_loader in val_loaders:
        assert len(val_loader.dataset) == 7

    for batch_user, batch_img, batch_feat in zip(*val_loaders):
        for i in range(9):
            if i not in [2, 3]:  # should exactly match outside of image features
                assert torch.all(batch_user[i] == batch_img[i])
                assert torch.all(batch_user[i] == batch_feat[i])
            elif i in [3]:
                assert batch_user[i] == batch_img[i]
                assert batch_user[i] == batch_feat[i]
            else:  # Check image features match
                assert isinstance(batch_user[i], dict)
                assert torch.all(batch_user[i]["img"] == batch_img[i])
                assert torch.all(batch_user[i]["resnet"] == batch_feat[i])

    for train_loader in train_loaders:
        assert len(train_loader.dataset) == 10 * 4

    for batch_user, batch_img, batch_feat in zip(*train_loaders):
        for i in range(9):
            if i not in [2, 3]:  # should exactly match outside of image features
                assert torch.all(batch_user[i] == batch_img[i])
                assert torch.all(batch_user[i] == batch_feat[i])
            elif i in [3]:
                assert batch_user[i] == batch_img[i]
                assert batch_user[i] == batch_feat[i]
            else:  # Check image features match
                assert isinstance(batch_user[i], dict)
                assert torch.all(batch_user[i]["img"] == batch_img[i])
                assert torch.all(batch_user[i]["resnet"] == batch_feat[i])
