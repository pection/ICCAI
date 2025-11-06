"""
NOTE: visualization test is under test_clevr;
"""

import pytest
from vqa_framework.data_modules.clevr_loader import CLEVRDataModule, ClevrFeature
from vqa_framework.data_modules.closure_loader import (
    ClosureDataModule,
    ClosureFamily,
    ClosureTestType,
)

import os
import shutil
import torch
from vqa_framework.vr.preprocess import decode
from typing import Dict, Any
import logging
from vqa_framework.models.symbolic_exec import SymbolicClevrExecutor

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


# @pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.human
def test_closure_by_human_test(tmp_path_factory, clean_h5):  # , shuffle):
    logging.critical("HUMAN TEST STARTING")
    import matplotlib.pyplot as plt
    import numpy as np

    CHECK_EVERY = 10

    data_dir = "./resources/r3/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #
    dm = fixed_closure_init(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IMAGES,
        # dm_shuffle_train_data=shuffle,
    )
    dm.prepare_data()
    dm.setup("test")
    test_loader = dm.test_dataloader()

    vocab = dm.get_vocab()

    total_seen = 0
    for batch in test_loader:
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


# Individual tests fine, slow all together
@pytest.mark.slow
@pytest.mark.parametrize(
    "im_feat", [ClevrFeature.IMAGES, ClevrFeature.IEP_RESNET, None]
)
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("closure_test_type", [x for x in ClosureTestType])
@pytest.mark.parametrize("closure_test_fam", [x for x in ClosureFamily])
def test_dummy_closure_vals(
    tmp_path_factory,
    clean_h5,
    caplog,
    im_feat,
    shuffle,
    closure_test_type,
    closure_test_fam,
):
    """
    Quick test, check closure dataloaders have plausible data
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r3/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()
        dm = ClosureDataModule(
            closure_test_question_fam=closure_test_fam,
            closure_test_type=closure_test_type,
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            dm_batch_size=2,
            val_batch_size=4,
            image_features=im_feat,
            dm_shuffle_train_data=shuffle,
        )

        assert "Extracted closure dataset already exists" not in caplog.text
        assert PROCESSED_EXISTS_MSGS not in caplog.text

        dm.prepare_data()

        # Check CLEVR dataset did not re-generate stuff, on any repeat
        assert "CLEVR DATALOADER: Skipping generation" in caplog.text

        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "CLOSURE DATALOADER: Skipping generation" in caplog.text
        )
        assert "Extracted closure dataset already exists" in caplog.text

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is still empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this crashes
        dm.setup("test")
        test_loader = dm.test_dataloader()

        dm.setup("fit")
        val_loaders = dm.val_dataloader()

        with pytest.raises(NotImplementedError):
            dm.train_dataloader()

        all_dataloaders = [
            (loader, "val_" + ClosureFamily(i + 1).name)
            for i, loader in enumerate(val_loaders)
        ] + [(test_loader, "test")]

        for dl, name in all_dataloaders:

            if "val" in name:
                split_name = name.lower()
            else:
                assert name == "test"
                split_name = closure_test_fam.name.lower()

            if "embed_mat_spa" in split_name:
                expected_len = 41
            elif "or_mat_spa" in split_name:
                expected_len = 43
            elif "embed_spa_mat" in split_name:
                expected_len = 43
            elif "compare_mat_spa" in split_name:
                expected_len = 84
            elif "or_mat" in split_name:
                expected_len = 42
            elif "compare_mat" in split_name:
                expected_len = 84
            elif "and_mat_spa" in split_name:
                expected_len = 76
            else:
                raise NotImplementedError(
                    f"No expected len known for {name} {split_name}"
                )

            assert len(dl.dataset) == expected_len
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
                assert torch.all(indices <= expected_len)

                questions = x[0]
                assert isinstance(questions, torch.Tensor)
                assert questions.dim() == 2
                assert (
                    questions.size(0) == N
                )  # Note: dataset length goes into batch size

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

                answers = x[4]
                assert isinstance(answers, torch.Tensor)
                assert len(answers) == N

                programs = x[5]
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

                q_families = x[6]
                assert isinstance(q_families, torch.Tensor)
                assert q_families.dim() == 1
                assert q_families.size(0) == N
                #       NOTE!!! This is actually "type"
                #       which is *different* from CLEVR question family. GAH.
                #       See data_modules README for details.

                q_len = x[7]
                assert isinstance(q_len, torch.Tensor)
                assert q_len.dim() == 1
                assert q_len.size(0) == N

                for q, l in zip(questions.numpy().tolist(), q_len.numpy().tolist()):
                    assert vocab["question_idx_to_token"][q[l - 1]] == "<END>"

                p_len = x[8]
                assert isinstance(p_len, torch.Tensor)
                assert p_len.dim() == 1
                assert p_len.size(0) == N
                for pr, l in zip(programs.numpy().tolist(), p_len.numpy().tolist()):
                    assert vocab["program_idx_to_token"][pr[l - 1]] == "<END>"

                # Check execution works:
                assert torch.all(symbolic_executor(scene_graphs, programs) == answers)


@pytest.mark.parametrize(
    "im_feat", [ClevrFeature.IMAGES, ClevrFeature.IEP_RESNET, None]
)
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("closure_test_type", [ClosureTestType(1)])
@pytest.mark.parametrize("closure_test_fam", [ClosureFamily(1)])
def test_dummy_closure_vals_quick(
    tmp_path_factory,
    clean_h5,
    caplog,
    im_feat,
    shuffle,
    closure_test_type,
    closure_test_fam,
):
    """
    Quick test, check closure dataloaders have plausible data
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r3/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()
        dm = ClosureDataModule(
            closure_test_question_fam=closure_test_fam,
            closure_test_type=closure_test_type,
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            dm_batch_size=2,
            val_batch_size=4,
            image_features=im_feat,
            dm_shuffle_train_data=shuffle,
        )

        assert "Extracted closure dataset already exists" not in caplog.text
        assert PROCESSED_EXISTS_MSGS not in caplog.text

        dm.prepare_data()

        # Check CLEVR dataset did not re-generate stuff, on any repeat
        assert "CLEVR DATALOADER: Skipping generation" in caplog.text

        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "CLOSURE DATALOADER: Skipping generation" in caplog.text
        )
        assert "Extracted closure dataset already exists" in caplog.text

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is still empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this crashes
        dm.setup("test")
        test_loader = dm.test_dataloader()

        dm.setup("fit")
        val_loaders = dm.val_dataloader()

        with pytest.raises(NotImplementedError):
            dm.train_dataloader()

        all_dataloaders = [
            (loader, "val_" + ClosureFamily(i + 1).name)
            for i, loader in enumerate(val_loaders)
        ] + [(test_loader, "test")]

        for dl, name in all_dataloaders:

            if "val" in name:
                split_name = name.lower()
            else:
                assert name == "test"
                split_name = closure_test_fam.name.lower()

            if "embed_mat_spa" in split_name:
                expected_len = 41
            elif "or_mat_spa" in split_name:
                expected_len = 43
            elif "embed_spa_mat" in split_name:
                expected_len = 43
            elif "compare_mat_spa" in split_name:
                expected_len = 84
            elif "or_mat" in split_name:
                expected_len = 42
            elif "compare_mat" in split_name:
                expected_len = 84
            elif "and_mat_spa" in split_name:
                expected_len = 76
            else:
                raise NotImplementedError(
                    f"No expected len known for {name} {split_name}"
                )

            assert len(dl.dataset) == expected_len
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
                assert torch.all(indices <= expected_len)

                questions = x[0]
                assert isinstance(questions, torch.Tensor)
                assert questions.dim() == 2
                assert (
                    questions.size(0) == N
                )  # Note: dataset length goes into batch size

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

                answers = x[4]
                assert isinstance(answers, torch.Tensor)
                assert len(answers) == N

                programs = x[5]
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

                q_families = x[6]
                assert isinstance(q_families, torch.Tensor)
                assert q_families.dim() == 1
                assert q_families.size(0) == N
                #       NOTE!!! This is actually "type"
                #       which is *different* from CLEVR question family. GAH.
                #       See data_modules README for details.

                q_len = x[7]
                assert isinstance(q_len, torch.Tensor)
                assert q_len.dim() == 1
                assert q_len.size(0) == N

                for q, l in zip(questions.numpy().tolist(), q_len.numpy().tolist()):
                    assert vocab["question_idx_to_token"][q[l - 1]] == "<END>"

                p_len = x[8]
                assert isinstance(p_len, torch.Tensor)
                assert p_len.dim() == 1
                assert p_len.size(0) == N
                for pr, l in zip(programs.numpy().tolist(), p_len.numpy().tolist()):
                    assert vocab["program_idx_to_token"][pr[l - 1]] == "<END>"

                # Check execution works:
                assert torch.all(symbolic_executor(scene_graphs, programs) == answers)


def test_closure_user_provided_feats(tmp_path_factory, clean_h5, caplog):
    """
    Quick and ugly test of user-provided hdf5 files of features.
    Uses the hdf5 files produced by the dataloader itself under normal operation ;)
    """
    data_dir = "./resources/r3/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    dm_img = ClosureDataModule(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IMAGES,
        dm_shuffle_train_data=False,
        dm_batch_size=3,
        closure_test_type=ClosureTestType.TEST,
        closure_test_question_fam=ClosureFamily.AND_MAT_SPA,
    )

    dm_feat = ClosureDataModule(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IEP_RESNET,
        dm_shuffle_train_data=False,
        dm_batch_size=3,
        closure_test_type=ClosureTestType.TEST,
        closure_test_question_fam=ClosureFamily.AND_MAT_SPA,
    )

    # Generate the HDF5 files of features
    dm_img.prepare_data()
    dm_feat.prepare_data()

    # Create new datamodule using the two created hdf5 files as "user provided" features.
    dm_user = ClosureDataModule(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.USER_PROVIDED,
        dm_shuffle_train_data=False,
        dm_batch_size=3,
        closure_test_type=ClosureTestType.TEST,
        closure_test_question_fam=ClosureFamily.AND_MAT_SPA,
        # Note: The CLEVR validation split files are used here
        user_image_features=[
            ("./resources/r3/hdf5_clevr/val_ims.h5", "features", "img"),
            ("./resources/r3/hdf5_clevr/val_features.h5", "features", "resnet"),
        ],
    )
    dm_user.prepare_data()

    test_loaders = []
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("test")
        test_loaders.append(dm.test_dataloader())

    for test_loader in test_loaders:
        assert len(test_loader.dataset) == 76

    for batch_user, batch_img, batch_feat in zip(*test_loaders):
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

    val_loader_tuples = []
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("fit")
        val_loader_tuples.append(dm.val_dataloader())

    for j in range(len(val_loader_tuples[0])):
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[1][j].dataset
        )
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[2][j].dataset
        )

    assert len(val_loader_tuples[0]) == len(val_loader_tuples[1])
    assert len(val_loader_tuples[0]) == len(val_loader_tuples[2])

    for j in range(len(val_loader_tuples[0])):  # iterate over CLOSURE subsets
        for batch_user, batch_img, batch_feat in zip(
            val_loader_tuples[0][j],
            val_loader_tuples[1][j],
            val_loader_tuples[2][j],
        ):
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
