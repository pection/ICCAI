"""
NOTE: visualization test is under test_clevr;
"""

import pytest
from vqa_framework.data_modules.clevr_loader import ClevrFeature
from vqa_framework.data_modules.generic_clevr_loader import GenericCLEVRDataModule

import os
import shutil
import torch
from vqa_framework.vr.preprocess import decode
from typing import Dict, Any
import logging
from vqa_framework.models.symbolic_exec import SymbolicClevrExecutor

TEST_FILES = [
    "and_mat_spa_test.json",
    "embed_mat_spa_test.json",
    "or_mat_test.json",
    "compare_mat_spa_test.json",
    "embed_spa_mat_test.json",
    "compare_mat_test.json",
    "or_mat_spa_test.json",
]

VAL_FILES = [
    "and_mat_spa_val.json",
    "embed_mat_spa_val.json",
    "or_mat_val.json",
    "compare_mat_spa_val.json",
    "embed_spa_mat_val.json",
    "compare_mat_val.json",
    "or_mat_spa_val.json",
]


@pytest.fixture
def clean_h5(tmpdir):
    """Clean processed data b/f and after test"""
    # Setup: fill with any logic you want
    if os.path.exists(os.path.join(".", "resources", "r2", "hdf5_clevr")):
        shutil.rmtree(os.path.join(".", "resources", "r2", "hdf5_clevr"))
    if os.path.exists(os.path.join(".", "resources", "r3", "hdf5_closure")):
        shutil.rmtree(os.path.join(".", "resources", "r3", "hdf5_closure"))
    for filename in [
        "gen_img_dir",
        "gen_ques_dir",
        "gen1_img_dir",
        "gen1_ques_dir",
        "gen2_img_dir",
        "gen2_ques_dir",
    ]:
        if os.path.exists(os.path.join(".", "resources", "r6", filename)):
            shutil.rmtree(os.path.join(".", "resources", "r6", filename))

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    if os.path.exists(os.path.join(".", "resources", "r2", "hdf5_clevr")):
        shutil.rmtree(os.path.join(".", "resources", "r2", "hdf5_clevr"))
    if os.path.exists(os.path.join(".", "resources", "r3", "hdf5_closure")):
        shutil.rmtree(os.path.join(".", "resources", "r3", "hdf5_closure"))
    for filename in [
        "gen_img_dir",
        "gen_ques_dir",
        "gen1_img_dir",
        "gen1_ques_dir",
        "gen2_img_dir",
        "gen2_ques_dir",
    ]:
        if os.path.exists(os.path.join(".", "resources", "r6", filename)):
            shutil.rmtree(os.path.join(".", "resources", "r6", filename))


def fixed_generic_clevr_init(
    data_dir,
    tmp_dir,
    image_features,
    dm_shuffle_train_data=False,
    user_image_features_train=None,
    user_image_features_val=None,
    user_image_features_test=None,
    dm_images_name="gen_img_dir",
    dm_questions_name="gen_ques_dir",
    pin_memory=False,
    drop_last_train=False,
):
    return GenericCLEVRDataModule(
        dm_images_name=dm_images_name,
        dm_questions_name=dm_questions_name,  # And the rest! Here on Gilligan's Isle!
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
        dm_batch_size=2,
        val_batch_size=4,
        user_image_features_val=user_image_features_val,
        user_image_features_test=user_image_features_test,
        user_image_features_train=user_image_features_train,
        pin_memory=pin_memory,
        drop_last_train=drop_last_train,
    )


@pytest.mark.human
def test_generic_by_human_test(tmp_path_factory, clean_h5):
    logging.critical("HUMAN TEST STARTING")
    import matplotlib.pyplot as plt
    import numpy as np

    CHECK_EVERY = 30

    data_dir = "./resources/r6/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #
    dm = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        image_features=ClevrFeature.IMAGES,
        # dm_shuffle_train_data=shuffle,
    )
    dm.prepare_data()
    dm.setup("test")
    test_loaders = dm.test_dataloader()

    vocab = dm.get_vocab()
    total_seen = 0
    for loader_num, test_loader in enumerate(test_loaders):
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
                logging.critical(
                    f"QUESTION from Loader {loader_num}; {TEST_FILES[loader_num]}:"
                )
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

                plt.imshow(
                    features[i].cpu().numpy().transpose(1, 2, 0).astype(np.int32)
                )
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


@pytest.mark.parametrize(
    "im_feat", [ClevrFeature.IMAGES, ClevrFeature.IEP_RESNET, None]
)
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("pin", [True, False])
@pytest.mark.parametrize("drop_last_train", [True, False])
def test_dummy_closure_vals(
    tmp_path_factory,
    clean_h5,
    caplog,
    im_feat,
    shuffle,
    pin,
    drop_last_train,
):
    """
    Quick test, check closure dataloaders have plausible data
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r6/"

    for repeat in range(3):
        caplog.clear()
        dm = fixed_generic_clevr_init(
            data_dir=data_dir,
            tmp_dir=None,
            image_features=im_feat,
            dm_shuffle_train_data=shuffle,
            pin_memory=pin,
            drop_last_train=drop_last_train,
        )

        assert (
            "Generic CLEVR: Skipping image processing; directory already exists"
            not in caplog.text
        )
        assert f"CLEVR DATALOADER: Processing train questions" not in caplog.text
        assert f"CLEVR DATALOADER: Processing val questions" not in caplog.text
        assert f"CLEVR DATALOADER: Processing test questions" not in caplog.text

        dm.prepare_data()

        # check dataset only exists after the first time we create DataModule
        for msg in (
            [
                f"CLEVR DATALOADER: Skipping generation of test_{x[:-4]}h5; file already exists"
                for x in TEST_FILES
            ]
            + [
                f"CLEVR DATALOADER: Skipping generation of val_{x[:-4]}h5; file already exists"
                for x in VAL_FILES
            ]
            + [
                f"CLEVR DATALOADER: Skipping generation of train_and_mat_spa_baseline.h5; file already exists"
            ]
        ):
            assert (repeat == 0) != (msg in caplog.text)

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.QUESTIONS_LOCAL_NAME))
        if im_feat is not None:
            assert os.path.isdir(os.path.join(data_dir, dm.IMAGES_LOCAL_NAME))

        # check this doesn't crash
        dm.setup("test")
        test_loaders = dm.test_dataloader()

        dm.setup("fit")
        val_loaders = dm.val_dataloader()
        train_loader = dm.train_dataloader()

        for dl, name in zip(
            val_loaders + test_loaders + [train_loader],
            VAL_FILES + TEST_FILES + ["and_mat_spa_baseline.json"],
        ):

            split_name = name.lower()

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
                if "baseline" in name:  # Train batch size
                    N = 2
                else:  # Val/Test batch size
                    N = 4

                if (
                    drop_last_train and "baseline" in name
                ):  # If in train batch & dropping the last
                    # drop_last_train should skip the final batch of less than
                    # N items, so we should always have at least N items
                    # that we haven't seen yet.
                    assert remaining_items >= N

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

                if pin:
                    # Bug in pytorch causes this conversion https://github.com/pytorch/pytorch/issues/48419
                    assert isinstance(scene_graphs, list)
                else:
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
                if not torch.all(symbolic_executor(scene_graphs, programs) == answers):
                    logging.critical(name)
                    logging.critical(indices)
                    logging.critical(scene_graphs)
                    # logging.critical(programs)
                    for pr in programs.numpy().tolist():
                        logging.critical([vocab["program_idx_to_token"][x] for x in pr])
                assert torch.all(symbolic_executor(scene_graphs, programs) == answers)

            # Unless we're skipping the last train batch, assert we've seen all the data.
            if not (drop_last_train and "baseline" in name):
                assert remaining_items == 0


@pytest.mark.parametrize("shuffle", [True, False])
def test_dummy_closure_vals_different_feats(
    tmp_path_factory,
    clean_h5,
    caplog,
    shuffle,
):
    """
    Quick test, check it works if we create 2 dataloaders sequentially using
    different features.

    Regression test; there was a bug where the existence of one feature type
    would prevent the generation of the other feature type.
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r6/"

    for im_feat in [ClevrFeature.IMAGES, ClevrFeature.IMAGES, None]:
        for repeat in range(3):
            caplog.clear()
            dm = fixed_generic_clevr_init(
                data_dir=data_dir,
                tmp_dir=None,
                image_features=im_feat,
                dm_shuffle_train_data=shuffle,
            )

            assert (
                "Generic CLEVR: Skipping image processing; directory already exists"
                not in caplog.text
            )
            assert f"CLEVR DATALOADER: Processing train questions" not in caplog.text
            assert f"CLEVR DATALOADER: Processing val questions" not in caplog.text
            assert f"CLEVR DATALOADER: Processing test questions" not in caplog.text

            dm.prepare_data()

            # check processed dataset was created
            assert os.path.isdir(os.path.join(data_dir, dm.QUESTIONS_LOCAL_NAME))
            if im_feat is not None:
                assert os.path.isdir(os.path.join(data_dir, dm.IMAGES_LOCAL_NAME))

            # check this doesn't crash
            dm.setup("test")
            test_loaders = dm.test_dataloader()

            dm.setup("fit")
            val_loaders = dm.val_dataloader()
            train_loader = dm.train_dataloader()

            for dl, name in zip(
                val_loaders + test_loaders + [train_loader],
                VAL_FILES + TEST_FILES + ["and_mat_spa_baseline.json"],
            ):

                split_name = name.lower()

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
                    if "baseline" in name:  # Train batch size
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
                    if not torch.all(
                        symbolic_executor(scene_graphs, programs) == answers
                    ):
                        logging.critical(name)
                        logging.critical(indices)
                        logging.critical(scene_graphs)
                        # logging.critical(programs)
                        for pr in programs.numpy().tolist():
                            logging.critical(
                                [vocab["program_idx_to_token"][x] for x in pr]
                            )
                    assert torch.all(
                        symbolic_executor(scene_graphs, programs) == answers
                    )


def test_closure_user_provided_feats(tmp_path_factory, clean_h5, caplog):
    """
    Quick and ugly test of user-provided hdf5 files of features.
    Uses the hdf5 files produced by the dataloader itself under normal operation ;)
    """
    data_dir = "./resources/r6/"

    dm_img = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.IMAGES,
        dm_shuffle_train_data=False,
        dm_images_name="gen1_img_dir",
        dm_questions_name="gen1_ques_dir",
    )

    dm_feat = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.IEP_RESNET,
        dm_shuffle_train_data=False,
        dm_images_name="gen2_img_dir",
        dm_questions_name="gen2_ques_dir",
    )

    # Generate the HDF5 files of features
    dm_img.prepare_data()
    dm_feat.prepare_data()

    # Create new datamodule using the two created hdf5 files as "user provided" features.
    dm_user = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.USER_PROVIDED,
        dm_shuffle_train_data=False,
        user_image_features_train=[
            ("./resources/r6/gen1_img_dir/train_ims.h5", "features", "img"),
            ("./resources/r6/gen2_img_dir/train_features.h5", "features", "resnet"),
        ],
        user_image_features_val=[
            ("./resources/r6/gen1_img_dir/val_ims.h5", "features", "img"),
            ("./resources/r6/gen2_img_dir/val_features.h5", "features", "resnet"),
        ],
        user_image_features_test=[
            ("./resources/r6/gen1_img_dir/test_ims.h5", "features", "img"),
            ("./resources/r6/gen2_img_dir/test_features.h5", "features", "resnet"),
        ],
    )
    dm_user.prepare_data()

    test_loaders = []  # List of lists of dataloaders
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("test")
        test_loaders.append(dm.test_dataloader())

    for test_loader in test_loaders:
        assert len(test_loader[0].dataset) == 76
        assert len(test_loader[-1].dataset) == 43

    for tl_u, tl_i, tl_f in zip(*test_loaders):
        for batch_user, batch_img, batch_feat in zip(tl_u, tl_i, tl_f):
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
    train_loaders = []
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("fit")
        val_loader_tuples.append(dm.val_dataloader())
        train_loaders.append(dm.train_dataloader())

    for j in range(len(val_loader_tuples[0])):
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[1][j].dataset
        )
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[2][j].dataset
        )

    assert len(val_loader_tuples[0]) == len(val_loader_tuples[1])
    assert len(val_loader_tuples[0]) == len(val_loader_tuples[2])

    for j in range(len(val_loader_tuples[0])):  # iterate over dataloaders
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

    for train_loader in train_loaders:
        assert len(train_loader.dataset) == 76

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


def test_closure_user_provided_feats_no_im(tmp_path_factory, clean_h5, caplog):
    """
    Quick and ugly test of user-provided hdf5 files of features.
    Uses the hdf5 files produced by the dataloader itself under normal operation ;)

    Regression test: Make sure it works if we disable image outputs
    """
    data_dir = "./resources/r6/"

    dm_img = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.IMAGES,
        dm_shuffle_train_data=False,
        dm_images_name="gen1_img_dir",
        dm_questions_name="gen1_ques_dir",
    )

    # Generate the HDF5 files of features
    dm_img.prepare_data()

    # Create new datamodule using the two created hdf5 files as "user provided" features.
    dm_user = GenericCLEVRDataModule(
        dm_images_name=None,
        dm_questions_name="gen1_ques_dir",  # And the rest! Here on Gilligan's Isle!
        dm_train_images=None,
        dm_val_images=None,
        dm_test_images=None,
        dm_train_scenes=None,
        dm_val_scenes=None,
        dm_test_scenes=None,
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
        image_features=None,
        dm_shuffle_train_data=False,
        dm_batch_size=2,
        val_batch_size=4,
        user_image_features_val=None,
        user_image_features_test=None,
        user_image_features_train=None,
    )

    dm_user.prepare_data()

    test_loaders = []  # List of lists of dataloaders
    for dm in [dm_user, dm_img]:
        dm.setup("test")
        test_loaders.append(dm.test_dataloader())

    for test_loader in test_loaders:
        assert len(test_loader[0].dataset) == 76
        assert len(test_loader[-1].dataset) == 43

    for tl_u, tl_i in zip(*test_loaders):
        for batch_user, batch_img in zip(tl_u, tl_i):
            for i in range(9):
                if i not in [
                    2,
                    3,
                ]:  # should exactly match outside of image features & scenes
                    assert torch.all(batch_user[i] == batch_img[i])
                elif i == 3:  # Check image features match
                    assert isinstance(batch_user[i], tuple)
                    assert all(x is None for x in batch_user[i])
                else:
                    assert i == 2
                    assert torch.all(batch_user[i] == 0)

    val_loader_tuples = []
    train_loaders = []
    for dm in [dm_user, dm_img]:
        dm.setup("fit")
        val_loader_tuples.append(dm.val_dataloader())
        train_loaders.append(dm.train_dataloader())

    for j in range(len(val_loader_tuples[0])):
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[1][j].dataset
        )

    assert len(val_loader_tuples[0]) == len(val_loader_tuples[1])

    for j in range(len(val_loader_tuples[0])):  # iterate over dataloaders
        for batch_user, batch_img in zip(
            val_loader_tuples[0][j],
            val_loader_tuples[1][j],
        ):
            for i in range(9):
                if i not in [
                    2,
                    3,
                ]:  # should exactly match outside of image features & scenes
                    assert torch.all(batch_user[i] == batch_img[i])
                elif i == 3:  # Check image features match
                    assert isinstance(batch_user[i], tuple)
                    assert all(x is None for x in batch_user[i])
                else:
                    assert i == 2
                    assert torch.all(batch_user[i] == 0)

    for train_loader in train_loaders:
        assert len(train_loader.dataset) == 76

    for batch_user, batch_img in zip(*train_loaders):
        for i in range(9):
            if i not in [
                2,
                3,
            ]:  # should exactly match outside of image features & scenes
                assert torch.all(batch_user[i] == batch_img[i])
            elif i == 3:  # Check image features match
                assert isinstance(batch_user[i], tuple)
                assert all(x is None for x in batch_user[i])
            else:
                assert i == 2
                assert torch.all(batch_user[i] == 0)


def test_closure_user_provided_feats_no_name(tmp_path_factory, clean_h5, caplog):
    """
    Quick and ugly test of user-provided hdf5 files of features.
    Uses the hdf5 files produced by the dataloader itself under normal operation ;)

    Regression test: Make sure we can say the image name is None when we're
    providing user features.
    """
    data_dir = "./resources/r6/"

    dm_img = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.IMAGES,
        dm_shuffle_train_data=False,
        dm_images_name="gen1_img_dir",
        dm_questions_name="gen1_ques_dir",
    )

    dm_feat = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.IEP_RESNET,
        dm_shuffle_train_data=False,
        dm_images_name="gen2_img_dir",
        dm_questions_name="gen2_ques_dir",
    )

    # Generate the HDF5 files of features
    dm_img.prepare_data()
    dm_feat.prepare_data()

    # Create new datamodule using the two created hdf5 files as "user provided" features.
    dm_user = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.USER_PROVIDED,
        dm_shuffle_train_data=False,
        user_image_features_train=[
            ("./resources/r6/gen1_img_dir/train_ims.h5", "features", "img"),
            ("./resources/r6/gen2_img_dir/train_features.h5", "features", "resnet"),
        ],
        user_image_features_val=[
            ("./resources/r6/gen1_img_dir/val_ims.h5", "features", "img"),
            ("./resources/r6/gen2_img_dir/val_features.h5", "features", "resnet"),
        ],
        user_image_features_test=[
            ("./resources/r6/gen1_img_dir/test_ims.h5", "features", "img"),
            ("./resources/r6/gen2_img_dir/test_features.h5", "features", "resnet"),
        ],
        dm_images_name=None,  # tada! This should be able to be None
    )
    dm_user.prepare_data()

    test_loaders = []  # List of lists of dataloaders
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("test")
        test_loaders.append(dm.test_dataloader())

    for test_loader in test_loaders:
        assert len(test_loader[0].dataset) == 76
        assert len(test_loader[-1].dataset) == 43

    for tl_u, tl_i, tl_f in zip(*test_loaders):
        for batch_user, batch_img, batch_feat in zip(tl_u, tl_i, tl_f):
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
    train_loaders = []
    for dm in [dm_user, dm_img, dm_feat]:
        dm.setup("fit")
        val_loader_tuples.append(dm.val_dataloader())
        train_loaders.append(dm.train_dataloader())

    for j in range(len(val_loader_tuples[0])):
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[1][j].dataset
        )
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[2][j].dataset
        )

    assert len(val_loader_tuples[0]) == len(val_loader_tuples[1])
    assert len(val_loader_tuples[0]) == len(val_loader_tuples[2])

    for j in range(len(val_loader_tuples[0])):  # iterate over dataloaders
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

    for train_loader in train_loaders:
        assert len(train_loader.dataset) == 76

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


def test_closure_user_provided_feats_direct_path(tmp_path_factory, clean_h5, caplog):
    """
    Quick and ugly test of user-provided hdf5 files of features.
    Uses the hdf5 files produced by the dataloader itself under normal operation ;)

    Tests that we can pass in the path to a feature file *directly*, instead
    of going with the tuple.
    """
    data_dir = "./resources/r6/"

    dm_img = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.IMAGES,
        dm_shuffle_train_data=False,
        dm_images_name="gen1_img_dir",
        dm_questions_name="gen1_ques_dir",
    )

    # Generate the HDF5 files of features
    dm_img.prepare_data()

    # Create new datamodule using the two created hdf5 files as "user provided" features.
    dm_user = fixed_generic_clevr_init(
        data_dir=data_dir,
        tmp_dir=None,
        image_features=ClevrFeature.USER_PROVIDED,
        dm_shuffle_train_data=False,
        user_image_features_train="./resources/r6/gen1_img_dir/train_ims.h5",
        user_image_features_val="./resources/r6/gen1_img_dir/val_ims.h5",
        user_image_features_test="./resources/r6/gen1_img_dir/test_ims.h5",
        dm_images_name=None,  # tada! This should be able to be None
    )
    dm_user.prepare_data()

    test_loaders = []  # List of lists of dataloaders
    for dm in [dm_user, dm_img]:
        dm.setup("test")
        test_loaders.append(dm.test_dataloader())

    for test_loader in test_loaders:
        assert len(test_loader[0].dataset) == 76
        assert len(test_loader[-1].dataset) == 43

    for tl_u, tl_i in zip(*test_loaders):
        for batch_user, batch_img in zip(tl_u, tl_i):
            for i in range(9):
                if i not in [2, 3]:  # should exactly match outside of image features
                    assert torch.all(batch_user[i] == batch_img[i])
                elif i in [3]:
                    assert batch_user[i] == batch_img[i]
                else:  # Check image features match
                    assert isinstance(batch_user[i], torch.Tensor)
                    assert torch.all(batch_user[i] == batch_img[i])

    val_loader_tuples = []
    train_loaders = []
    for dm in [dm_user, dm_img]:
        dm.setup("fit")
        val_loader_tuples.append(dm.val_dataloader())
        train_loaders.append(dm.train_dataloader())

    for j in range(len(val_loader_tuples[0])):
        assert len(val_loader_tuples[0][j].dataset) == len(
            val_loader_tuples[1][j].dataset
        )

    assert len(val_loader_tuples[0]) == len(val_loader_tuples[1])

    for j in range(len(val_loader_tuples[0])):  # iterate over dataloaders
        for batch_user, batch_img in zip(
            val_loader_tuples[0][j],
            val_loader_tuples[1][j],
        ):
            for i in range(9):
                if i not in [2, 3]:  # should exactly match outside of image features
                    assert torch.all(batch_user[i] == batch_img[i])
                elif i in [3]:
                    assert batch_user[i] == batch_img[i]
                else:  # Check image features match
                    assert isinstance(batch_user[i], torch.Tensor)
                    assert torch.all(batch_user[i] == batch_img[i])

    for train_loader in train_loaders:
        assert len(train_loader.dataset) == 76

    for batch_user, batch_img in zip(*train_loaders):
        for i in range(9):
            if i not in [2, 3]:  # should exactly match outside of image features
                assert torch.all(batch_user[i] == batch_img[i])
            elif i in [3]:
                assert batch_user[i] == batch_img[i]
            else:  # Check image features match
                assert isinstance(batch_user[i], torch.Tensor)
                assert torch.all(batch_user[i] == batch_img[i])
