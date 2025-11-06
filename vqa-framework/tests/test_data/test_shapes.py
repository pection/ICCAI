import pytest
from vqa_framework.data_modules.shapes_loader import SHAPESDataModule
import os
import shutil
import torch
from vqa_framework.vr.preprocess import decode
from typing import Dict, Any
import logging
from vqa_framework.models.symbolic_exec import SymbolicShapesExecutor


@pytest.fixture
def clean_h5(tmpdir):
    """Clean processed data b/f and after test"""
    # Setup: fill with any logic you want
    if os.path.exists(os.path.join(".", "resources", "r1", "hdf5_shapes")):
        shutil.rmtree(os.path.join(".", "resources", "r1", "hdf5_shapes"))
    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    shutil.rmtree(os.path.join(".", "resources", "r1", "hdf5_shapes"))


@pytest.mark.parametrize("stateful", [(True,), (False,)])
def test_dummy_SHAPES_prepare_exists(tmp_path_factory, clean_h5, caplog, stateful):
    """
    Minimal test, check just that SHAPES Data module can processed
    _something_, and that it doesn't crash in the subsequent steps.

    Also checks resulting dataloaders contain the correct number of items, and
    that if you run the DataModule again, it won't re-process the data.

    This test assumes the repo has already been downloaded, and tests
    accordingly.
    """
    caplog.set_level(logging.INFO)
    data_dir = "./resources/r1/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()
        dm = SHAPESDataModule(data_dir=data_dir, tmp_dir=tmp_dir, stateful=stateful)

        assert "Repository already downloaded" not in caplog.text
        assert "Processed SHAPES dataset already exists" not in caplog.text

        dm.prepare_data()

        assert "Repository already downloaded" in caplog.text
        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "Processed SHAPES dataset already exists" in caplog.text
        )

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is again empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        assert len(test_loader.dataset) == 64

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert len(val_loader.dataset) == 64
        train_loader = dm.train_dataloader()
        assert len(train_loader.dataset) == 64 * 4  # tiny, repeated four times


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
    assert pos[2] == 0
    assert pos[0] in [0, 1, 2]
    assert pos[1] in [0, 1, 2]

    assert sg_obj["color"] in ["blue", "green", "red"]

    assert sg_obj["material"] == "rubber"

    assert sg_obj["shape"] in ["cube", "cylinder", "sphere"]

    assert sg_obj["size"] == "small"


@pytest.mark.parametrize("stateful", [(True,), (False,)])
def test_dummy_SHAPES_vals(tmp_path_factory, clean_h5, caplog, stateful):
    """
    Quick test, check dataloaders have the correct data (at least, when
    shuffling is off)
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r1/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()
        dm = SHAPESDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            dm_shuffle_train_data=False,
            dm_batch_size=2,
            val_batch_size=4,
            stateful=stateful,
        )

        assert "Repository already downloaded" not in caplog.text
        assert "Processed SHAPES dataset already exists" not in caplog.text

        dm.prepare_data()

        assert "Repository already downloaded" in caplog.text
        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "Processed SHAPES dataset already exists" in caplog.text
        )

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
        # check_dataloader(val_loader, dm)

        train_loader = dm.train_dataloader()
        # check_dataloader(train_loader, dm, repeats=4)

        for dl, repeats in [(test_loader, 1), (val_loader, 1), (train_loader, 4)]:
            assert len(dl.dataset) == 64 * repeats
            vocab = dm.vocab
            assert vocab is not None

            symbolic_executor = SymbolicShapesExecutor(vocab)

            for x in dl:
                # print(x)
                # text indices and ??? (2-element list containing: Nxtextlen tensor, and N tensor)
                # images, (Nx3x30x30 tensor)
                # symbolic repr (tuple of scenegraphs; each is a list of dicts, ea. dict represents an object)
                #   'id', 'position', 'color', 'material', 'shape', 'size'
                # ?Answer? (N tensor)
                # question index, (N tensor)
                if repeats > 1:  # Train batch size
                    N = 2
                else:  # Val/Test batch size
                    N = 4

                C = 3  # 3-channel images

                questions = x[0]
                assert isinstance(questions, torch.Tensor)
                assert questions.dim() == 2
                assert (
                    questions.size(0) == N
                )  # Note: dataset length goes into batch size
                assert questions.size(1) == 9 + 2  # 9 words, plus start & end tokens
                for qu in questions:
                    assert (
                        decode(
                            qu.numpy().tolist(),
                            vocab["question_idx_to_token"],
                            delim=" ",
                        )
                        == "<START> is a green shape left of a red shape <END>"
                    )

                indices = x[1]
                assert isinstance(indices, torch.Tensor)
                assert indices.size() == (N,)
                # logging.warning(indices)
                assert torch.all(indices <= 64 * repeats)

                images = x[2]
                assert isinstance(images, torch.Tensor)
                assert images.dim() == 4
                assert images.size() == (N, C, 30, 30)
                assert torch.all(0 <= images)  # SHAPES images are normalized
                assert torch.all(images <= 1)  # SHAPES images are normalized

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
                assert torch.all(
                    torch.logical_or(answers == 4, answers == 5)
                )  # All answers are true or false.

                programs = x[5]
                assert isinstance(programs, torch.Tensor)
                assert programs.dim() == 2
                assert (
                    programs.size(0) == N
                )  # Note: dataset length goes into batch size
                assert (
                    programs.size(1) == 5 + 2
                )  # 5 primitives, plus start & end tokens
                for pr in programs:
                    assert (
                        decode(
                            pr.numpy().tolist(),
                            vocab["program_idx_to_token"],
                            delim=" ",
                        )
                        == "<START> _Answer _And _Transform[left_of] _Find[red] _Find[green] <END>"
                    )

                q_families = x[6]
                assert isinstance(q_families, torch.Tensor)
                assert q_families.dim() == 1
                assert q_families.size(0) == N
                assert all(q_families == 0)  # For this setup, all families are 0

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


@pytest.mark.slow
@pytest.mark.parametrize("stateful", [(True,), (False,)])
def test_SHAPES_prepare_exists(tmp_path_factory, stateful):
    """
    Minimal test, check just it downloaded and processed _something_, and
    doesn't crash in the subsequent steps.

    Also check resulting dataloaders contain the correct number of items.
    """
    data_dir = tmp_path_factory.mktemp("target")
    tmp_dir = tmp_path_factory.mktemp("tmp")
    dm = SHAPESDataModule(data_dir=data_dir, tmp_dir=tmp_dir, stateful=stateful)
    dm.prepare_data()
    # check SHAPES dataset was downloaded
    assert os.path.isdir(os.path.join(data_dir, dm.ORI_NAME))

    # check processed dataset was created
    assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

    # check that tmp directory is again empty
    assert len(os.listdir(tmp_dir)) == 0

    # check this doesn't crash
    dm.setup("test")
    test_loader = dm.test_dataloader()
    assert len(test_loader.dataset) == 1024

    dm.setup("fit")
    val_loader = dm.val_dataloader()
    assert len(val_loader.dataset) == 1024
    train_loader = dm.train_dataloader()
    assert (
        len(train_loader.dataset) == 64 + 640 + 6400 + 13568
    )  # tiny + small + medium + large

    symbolic_executor = SymbolicShapesExecutor(dm.vocab)
    for dl, repeats in [(test_loader, 1), (val_loader, 1), (train_loader, 4)]:
        vocab = dm.vocab
        assert vocab is not None
        for x in dl:
            questions = x[0]
            assert isinstance(questions, torch.Tensor)
            assert questions.dim() == 2
            for qu in questions:
                qu_eng = decode(
                    qu.numpy().tolist(), vocab["question_idx_to_token"], delim=" "
                )
                assert qu_eng.startswith("<START>")
                assert qu_eng.endswith("<END>")

            indices = x[1]
            assert isinstance(indices, torch.Tensor)
            assert indices.dim() == 1

            images = x[2]
            assert isinstance(images, torch.Tensor)
            assert images.dim() == 4
            assert torch.all(0 <= images)  # SHAPES images are normalized
            assert torch.all(images <= 1)  # SHAPES images are normalized

            scene_graphs = x[3]
            assert isinstance(scene_graphs, tuple)
            for sg in scene_graphs:
                assert isinstance(sg, list)
                for obj in sg:
                    assert isinstance(obj, dict)
                    for key in ["id", "position", "color", "material", "shape", "size"]:
                        assert key in obj
                    check_scene_graph_obj(obj)

            answers = x[4]
            assert isinstance(answers, torch.Tensor)
            assert answers.dim() == 1
            assert torch.all(
                torch.logical_or(answers == 4, answers == 5)
            )  # All answers are true or false.

            programs = x[5]
            assert isinstance(programs, torch.Tensor)
            assert programs.dim() == 2
            for pr in programs:
                pr_eng = decode(
                    pr.numpy().tolist(), vocab["program_idx_to_token"], delim=" "
                )
                assert pr_eng.startswith("<START>")
                assert pr_eng.endswith("<END>")

            q_families = x[6]
            assert isinstance(q_families, torch.Tensor)
            assert q_families.dim() == 1

            q_len = x[7]
            assert isinstance(q_len, torch.Tensor)
            assert q_len.dim() == 1

            for q, l in zip(questions.numpy().tolist(), q_len.numpy().tolist()):
                assert vocab["question_idx_to_token"][q[l - 1]] == "<END>"

            p_len = x[8]
            assert isinstance(p_len, torch.Tensor)
            assert p_len.dim() == 1
            for pr, l in zip(programs.numpy().tolist(), p_len.numpy().tolist()):
                assert vocab["program_idx_to_token"][pr[l - 1]] == "<END>"

            # Check execution works:
            assert torch.all(symbolic_executor(scene_graphs, programs) == answers)


@pytest.mark.globaldataset
def test_against_ori_loader():
    from tqdm import tqdm
    from tests.test_data.ori_shapes_loader import OriSHAPESDataModule

    from vqa_framework.global_settings import comp_env_vars

    ori_dm = OriSHAPESDataModule(
        data_dir=comp_env_vars.DATA_DIR,
        tmp_dir=comp_env_vars.TMP_DIR,
        dm_shuffle_train_data=False,
        test_disable_download_process=True,
    )
    ori_dm.prepare_data()
    ori_dm.setup()
    from vqa_framework.data_modules.shapes_loader import SHAPESDataModule

    new_dm = SHAPESDataModule(
        data_dir=comp_env_vars.DATA_DIR,
        tmp_dir=comp_env_vars.TMP_DIR,
        dm_shuffle_train_data=False,
    )
    new_dm.prepare_data()
    new_dm.setup()

    ori_train = ori_dm.train_dataloader()
    new_train = new_dm.train_dataloader()
    assert len(ori_train) == len(new_train)

    for repeat in range(2):
        for ori_batch, new_batch in tqdm(
            zip(ori_train, new_train), total=len(ori_train)
        ):
            # Check non-tensors equal
            for i in [3]:
                assert ori_batch[i] == new_batch[i]

            # Check tensors equal
            for i in [0, 1, 2, 4, 5, 6, 7, 8]:
                assert torch.all(ori_batch[i] == new_batch[i])

        # Again for paranoia's sake; now with new loader
        print("Again, with new dataloader")
        ori_train = ori_dm.train_dataloader()
        new_train = new_dm.train_dataloader()
        assert len(ori_train) == len(new_train)
        for repeat2 in range(2):
            for ori_batch, new_batch in tqdm(
                zip(ori_train, new_train), total=len(ori_train)
            ):
                # Check non-tensors equal
                for i in [3]:
                    assert ori_batch[i] == new_batch[i]

                # Check tensors equal
                for i in [0, 1, 2, 4, 5, 6, 7, 8]:
                    assert torch.all(ori_batch[i] == new_batch[i])
