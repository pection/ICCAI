import pytest
from vqa_framework.data_modules.syget_loader import SHAPESSyGeTDataModule
import os
import shutil
import torch
from vqa_framework.vr.preprocess import decode
from typing import Dict, Any, List
import logging
from vqa_framework.models.symbolic_exec import SymbolicShapesExecutor


@pytest.fixture
def clean_h5(tmpdir):
    """Clean processed data b/f and after test"""
    # Setup: fill with any logic you want
    if os.path.exists(os.path.join(".", "resources", "r4", "hdf5_syget")):
        shutil.rmtree(os.path.join(".", "resources", "r4", "hdf5_syget"))
    if os.path.exists(os.path.join(".", "resources", "r5", "hdf5_syget")):
        shutil.rmtree(os.path.join(".", "resources", "r5", "hdf5_syget"))
    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    if os.path.exists(os.path.join(".", "resources", "r4", "hdf5_syget")):
        shutil.rmtree(os.path.join(".", "resources", "r4", "hdf5_syget"))
    if os.path.exists(os.path.join(".", "resources", "r5", "hdf5_syget")):
        shutil.rmtree(os.path.join(".", "resources", "r5", "hdf5_syget"))


def test_dummy_syget_prepare_exists(tmp_path_factory, clean_h5, caplog):
    """
    Minimal test, check just that SHAPES Data module can processed
    _something_, and that it doesn't crash in the subsequent steps.

    Also checks resulting dataloaders contain the correct number of items, and
    that if you run the DataModule again, it won't re-process the data.

    This test assumes the repo has already been downloaded, and tests
    accordingly.

    Note: this test defaults to all data for each loader
    """
    caplog.set_level(logging.INFO)
    data_dir = "./resources/r5/"
    tmp_dir = tmp_path_factory.mktemp("tmp")

    for repeat in range(3):
        caplog.clear()
        dm = SHAPESSyGeTDataModule(data_dir=data_dir, tmp_dir=tmp_dir)

        assert "Repository already downloaded" not in caplog.text
        assert "Processed SyGeT dataset already exists" not in caplog.text

        dm.prepare_data()

        assert "Repository already downloaded" in caplog.text
        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "Processed SyGeT dataset already exists" in caplog.text
        )

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is again empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        assert len(test_loader) == 1
        assert len(test_loader[0].dataset) == 251  # 6976  # Val OOD

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert len(val_loader) == 1
        assert len(val_loader[0].dataset) == 351  # 1080  # Val IID
        train_loader = dm.train_dataloader()
        assert len(train_loader.dataset) == 351  # 7560  # Train


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


def test_dummy_syget_vals_all_fams(tmp_path_factory, clean_h5, caplog):
    """
    Quick test, check dataloaders have the correct data (at least, when
    shuffling is off)
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r5/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()

        dm = SHAPESSyGeTDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            dm_shuffle_train_data=False,
            dm_batch_size=2,
            val_batch_size=4,
        )

        assert "Repository already downloaded" not in caplog.text
        assert "Processed SyGeT dataset already exists" not in caplog.text

        dm.prepare_data()

        assert "Repository already downloaded" in caplog.text
        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "Processed SyGeT dataset already exists" in caplog.text
        )

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is still empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        assert len(test_loader) == 1
        test_loader = test_loader[0]
        # check_dataloader(test_loader, dm)

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert len(val_loader) == 1
        val_loader = val_loader[0]
        # check_dataloader(val_loader, dm)

        train_loader = dm.train_dataloader()
        # check_dataloader(train_loader, dm, repeats=4)

        symbolic_executor = SymbolicShapesExecutor(dm.vocab)
        for dl in [test_loader, val_loader, train_loader]:
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


def get_q_fam(program: str) -> int:
    QUESTION_FAMILY = {
        # SyGeT Families; numbers from https://github.com/ankitkv/SHAPES-SyGeT
        # Train families
        tuple(["_Find_colour", "_Find_colour", "_Transform", "_And", "_Answer"]): 1,
        tuple(["_Find_shape", "_Find_colour", "_Transform", "_And", "_Answer"]): 2,
        tuple(
            [
                "_Find_shape",
                "_Find_colour",
                "_Transform",
                "_Transform",
                "_And",
                "_Answer",
            ]
        ): 3,
        tuple(
            [
                "_Find_shape",
                "_Find_shape",
                "_Transform",
                "_Transform",
                "_And",
                "_Answer",
            ]
        ): 4,
        tuple(["_Find_colour", "_Find_shape", "_And", "_Answer"]): 5,
        tuple(["_Find_shape", "_Find_colour", "_And", "_Answer"]): 6,
        tuple(["_Find_shape", "_Find_shape", "_And", "_Answer"]): 7,
        # Val OOD families
        tuple(
            [
                "_Find_colour",
                "_Find_colour",
                "_Transform",
                "_Transform",
                "_And",
                "_Answer",
            ]
        ): 8,
        tuple(["_Find_colour", "_Find_shape", "_Transform", "_And", "_Answer"]): 9,
        tuple(
            [
                "_Find_colour",
                "_Find_shape",
                "_Transform",
                "_Transform",
                "_And",
                "_Answer",
            ]
        ): 10,
        tuple(["_Find_shape", "_Find_shape", "_Transform", "_And", "_Answer"]): 11,
        tuple(["_Find_colour", "_Find_colour", "_And", "_Answer"]): 12,
    }

    program = program.strip().split()
    program = program[1:-1]  # remove start/end tokens

    def convert(x):
        if "[" in x:
            if "_Transform" in x:
                return x[: x.find("[")]
            else:
                arg = x[x.find("[") + 1 : -1]
                x = x[: x.find("[")]

                if arg in ["red", "green", "blue"]:
                    return x + "_colour"
                else:
                    assert arg in ["triangle", "circle", "square"]
                    return x + "_shape"

        else:
            return x

    program = [convert(x) for x in program]
    return QUESTION_FAMILY[tuple(program[::-1])]


def test_dummy_syget_vals_subset_fams_subset_dataset(
    tmp_path_factory, clean_h5, caplog
):
    """
    Quick test, check dataloaders have the correct data (at least, when
    shuffling is off)
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r5/"
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()

        train_fam = [1, 2, 3, 4]
        val_fams = [[7], [6], [5], [4], [3], [2], [1], [1, 2, 3, 4]]
        test_fams = [[8], [9], [10], [11], [12], [8, 10, 12]]

        dm = SHAPESSyGeTDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            dm_shuffle_train_data=False,
            dm_batch_size=2,
            val_batch_size=4,
            train_fam=train_fam,
            val_fams=val_fams,
            test_fams=test_fams,
        )

        assert "Repository already downloaded" not in caplog.text
        assert "Processed SyGeT dataset already exists" not in caplog.text

        dm.prepare_data()

        assert "Repository already downloaded" in caplog.text
        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "Processed SyGeT dataset already exists" in caplog.text
        )

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is still empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        assert len(test_loader) == len(test_fams)
        # check_dataloader(test_loader, dm)

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert len(val_loader) == len(val_fams)
        # check_dataloader(val_loader, dm)

        train_loader = dm.train_dataloader()
        assert len(train_loader.dataset) < 7560  # only a subset

        # check_dataloader(train_loader, dm, repeats=4)

        assert sum([len(x.dataset) for x in val_loader[:-1]]) == 351  # 1080
        assert (
            sum([len(x.dataset) for x in val_loader[:3]]) + len(val_loader[-1].dataset)
            == 351
        )  # 1080
        assert sum([len(x.dataset) for x in test_loader[:-1]]) == 251  # 6976
        assert (
            len(test_loader[1].dataset)
            + len(test_loader[3].dataset)
            + len(test_loader[-1].dataset)
            == 251
        )  # 6976

        symbolic_executor = SymbolicShapesExecutor(dm.vocab)
        for dl, fams in (
            [(train_loader, train_fam)]
            + list(zip(val_loader, val_fams))
            + list(zip(test_loader, test_fams))
        ):
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
                    assert get_q_fam(pr_eng) in fams  # check program family is correct

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


@pytest.mark.slow
def test_dummy_syget_vals_subset_fams_full_dataset(tmp_path_factory, clean_h5, caplog):
    """
    Quick test, check dataloaders have the correct data (at least, when
    shuffling is off).

    Note; does not download dataset again
    """
    caplog.set_level(logging.INFO)

    data_dir = "./resources/r4/"  # Full dataset
    tmp_dir = tmp_path_factory.mktemp("tmp")  # './tests/resources/empty2' #

    for repeat in range(3):
        caplog.clear()

        train_fam = [1, 2, 3, 4]
        val_fams = [[7], [6], [5], [4], [3], [2], [1], [1, 2, 3, 4]]
        test_fams = [[8], [9], [10], [11], [12], [8, 10, 12]]

        dm = SHAPESSyGeTDataModule(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            dm_shuffle_train_data=False,
            dm_batch_size=2,
            val_batch_size=4,
            train_fam=train_fam,
            val_fams=val_fams,
            test_fams=test_fams,
        )

        assert "Repository already downloaded" not in caplog.text
        assert "Processed SyGeT dataset already exists" not in caplog.text

        dm.prepare_data()

        assert "Repository already downloaded" in caplog.text
        # check dataset only exists after the first time we create DataModule
        assert (repeat == 0) != (
            "Processed SyGeT dataset already exists" in caplog.text
        )

        # check processed dataset was created
        assert os.path.isdir(os.path.join(data_dir, dm.LOCAL_NAME))

        # check that tmp directory is still empty
        assert len(os.listdir(tmp_dir)) == 0

        # check this doesn't crash
        dm.setup("test")
        test_loader = dm.test_dataloader()
        assert len(test_loader) == len(test_fams)
        # check_dataloader(test_loader, dm)

        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert len(val_loader) == len(val_fams)
        # check_dataloader(val_loader, dm)

        train_loader = dm.train_dataloader()
        assert len(train_loader.dataset) < 7560  # only a subset

        # check_dataloader(train_loader, dm, repeats=4)

        assert sum([len(x.dataset) for x in val_loader[:-1]]) == 1080
        assert (
            sum([len(x.dataset) for x in val_loader[:3]]) + len(val_loader[-1].dataset)
            == 1080
        )
        assert sum([len(x.dataset) for x in test_loader[:-1]]) == 6976
        assert (
            len(test_loader[1].dataset)
            + len(test_loader[3].dataset)
            + len(test_loader[-1].dataset)
            == 6976
        )

        symbolic_executor = SymbolicShapesExecutor(dm.vocab)
        for dl, fams in (
            [(train_loader, train_fam)]
            + list(zip(val_loader, val_fams))
            + list(zip(test_loader, test_fams))
        ):
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
                    assert get_q_fam(pr_eng) in fams  # check program family is correct

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


@pytest.mark.slow
def test_syget_prepare_exists(tmp_path_factory):
    """
    Create dataloader from scratch and check it works.
    Does not test subsets of question families.
    Also check resulting dataloaders contain the correct number of items.
    """
    data_dir = tmp_path_factory.mktemp("target")
    tmp_dir = tmp_path_factory.mktemp("tmp")
    dm = SHAPESSyGeTDataModule(data_dir=data_dir, tmp_dir=tmp_dir)
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
    assert len(test_loader) == 1
    test_loader = test_loader[0]
    assert len(test_loader.dataset) == 6976

    dm.setup("fit")
    val_loader = dm.val_dataloader()
    assert len(val_loader) == 1
    val_loader = val_loader[0]
    assert len(val_loader.dataset) == 1080

    train_loader = dm.train_dataloader()
    assert len(train_loader.dataset) == 7560  # tiny + small + medium + large

    symbolic_executor = SymbolicShapesExecutor(dm.vocab)
    for dl in [test_loader, val_loader, train_loader]:
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
