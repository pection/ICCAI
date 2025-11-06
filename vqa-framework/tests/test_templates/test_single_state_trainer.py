import os.path
import logging

import mlflow.pytorch
import torch
from pytest import approx
from pytorch_lightning import Trainer
from vqa_framework.global_settings import ComputeEnv
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
import pytest

from tests.test_data_module.counting_loader import (
    CrashOnDemandModel,
    CountingDataModule,
)
from argparse import Namespace
from vqa_framework.templates.single_state_trainer import OneStageTrainer
import math


def counting_init(args: Namespace):
    model = CrashOnDemandModel(
        logging=True,
        sync_dist=False,
        weight_init=0.5,
    )
    logging.critical(f"Initial model weights: {model.linear.weight}")
    return (
        CountingDataModule(**vars(args)),
        model,
        math.ceil(500 / args.batch_size),
    )


def check_model_is_id(model: pl.LightningModule):
    """
    for i in range(10):
        input_val = torch.Tensor([i], device=model.device)
        output_val = model(input_val)
        assert output_val.item() == approx(i, abs=1e-3)
    """
    # Eh, works for the simple dummy model we're traiining
    assert model.linear.weight.data.item() == approx(
        1, abs=0.2
    )  # Don't ask. Batch size probably stupid.


def check_train_resume(
    ckpt_path: str,
    run_args: Namespace,
    batch_size: int,
    trainer_max_epochs: int,
    trainer_gpus: int,
):
    # Make test deterministic(ish; shuffling borked)
    pl.utilities.seed.seed_everything(seed=42, workers=True)

    pl_trainer = Trainer(
        logger=False, enable_checkpointing=False, max_epochs=trainer_max_epochs + 1
    )
    dm, model, _ = counting_init(run_args)
    pl_trainer.fit(model, dm, ckpt_path=ckpt_path)

    # Check model has indeed trained for 4 epochs in total
    if trainer_gpus <= 1:
        assert pl_trainer.global_step == (trainer_max_epochs + 1) * (5000 / batch_size)
    elif trainer_gpus == 2:
        assert (
            pl_trainer.global_step
            >= ((trainer_max_epochs + 1) * (5000 / batch_size)) // 2
        )

    # Check we more or less have identity function
    check_model_is_id(model)


@pytest.mark.slow
@pytest.mark.parametrize("shuffle_train", [True, False])
@pytest.mark.parametrize(
    "trainer_gpus, disable_checkpoints",
    [
        pytest.param(
            2,
            False,
            marks=pytest.mark.skip(
                reason="Bug: 2GPU doesn't save ckpts"
            ),  # marks=pytest.mark.xfail(reason="Bug: 2GPU doesn't save ckpts")
        ),
        (2, True),  # This test does not seem to be stable; has failed at least once.
        (1, True),
        (1, False),
        (0, True),
        (0, False),
    ],
)
@pytest.mark.parametrize("restart_from_interrupt", [False, True])
def test_simple_train(
    tmp_path_factory,
    restart_from_interrupt,
    trainer_gpus,
    disable_checkpoints,
    shuffle_train,
):
    """
    Make sure we can instantiate and train simple model under all configs.
    Skip any tests that cannot be run on this machine (e.g, if there aren't multiple GPUs).
    """
    logging.basicConfig(level=logging.INFO)
    # disable_checkpoints = True

    # Make test deterministic(ish, screws up with data shuffling)
    pl.utilities.seed.seed_everything(seed=42, workers=True)

    if trainer_gpus >= 2:
        logging.critical(
            f"DANGER!!! Doesn't save checkpoints!!! Not currently testing the logged values!!! Probably borked!!!"
        )
    logging.warning(
        f"""Test: disable_checkpoints: {disable_checkpoints}
restart_from_interrupt:{restart_from_interrupt}
trainer_gpus: {trainer_gpus}
shuffle_train: {shuffle_train}"""
    )
    # Skip test if insufficient GPUs on this machine trainer_gpus = 1
    if torch.cuda.device_count() < trainer_gpus:
        pytest.skip(
            f"Insufficient GPUs, Required: {trainer_gpus} Available: {torch.cuda.device_count()}"
        )

    crash_index = "-1"
    os.environ["CRASH_VAL"] = crash_index

    mlflow_path = tmp_path_factory.mktemp("mlflow")
    curr_env = Namespace(
        DATA_DIR=tmp_path_factory.mktemp("target"),
        TMP_DIR=tmp_path_factory.mktemp("tmp"),
        LIGHTNING_CHKPT_DIR=tmp_path_factory.mktemp("light"),
        MLFLOW_URI=mlflow_path.as_uri(),
        COMPUTE_ENV=ComputeEnv.GAME_MLFLOW_REMOTE_CG,
    )

    profiler = None  # Screw it, not used often enough.
    callbacks = None  # Screw it; probably works (just gets passed into lightning)
    sub_epoch_checkpoint_freq = (
        -1
    )  # Mega screw it; relates to fault-tolerance, which doesn't work
    batch_size = 10
    trainer_max_epochs = 3
    num_dataloader_workers = (
        4  # Screw it; 4 is most representative (realistically never gonna do 0 or 1)
    )

    trainer = OneStageTrainer(
        env_args=curr_env,
        description="test descr",
        exp="TE",
        global_run_name="_test_name",
        task_name="task_test_simple",
        checkpoint_monitor="val_loss_epoch",
        disable_commit_check=True,
        disable_checkpoints=disable_checkpoints,
        profiler=profiler,
        callbacks=callbacks,
        restart_from_interrupt=restart_from_interrupt,
        sub_epoch_checkpoint_freq=sub_epoch_checkpoint_freq,
    )

    run_args = Namespace(
        shuffle_train=shuffle_train,
        batch_size=batch_size,
        trainer_gpus=trainer_gpus,
        trainer_max_epochs=trainer_max_epochs,
        dm_workers=num_dataloader_workers,
    )
    logging.info("Training model")
    model = trainer.run(args=run_args, modules_init=counting_init)

    # Check fitted model is identity function (give or take)
    check_model_is_id(model)

    del trainer

    lightning_save_dir = os.path.join(
        curr_env.LIGHTNING_CHKPT_DIR, "TE_test_name", "version_0"
    )
    lightning_ckpt_dir = os.path.join(lightning_save_dir, "checkpoints")

    # ==============================================================
    # Check tensorboard files exist
    logging.info("Check tensorboard files exist")
    contents = os.listdir(lightning_save_dir)
    logging.info("Lightning dir contents:")
    logging.info(contents)

    filtered = [x for x in contents if x.startswith("events.out.tfevents")]
    assert len(filtered) == 1  # Should only be one checkpoint (unless interrupted)

    # ==============================================================
    # Check checkpoints created & marked as safe
    if disable_checkpoints:
        assert not os.path.exists(os.path.join(lightning_ckpt_dir))
    else:
        logging.info("Check checkpoints created & marked as safe")
        assert os.path.exists(
            os.path.join(lightning_ckpt_dir, "safe_last.ckpt")
        )  # last epoch checkpoint exists
        contents = os.listdir(lightning_ckpt_dir)
        light_ckpts = [
            os.path.join(lightning_ckpt_dir, x)
            for x in contents
            if x.endswith(".ckpt")
            and (x.startswith("best_val") or x == "safe_last.ckpt")
        ]
        assert len(light_ckpts) >= 3
        logging.info("Lightning ckpt dir contents:")
        logging.info(contents)
        matches = 0
        for file in contents:
            if file.startswith("best_val_acc_Epoch"):
                matches += 1
                # Assert we've marked this checkpoint as safe
                assert f"safe_{file}.txt" in contents
        # assert we've only created 2 top-k checkpoints
        assert matches == 2

        # ==============================================================
        # Check model performance is plausible, and can load lightning checkpoints
        # saved locally
        # -- load both to resume training, and just the weights
        for ckpt_path in light_ckpts:
            logging.info(f"Testing checkpoint {ckpt_path} is loadable/OK")
            model = CrashOnDemandModel.load_from_checkpoint(ckpt_path)
            # Check we more or less have identity function
            check_model_is_id(model)
            # Check we can resume training
            check_train_resume(
                ckpt_path,
                run_args=run_args,
                batch_size=batch_size,
                trainer_max_epochs=trainer_max_epochs,
                trainer_gpus=trainer_gpus,
            )

    # ==============================================================
    # Check can load MLFlow (official model & lightning checkpoints), and repeat the above
    run_id = os.listdir(os.path.join(mlflow_path, "0"))

    run_id = [tmp_id for tmp_id in run_id if tmp_id != "meta.yaml"]

    assert len(run_id) == 1
    run_id = run_id[0]
    mlflow.set_tracking_uri(curr_env.MLFLOW_URI)
    model_uri = f"runs:/{run_id}/model"
    client = MlflowClient()

    if disable_checkpoints:
        # Check MLFlow did not log model
        with pytest.raises(FileNotFoundError):
            mlflow.pytorch.load_model(
                model_uri=model_uri, dst_path=tmp_path_factory.mktemp("tmp2")
            )
        assert (
            len(client.list_artifacts(run_id=run_id, path="lightning_checkpoints")) == 0
        )
    else:
        # Check we can load the model saved via MLFlow

        model = mlflow.pytorch.load_model(
            model_uri=model_uri, dst_path=tmp_path_factory.mktemp("tmp2")
        )

        check_model_is_id(model)

        # Check we can load saved checkpoints
        artifacts = client.list_artifacts(run_id=run_id, path="lightning_checkpoints")
        artifact_paths = [a.path for a in artifacts]
        artifact_names = [a[22:] for a in artifact_paths]

        assert "safe_last.ckpt" in artifact_names
        matches = 0
        for file in artifact_names:
            if file.startswith("best_val_acc_Epoch"):
                matches += 1
                # Assert we've marked this checkpoint as safe
                assert f"safe_{file}.txt" in artifact_names
        # assert we've only created 2 top-k checkpoints
        assert matches == 2

        checkpoints = [a for a in artifact_paths if a.endswith(".ckpt")]
        assert len(checkpoints) >= 3
        for artifact_path in checkpoints:
            local_path = client.download_artifacts(
                run_id=run_id,
                path=artifact_path,
                dst_path=str(tmp_path_factory.mktemp("tmp3")),
            )

            logging.info(
                f"Testing checkpoint {artifact_path} from MLFlow is loadable/OK"
            )
            model = CrashOnDemandModel.load_from_checkpoint(local_path)
            # Check we more or less have identity function
            check_model_is_id(model)
            # Check we can resume training
            check_train_resume(
                local_path,
                run_args=run_args,
                batch_size=batch_size,
                trainer_max_epochs=trainer_max_epochs,
                trainer_gpus=trainer_gpus,
            )

    # ==============================================================
    # Check mlflow metric data logged, all the way to end (ish)

    if trainer_gpus <= 1:
        for metric_name in ["train_loss_step", "val_loss_step"]:
            logging.info(f"Checking stored MLFlow values for {metric_name}")
            dset_len = 5000 if "train" in metric_name else 500
            # By default, lightning saves every 50th step for train. ?Val is every step?
            # 1 step = 1 batch
            ckpt_freq = 50 if "train" in metric_name else 1

            metric_vals = client.get_metric_history(run_id=run_id, key=metric_name)
            metric_timesteps = [val.step for val in metric_vals]
            metric_timesteps.sort()
            print(metric_timesteps)

            assert (
                len(metric_vals)
                == trainer_max_epochs * (dset_len / batch_size) / ckpt_freq
            )

            # Make sure we have logging values until the very end (e.g., no weird restart from zero or data loss)
            if "train" in metric_name:
                expected_max_step = (
                    math.floor(trainer_max_epochs * (dset_len / batch_size) / ckpt_freq)
                    * ckpt_freq
                )
                expected_min_step = 50
            else:
                # Validation steps start from 0, train don't. Go figure.
                expected_max_step = trainer_max_epochs * (dset_len / batch_size) - 1
                expected_min_step = 0

            assert max(metric_timesteps) == expected_max_step
            assert min(metric_timesteps) == expected_min_step

        for metric_name in ["train_loss_epoch", "val_loss_epoch"]:
            logging.info(f"Checking stored MLFlow values for {metric_name}")
            metric_vals = client.get_metric_history(run_id=run_id, key=metric_name)
            assert len(metric_vals) == trainer_max_epochs
            metric_vals = [val.step for val in metric_vals]

            # Note: epoch values for both train & val logged with global step
            train_dset_len = 5000
            assert min(metric_vals) == 1 * train_dset_len / batch_size
            assert max(metric_vals) == trainer_max_epochs * train_dset_len / batch_size

    # ==============================================================
    # Check MLFlow tags are there
    mlflow_run = mlflow.get_run(run_id=run_id)

    tags = {
        k: v for k, v in mlflow_run.data.tags.items() if not k.startswith("mlflow.")
    }
    assert tags["compute_env"] == curr_env.COMPUTE_ENV.name
    assert tags["exp"] == "TE"
    assert tags["tb_version"] == "0"
    val_size = 500
    assert tags["val_steps_per_epoch"] == str(math.ceil(val_size / batch_size))

    # ==============================================================
    # Check parameters recorded in MLFlow
    params = mlflow_run.data.params
    assert params["shuffle_train"] == str(run_args.shuffle_train)
    assert params["batch_size"] == str(batch_size)
    assert params["dummy_arg"] == str(None)

    # Check ignored argument in CrashOnDemandModel's __init__ is not logged
    assert "logging" not in params

    # ==============================================================
    # Check MLFlow experiment name
    assert mlflow_run.info.experiment_id == "0"
    # Check run is done
    assert mlflow_run.info.status == "FINISHED"


@pytest.mark.parametrize("shuffle_train", [True, False])
@pytest.mark.parametrize(
    "trainer_gpus",
    [
        pytest.param(2, marks=pytest.mark.skip(reason="Bug: 2GPU doesn't save ckpts")),
        1,
        0,
    ],
)
def test_interrupt_resume_train(
    tmp_path_factory,
    trainer_gpus,
    shuffle_train,
):
    """
    Make sure we can instantiate and train simple model under all configs.
    Skip any tests that cannot be run on this machine (e.g, if there aren't multiple GPUs).
    """
    logging.basicConfig(level=logging.INFO)

    # Make test deterministic-ish (doesn't quite work with data shuffling)
    pl.utilities.seed.seed_everything(seed=42, workers=True)

    restart_from_interrupt = True
    disable_checkpoints = False

    if trainer_gpus >= 2:
        logging.critical(f"DANGER!!! Doesn't save checkpoints!!! Borked!!!")
    logging.warning(
        f"""Test: restart_from_interrupt:{restart_from_interrupt}
trainer_gpus: {trainer_gpus}"""
    )

    # Skip test if insufficient GPUs on this machine trainer_gpus = 1
    if torch.cuda.device_count() < trainer_gpus:
        pytest.skip(
            f"Insufficient GPUs, Required: {trainer_gpus} Available: {torch.cuda.device_count()}"
        )

    profiler = None  # Screw it, not used often enough.
    callbacks = None  # Screw it; probably works (just gets passed into lightning)
    sub_epoch_checkpoint_freq = (
        -1
    )  # Mega screw it; relates to fault-tolerance, which doesn't work
    trainer_max_epochs = 3
    num_dataloader_workers = (
        4  # Screw it; 4 is most representative (realistically never gonna do 0 or 1)
    )
    batch_size = 10  # Train set is size 5000, therefore 500 batches

    # Complete first epoch (500 batches), save, and crash half-way through second epoch
    crash_val = 800
    os.environ["CRASH_VAL"] = f"{crash_val}"

    mlflow_path = tmp_path_factory.mktemp("mlflow")
    curr_env = Namespace(
        DATA_DIR=tmp_path_factory.mktemp("target"),
        TMP_DIR=tmp_path_factory.mktemp("tmp"),
        LIGHTNING_CHKPT_DIR=tmp_path_factory.mktemp("light"),
        MLFLOW_URI=mlflow_path.as_uri(),
        COMPUTE_ENV=ComputeEnv.GAME_MLFLOW_REMOTE_CG,
    )

    trainer = OneStageTrainer(
        env_args=curr_env,
        description="test descr",
        exp="TE",
        global_run_name="_test_name",
        task_name="task_test_simple",
        checkpoint_monitor="val_loss_epoch",
        disable_commit_check=True,
        disable_checkpoints=disable_checkpoints,
        profiler=profiler,
        callbacks=callbacks,
        restart_from_interrupt=restart_from_interrupt,
        sub_epoch_checkpoint_freq=sub_epoch_checkpoint_freq,
    )

    run_args = Namespace(
        shuffle_train=shuffle_train,
        batch_size=batch_size,
        trainer_gpus=trainer_gpus,
        trainer_max_epochs=trainer_max_epochs,
        dm_workers=num_dataloader_workers,
    )
    logging.info("Training model")

    # Model should fail during 2nd epoch
    with pytest.raises(ZeroDivisionError):
        trainer.run(args=run_args, modules_init=counting_init)

    # Attempt to resume training
    trainer = OneStageTrainer(
        env_args=curr_env,
        description="test descr",
        exp="TE",
        global_run_name="_test_name",
        task_name="task_test_simple",
        checkpoint_monitor="val_loss_epoch",
        disable_commit_check=True,
        disable_checkpoints=disable_checkpoints,
        profiler=profiler,
        callbacks=callbacks,
        restart_from_interrupt=restart_from_interrupt,
        sub_epoch_checkpoint_freq=sub_epoch_checkpoint_freq,
    )
    # Should automatically resume based off of saved interim files.
    os.environ["CRASH_VAL"] = "-1"
    model = trainer.run(args=run_args, modules_init=counting_init)

    # Check fitted model is identity function (give or take)
    check_model_is_id(model)

    del trainer

    lightning_save_dir = os.path.join(
        curr_env.LIGHTNING_CHKPT_DIR, "TE_test_name", "version_0"
    )
    lightning_ckpt_dir = os.path.join(lightning_save_dir, "checkpoints")

    # ==============================================================
    # Check tensorboard files exist
    logging.info("Check tensorboard files exist")
    contents = os.listdir(lightning_save_dir)
    logging.info("Lightning dir contents:")
    logging.info(contents)

    filtered = [x for x in contents if x.startswith("events.out.tfevents")]
    assert len(filtered) == 2  # Should only be one checkpoint per interruption

    # ==============================================================
    # Check mlflow metric data logged, all the way to end (ish)
    run_id = os.listdir(os.path.join(mlflow_path, "0"))

    run_id = [tmp_id for tmp_id in run_id if tmp_id != "meta.yaml"]

    assert len(run_id) == 1
    run_id = run_id[0]
    mlflow.set_tracking_uri(curr_env.MLFLOW_URI)
    model_uri = f"runs:/{run_id}/model"
    client = MlflowClient()

    if trainer_gpus <= 1:
        for metric_name in ["train_loss_step", "val_loss_step"]:
            logging.info(f"Checking stored MLFlow values for {metric_name}")
            dset_len = 5000 if "train" in metric_name else 500
            # By default, lightning saves every 50th step for train. ?Val is every step?
            # 1 step = 1 batch
            ckpt_freq = 50 if "train" in metric_name else 1

            metric_vals = client.get_metric_history(run_id=run_id, key=metric_name)
            metric_timesteps = [val.step for val in metric_vals]
            metric_timesteps.sort()
            print(metric_timesteps)

            # Gain some extra logged values due to restart
            assert (
                len(metric_vals)
                == trainer_max_epochs * (dset_len / batch_size) / ckpt_freq
                + (crash_val % (dset_len / batch_size))
                // ckpt_freq  # Note: crash_val is already in batches
            )
            # Check we have the expected number of values, once duplicate timesteps are removed
            num_unique_times = len(set(metric_timesteps))
            assert (
                num_unique_times
                == trainer_max_epochs * (dset_len / batch_size) / ckpt_freq
            )

            # Make sure we have logging values until the very end (e.g., no weird restart from zero or data loss)
            if "train" in metric_name:
                expected_max_step = (
                    math.floor(trainer_max_epochs * (dset_len / batch_size) / ckpt_freq)
                    * ckpt_freq
                )
                expected_min_step = 50
            else:
                # Validation steps start from 0, train don't. Go figure.
                expected_max_step = trainer_max_epochs * (dset_len / batch_size) - 1
                expected_min_step = 0

            assert max(metric_timesteps) == expected_max_step
            assert min(metric_timesteps) == expected_min_step

        for metric_name in ["train_loss_epoch", "val_loss_epoch"]:
            logging.info(f"Checking stored MLFlow values for {metric_name}")
            metric_vals = client.get_metric_history(run_id=run_id, key=metric_name)
            assert len(metric_vals) == trainer_max_epochs
            metric_vals = [val.step for val in metric_vals]

            # Note: epoch values for both train & val logged with global step
            train_dset_len = 5000
            assert min(metric_vals) == 1 * train_dset_len / batch_size
            assert max(metric_vals) == trainer_max_epochs * train_dset_len / batch_size

    # ==============================================================
    # Check MLFlow tags are there
    mlflow_run = mlflow.get_run(run_id=run_id)

    tags = {
        k: v for k, v in mlflow_run.data.tags.items() if not k.startswith("mlflow.")
    }
    assert tags["compute_env"] == curr_env.COMPUTE_ENV.name
    assert tags["exp"] == "TE"
    assert tags["tb_version"] == "0"
    val_size = 500
    assert tags["val_steps_per_epoch"] == str(math.ceil(val_size / batch_size))

    # ==============================================================
    # Check parameters recorded in MLFlow
    params = mlflow_run.data.params
    assert params["shuffle_train"] == str(run_args.shuffle_train)
    assert params["batch_size"] == str(batch_size)
    assert params["dummy_arg"] == str(None)

    # Check ignored argument in CrashOnDemandModel's __init__ is not logged
    assert "logging" not in params

    # ==============================================================
    # Check MLFlow experiment name
    assert mlflow_run.info.experiment_id == "0"
    # Check run is done
    assert mlflow_run.info.status == "FINISHED"

    # ==============================================================
    # Check checkpoints created & marked as safe
    if disable_checkpoints:
        assert not os.path.exists(os.path.join(lightning_ckpt_dir))
    else:
        logging.info("Check checkpoints created & marked as safe")
        assert os.path.exists(
            os.path.join(lightning_ckpt_dir, "safe_last.ckpt")
        )  # last epoch checkpoint exists
        contents = os.listdir(lightning_ckpt_dir)
        light_ckpts = [
            os.path.join(lightning_ckpt_dir, x)
            for x in contents
            if x.endswith(".ckpt")
            and (x.startswith("best_val") or x == "safe_last.ckpt")
        ]
        assert len(light_ckpts) >= 3
        logging.info("Lightning ckpt dir contents:")
        logging.info(contents)
        matches = 0
        for file in contents:
            if file.startswith("best_val_acc_Epoch"):
                matches += 1
                # Assert we've marked this checkpoint as safe
                assert f"safe_{file}.txt" in contents
        # assert we've only created 2 top-k checkpoints
        assert matches == 2

        # ==============================================================
        # Check model performance is plausible, and can load lightning checkpoints
        # saved locally
        # -- load both to resume training, and just the weights
        for ckpt_path in light_ckpts:
            logging.info(f"Testing checkpoint {ckpt_path} is loadable/OK")
            model = CrashOnDemandModel.load_from_checkpoint(ckpt_path)
            # Check we more or less have identity function
            check_model_is_id(model)
            # Check we can resume training
            check_train_resume(
                ckpt_path,
                run_args=run_args,
                batch_size=batch_size,
                trainer_max_epochs=trainer_max_epochs,
                trainer_gpus=trainer_gpus,
            )

    # ==============================================================
    # Check can load MLFlow (official model & lightning checkpoints), and repeat the above

    if disable_checkpoints:
        # Check MLFlow did not log model
        with pytest.raises(FileNotFoundError):
            mlflow.pytorch.load_model(
                model_uri=model_uri, dst_path=tmp_path_factory.mktemp("tmp2")
            )
        assert (
            len(client.list_artifacts(run_id=run_id, path="lightning_checkpoints")) == 0
        )
    else:
        # Check we can load the model saved via MLFlow

        model = mlflow.pytorch.load_model(
            model_uri=model_uri, dst_path=tmp_path_factory.mktemp("tmp2")
        )

        check_model_is_id(model)

        # Check we can load saved checkpoints
        artifacts = client.list_artifacts(run_id=run_id, path="lightning_checkpoints")
        artifact_paths = [a.path for a in artifacts]
        artifact_names = [a[22:] for a in artifact_paths]

        assert "safe_last.ckpt" in artifact_names
        matches = 0
        for file in artifact_names:
            if file.startswith("best_val_acc_Epoch"):
                matches += 1
                # Assert we've marked this checkpoint as safe
                assert f"safe_{file}.txt" in artifact_names
        # assert we've only created 2 top-k checkpoints
        assert matches == 2

        checkpoints = [a for a in artifact_paths if a.endswith(".ckpt")]
        assert len(checkpoints) >= 3
        for artifact_path in checkpoints:
            local_path = client.download_artifacts(
                run_id=run_id,
                path=artifact_path,
                dst_path=str(tmp_path_factory.mktemp("tmp3")),
            )

            logging.info(
                f"Testing checkpoint {artifact_path} from MLFlow is loadable/OK"
            )
            model = CrashOnDemandModel.load_from_checkpoint(local_path)
            # Check we more or less have identity function
            check_model_is_id(model)
            # Check we can resume training
            check_train_resume(
                local_path,
                run_args=run_args,
                batch_size=batch_size,
                trainer_max_epochs=trainer_max_epochs,
                trainer_gpus=trainer_gpus,
            )
