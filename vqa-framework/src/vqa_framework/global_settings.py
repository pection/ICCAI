from enum import Enum
import os
import logging
from collections import namedtuple


class ComputeEnv(Enum):
    GAME_MLFLOW_LOCAL = 1  # MLFlow with local storage
    GAME_MLFLOW_REMOTE_CG = 2  # MLFlow with CompGen S3 bucket
    UNSPECIFIED = (3,)  # Set all to None,
    VECTOR_VAUGHAN = 4
    TORONTO_CS_CLUSTER = 5


# High-level Settings:
# 1) Compute Env
# GAME_MLFLOW_LOCAL
# GAME_MLFLOW_REMOTE_CG
COMPUTE_ENV: ComputeEnv = ComputeEnv[
    os.getenv("THUNDER_COMPUTE_ENV", "GAME_MLFLOW_REMOTE_CG")
]

IAN_MLFLOW_URI = ""

ConfigVals = namedtuple(
    "Config",
    ["DATA_DIR", "TMP_DIR", "LIGHTNING_CHKPT_DIR", "MLFLOW_URI", "COMPUTE_ENV"],
)
comp_env_vars: ConfigVals

if COMPUTE_ENV == ComputeEnv.GAME_MLFLOW_LOCAL:
    comp_env_vars = ConfigVals(
        DATA_DIR="/media/administrator/extdrive/vqa-frame",
        TMP_DIR="/media/administrator/3227344f-bcd2-48f5-946e-78d2e7db3d26/vqa-frame",
        LIGHTNING_CHKPT_DIR="/media/administrator/extdrive/vqa-frame_light_chckpts",
        MLFLOW_URI=None,
        COMPUTE_ENV=COMPUTE_ENV,
    )
elif COMPUTE_ENV == ComputeEnv.GAME_MLFLOW_REMOTE_CG:
    comp_env_vars = ConfigVals(
        DATA_DIR="/media/administrator/extdrive/vqa-frame",
        TMP_DIR="/media/administrator/3227344f-bcd2-48f5-946e-78d2e7db3d26/vqa-frame",
        LIGHTNING_CHKPT_DIR="/media/administrator/extdrive/vqa-frame_light_chckpts",
        MLFLOW_URI=IAN_MLFLOW_URI,
        COMPUTE_ENV=COMPUTE_ENV,
    )
    os.environ["AWS_PROFILE"] = "mlflow"
elif COMPUTE_ENV == ComputeEnv.UNSPECIFIED:
    comp_env_vars = ConfigVals(
        DATA_DIR=None,
        TMP_DIR=None,
        LIGHTNING_CHKPT_DIR=None,
        MLFLOW_URI=None,
        COMPUTE_ENV=COMPUTE_ENV,
    )
    logging.warning(
        "Compute env not specified; fallback to UNSPECIFIED -- likely to crash unless testing."
    )
elif COMPUTE_ENV == ComputeEnv.VECTOR_VAUGHAN:
    if "SLURM_JOB_ID" in os.environ:
        # 25GB/job, and is purged. Need to copy local files over to prevent loss
        ckpt_dir = f'/checkpoint/ianberlot/{os.environ["SLURM_JOB_ID"]}'
    else:
        logging.warning(
            "ENVIRONMENT: No SLURM job # found; using working dir to store checkpoints"
        )
        ckpt_dir = ".."

    comp_env_vars = ConfigVals(
        DATA_DIR="/scratch/hdd001/home/ianberlot/vqa-frame",
        TMP_DIR="/scratch/hdd001/home/ianberlot/tmp",
        LIGHTNING_CHKPT_DIR=ckpt_dir,
        MLFLOW_URI=IAN_MLFLOW_URI,
        COMPUTE_ENV=COMPUTE_ENV,
    )
    # Assume that this is the AWS_PROFILE with access to the MLFlow S3 bucket
    os.environ["AWS_PROFILE"] = "mlflow"

    # HDF5 file shouldn't be being altered.
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
elif COMPUTE_ENV == ComputeEnv.TORONTO_CS_CLUSTER:
    ckpt_dir = f"/u/ianberlot/checkpoints/"

    comp_env_vars = ConfigVals(
        DATA_DIR="/u/ianberlot/datasets/",
        TMP_DIR="/u/ianberlot/tmp/",
        LIGHTNING_CHKPT_DIR=ckpt_dir,
        MLFLOW_URI=IAN_MLFLOW_URI,
        COMPUTE_ENV=COMPUTE_ENV,
    )
    # Assume that this is the AWS_PROFILE with access to the MLFlow S3 bucket
    os.environ["AWS_PROFILE"] = "mlflow"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
else:
    raise NotImplementedError
