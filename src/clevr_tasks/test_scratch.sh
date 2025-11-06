#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE #Disable locking b/c read only

# Run number; change to re-run the experiment
RUN="$3"

# Number of questions to train the model on. 
# We experimented with 25000, 200000, and 560000
MAX_Q="$1"
echo "Max Questions:"
echo $MAX_Q

echo "Max Sequence Length:"
MSL=49
echo $MSL

# The Held Out Pair index must be between 0 and 28 inclusive.
HOP_IDX="$2"
echo "HO"
echo $HOP_IDX

EXP_NAME="no_text_no_vis"
CKPT_DIR="snap/clevr/fromScratch/run${RUN}_clevr_ntnv_ho${HOP_IDX}_msl${MSL}_scale${MAX_Q}_steps481k_lr1e-5"

export PYTHONPATH=${PYTHONPATH}:src
HO=${HOP_IDX}
echo "=============================================="
echo ${EXP_NAME}
echo "ho=${HO}"
echo ${CKPT_DIR}
echo "----------------------------------------------"


source alt_env/bin/activate
export PYTHONPATH=${PYTHONPATH}:src

# Run eval on minimal splits
python3 src/clevr_tasks/clevr.py --max_seq_length ${MSL} --dataset "clevr_atom_ho_exist" --optim none --test "val test" --load "${CKPT_DIR}/LAST" --experiment_name ${EXP_NAME} --ho_idx ${HO} --llayers 9 --xlayers 5 --rlayers 5 --batchSize 32 --tqdm --output ${CKPT_DIR} --clevr_config "src/clevr_tasks/symlink_config.yaml"

# Run eval on complex splits
python3 src/clevr_tasks/clevr.py --max_seq_length ${MSL} --test "val test" --load "${CKPT_DIR}/LAST" --experiment_name ${EXP_NAME} --ho_idx ${HO} --llayers 9 --xlayers 5 --rlayers 5 --batchSize 32 --optim bert --tqdm --output ${CKPT_DIR} --clevr_config "src/clevr_tasks/symlink_config.yaml"
