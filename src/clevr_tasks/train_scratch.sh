#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE #Disable locking b/c read only

# Run number; change to re-run the experiment
RUN="$3"
echo "RUN"
echo $RUN

# Number of questions to train the model on. 
# We experimented with 25000, 200000, and 560000
MAX_Q="$1"
echo $MAX_Q

# The Held Out Pair index must be between 0 and 28 inclusive.
HOP_IDX="$2"

echo "Max Sequence Length:"
MSL=49
echo $MSL

TRAIN_STEPS=481000
echo "Train steps"
echo $TRAIN_STEPS

LEARNING_RATE="0.00001"
echo "Learning Rate"
echo $LEARNING_RATE

source alt_env/bin/activate
export PYTHONPATH=${PYTHONPATH}:src
python3 src/clevr_tasks/clevr.py --fromScratch --max_seq_length $MSL --preemptable  --dataset clevr_ho --max_questions $MAX_Q --train_steps $TRAIN_STEPS  --epochs 999999 --ho_idx ${HOP_IDX} --llayers 9 --xlayers 5 --rlayers 5 --batchSize 32 --optim bert --lr $LEARNING_RATE --tqdm --output snap/clevr/fromScratch/run${RUN}_clevr_ntnv_ho${HOP_IDX}_msl${MSL}_scale${MAX_Q}_steps481k_lr1e-5  --clevr_config "src/clevr_tasks/symlink_config.yaml"
