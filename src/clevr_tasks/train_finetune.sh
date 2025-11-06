#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE #Disable locking b/c read only

# Number of questions to train the model on. 
# We experimented with 25000, 200000, and 560000
MAX_Q="$1"
echo $MAX_Q

# The Held Out Pair index must be between 0 and 28 inclusive.
HOP_IDX="$2"

# Run number; change to re-run the experiment
RUN="$3"
echo "RUN"
echo $RUN

# Set the max sequence length to be long enough to encode the full question.
echo "Max Sequence Length:"
MSL=49
echo $MSL

source alt_env/bin/activate
export PYTHONPATH=${PYTHONPATH}:src
python3 src/clevr_tasks/clevr.py --max_seq_length $MSL --preemptable  --dataset clevr_ho --max_questions $MAX_Q --train_steps 218750  --epochs 999999 --ho_idx ${HOP_IDX} --llayers 9 --xlayers 5 --rlayers 5 --loadLXMERTQA snap/pretrained/model --batchSize 32 --optim bert --lr 5e-5 --tqdm --output snap/clevr/run${RUN}_clevr_ntnv_ho${HOP_IDX}_msl${MSL}_scale${MAX_Q}_5e-5 --clevr_config "src/clevr_tasks/symlink_config.yaml"
