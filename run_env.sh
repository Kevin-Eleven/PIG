#!/bin/bash
# Shared environment setup for interactive (salloc) runs.
# SOURCE this, don't execute it:   source run_env.sh

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate pig_env

cd /home/harisa_iitp/PIG || return 1

# main.py imports `from PIG.config import ...`, so the repo's PARENT must be on
# the path for `PIG` to resolve as a package.
export PYTHONPATH="$(dirname "$PWD")"

# Training writes checkpoints to ./log, which must land on scratch (/home is
# only 50GB and checkpoints are ~hundreds of MB each).
mkdir -p /scratch/harisa_iitp/PIG_outputs/log
ln -sfn /scratch/harisa_iitp/PIG_outputs/log ./log

echo "env ready: python=$(which python3)"
echo "PYTHONPATH=$PYTHONPATH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: no GPU visible -- are you inside an salloc/srun session?"
