#!/bin/bash
#SBATCH --job-name=PIG_train_x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/train_x_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/train_x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khalieqahmad@gmail.com

set -eo pipefail

module purge
module load MLDL/miniconda3
conda activate pig_env

cd /home/harisa_iitp/PIG

mkdir -p /scratch/harisa_iitp/PIG_outputs/log
ln -sfn /scratch/harisa_iitp/PIG_outputs/log ./log

# Phase 1: train the x-model (guider, conditioned on mask) -- no dependency on
# any prior checkpoint, so only train_model_type needs patching.
python3 preprocess/patch_config.py --train_model_type x

python main.py

echo "Phase 1 (x-model) training complete."
