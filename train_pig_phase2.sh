#!/bin/bash
#SBATCH --job-name=PIG_train_y
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/train_y_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/train_y_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khalieqahmad@gmail.com

set -euo pipefail

module purge
module load mldl/Miniconda
conda activate pig_env

cd /home/harisa_iitp/PIG

mkdir -p /scratch/harisa_iitp/PIG_outputs/log
ln -sfn /scratch/harisa_iitp/PIG_outputs/log ./log

# Phase 2: train the y-model (target, conditioned on Phase 1's x-model output).
# Auto-locate Phase 1's best checkpoint -- desc for Phase 1 is
# "x_mask_y_img_x_pretrained" (x=mask, y=img, train_model_type=x).
BEST_CKPT=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /scratch/harisa_iitp/PIG_outputs/log \
    --desc_suffix x_mask_y_img_x_pretrained)
echo "Using Phase 1 checkpoint: $BEST_CKPT"

python3 preprocess/patch_config.py --train_model_type y --x_model_path "$BEST_CKPT"

python main.py

echo "Phase 2 (y-model) training complete."
