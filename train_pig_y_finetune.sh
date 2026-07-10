#!/bin/bash
#SBATCH --job-name=PIG_y_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/y_finetune_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/y_finetune_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khalieqahmad@gmail.com

set -eo pipefail

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate pig_env

cd /home/harisa_iitp/PIG

mkdir -p /scratch/harisa_iitp/PIG_outputs/log
ln -sfn /scratch/harisa_iitp/PIG_outputs/log ./log

# ---------------------------------------------------------------------------
# Stage C: fine-tune the VinDr-pretrained y-model on the DBT slices.
# finetune=true => resume_train loads ONLY the y-model weights; optimizer,
# learning rate, and epoch counter reset (so the decayed pretrain LR doesn't
# freeze fine-tuning). Same DBT x-model guider as Stage B.
# ---------------------------------------------------------------------------

X_MODEL=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /scratch/harisa_iitp/PIG_outputs/log \
    --desc_suffix x_mask_y_img_x_pretrained)
echo "Using DBT x-model guider: $X_MODEL"

# Best y-model from Stage B (VinDr pretrain).
Y_PRETRAINED=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /scratch/harisa_iitp/PIG_outputs/log \
    --desc_suffix x_mask_y_img_y_vindr_pretrain)
echo "Fine-tuning from VinDr-pretrained y-model: $Y_PRETRAINED"

python3 preprocess/patch_config.py \
    --train_model_type y \
    --x_model_path "$X_MODEL" \
    --ckpt_path "$Y_PRETRAINED" \
    --data_dir /scratch/harisa_iitp/data/slices_512 \
    --desc x_mask_y_img_y_dbt_finetune \
    --resume_train true \
    --finetune true \
    --lr 0.00005 \
    --n_epochs 40

python main.py

echo "Stage C (y-model DBT fine-tune) complete."
