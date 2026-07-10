#!/bin/bash
#SBATCH --job-name=PIG_y_pretrain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/y_pretrain_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/y_pretrain_%j.err
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
# Stage B: pretrain the y-model (image appearance) on VinDr-Mammo.
# Uses the existing DBT x-model as the diffusion guider (exactly the paper's
# scheme and what Stage C/D fine-tuning + sampling will use). Fresh training
# from scratch (resume_train=False), so the DBT x-model just provides the mask
# conditioning signal.
# ---------------------------------------------------------------------------

# Locate the DBT x-model trained in Phase 1 (desc x_mask_y_img_x_pretrained).
# NOTE: this lives in ~/log, not the scratch log dir -- the scratch copies are
# older/worse runs of the same desc, and picking those would degrade the guider.
X_MODEL=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /home/harisa_iitp/log \
    --desc_suffix x_mask_y_img_x_pretrained)
echo "Using DBT x-model guider: $X_MODEL"

python3 preprocess/patch_config.py \
    --train_model_type y \
    --x_model_path "$X_MODEL" \
    --data_dir /scratch/harisa_iitp/data/vindr/slices_512 \
    --desc x_mask_y_img_y_vindr_pretrain \
    --resume_train false \
    --finetune false \
    --n_epochs 120

python main.py

echo "Stage B (y-model VinDr pretrain) complete."
