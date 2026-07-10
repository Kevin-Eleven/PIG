#!/bin/bash
# Stage B: pretrain the y-model (image appearance) on VinDr-Mammo, using the
# existing DBT x-model as the diffusion guider.
#
#   salloc --nodes=1 --time=6:00:00 --partition=gpu --gres=gpu:2
#   source run_env.sh
#   bash run_y_pretrain.sh 30          # run 30 epochs this session
#
# Interactive sessions are time-capped, so this trains in CHUNKS. First call
# starts from scratch; use run_y_pretrain_resume.sh for every session after.

set -eo pipefail

N_EPOCHS="${1:-30}"
DESC=x_mask_y_img_y_vindr_pretrain

X_MODEL=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /home/harisa_iitp/log \
    --desc_suffix x_mask_y_img_x_pretrained)
echo "DBT x-model guider: $X_MODEL"

python3 preprocess/patch_config.py \
    --train_model_type y \
    --x_model_path "$X_MODEL" \
    --data_dir /scratch/harisa_iitp/data/vindr/slices_512 \
    --desc "$DESC" \
    --resume_train false \
    --finetune false \
    --n_epochs "$N_EPOCHS"

python3 main.py

echo "Stage B chunk complete ($N_EPOCHS epochs). Resume with run_y_pretrain_resume.sh"
