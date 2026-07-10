#!/bin/bash
# Stage B, continued: resume VinDr y-model pretraining from the newest checkpoint.
# Run this in every session AFTER the first run_y_pretrain.sh.
#
#   salloc --nodes=1 --time=6:00:00 --partition=gpu --gres=gpu:2
#   source run_env.sh
#   bash run_y_pretrain_resume.sh 30
#
# resume_train=true (finetune=false) restores model + optimizer + epoch counter,
# so the LR schedule continues rather than restarting. n_epochs is the number of
# ADDITIONAL epochs to run (the loop is range(start_epoch, start_epoch+n_epochs)).

set -eo pipefail

N_EPOCHS="${1:-30}"
DESC=x_mask_y_img_y_vindr_pretrain

X_MODEL=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /home/harisa_iitp/log \
    --desc_suffix x_mask_y_img_x_pretrained)
echo "DBT x-model guider: $X_MODEL"

# --latest, not best: resuming from the lowest-loss ckpt would rewind training.
CKPT=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /scratch/harisa_iitp/PIG_outputs/log \
    --desc_suffix "$DESC" --latest)
echo "Resuming from: $CKPT"

python3 preprocess/patch_config.py \
    --train_model_type y \
    --x_model_path "$X_MODEL" \
    --ckpt_path "$CKPT" \
    --data_dir /scratch/harisa_iitp/data/vindr/slices_512 \
    --desc "$DESC" \
    --resume_train true \
    --finetune false \
    --n_epochs "$N_EPOCHS"

python3 main.py

echo "Stage B chunk complete (+$N_EPOCHS epochs)."
