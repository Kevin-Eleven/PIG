#!/bin/bash
# Stage C: fine-tune the VinDr-pretrained y-model on the DBT slices.
#
#   salloc --nodes=1 --time=4:00:00 --partition=gpu --gres=gpu:2
#   source run_env.sh
#   bash run_y_finetune.sh 40
#
# finetune=true => loads ONLY the y-model weights from ckpt_path; optimizer, LR
# and epoch counter reset. (A plain resume would carry over the decayed pretrain
# LR and effectively freeze fine-tuning.) LR is dropped to 5e-5 for the same reason.

set -eo pipefail

N_EPOCHS="${1:-40}"
DESC=x_mask_y_img_y_dbt_finetune

X_MODEL=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /home/harisa_iitp/log \
    --desc_suffix x_mask_y_img_x_pretrained)
echo "DBT x-model guider: $X_MODEL"

# Best (lowest-loss) y-model from Stage B -- here 'best' IS what we want.
Y_PRETRAINED=$(python3 preprocess/find_best_checkpoint.py \
    --log_root /scratch/harisa_iitp/PIG_outputs/log \
    --desc_suffix x_mask_y_img_y_vindr_pretrain)
echo "Fine-tuning from: $Y_PRETRAINED"

python3 preprocess/patch_config.py \
    --train_model_type y \
    --x_model_path "$X_MODEL" \
    --ckpt_path "$Y_PRETRAINED" \
    --data_dir /scratch/harisa_iitp/data/slices_512 \
    --desc "$DESC" \
    --resume_train true \
    --finetune true \
    --lr 0.00005 \
    --n_epochs "$N_EPOCHS"

python3 main.py

echo "Stage C complete."
