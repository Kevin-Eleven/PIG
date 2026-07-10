#!/bin/bash
# Full VinDr preprocessing, run interactively inside an salloc GPU session.
# Needs a GPU only for the SAM mask step; the other two steps are CPU-bound but
# run fine on the GPU node.
#
#   salloc --nodes=1 --time=3:00:00 --partition=gpu --gres=gpu:1
#   source run_env.sh
#   bash run_vindr_preprocess.sh
#
# Idempotent: safe to re-run. Steps that already produced their output are
# still re-done (cheap enough), so a partial failure just needs a re-run.

set -eo pipefail

VINDR_ROOT=/scratch/harisa_iitp/data/vindr/physionet.org/files/vindr-mammo/1.0.0
OUT=/scratch/harisa_iitp/data/vindr/slices_512
SAM_CKPT=/scratch/harisa_iitp/sam_vit_h_4b8939.pth

echo "############ STEP 1/3: DICOM -> JPG ############"
python3 preprocess/vindr_dicom_to_jpg.py \
    --vindr_root "$VINDR_ROOT" \
    --out_dir "$OUT/img" \
    --annotations_csv VinDr-Mammo/finding_annotations.csv \
    --include_normals 0

echo "############ STEP 2/3: SAM masks (GPU) ############"
python3 preprocess/vindr_generate_masks.py \
    --annotations_csv VinDr-Mammo/finding_annotations.csv \
    --img_dir "$OUT/img" \
    --mask_dir "$OUT/mask" \
    --sam_checkpoint "$SAM_CKPT" \
    --img_size 512 512

echo "############ STEP 3/3: masked/ + multi_mask/ ############"
python3 preprocess/generate_derived.py \
    --data_dir "$OUT" \
    --dilate_kernel 15

echo "############ DONE. File counts: ############"
for d in img mask masked multi_mask; do
    printf "%-12s %s\n" "$d" "$(ls "$OUT/$d" | wc -l)"
done
