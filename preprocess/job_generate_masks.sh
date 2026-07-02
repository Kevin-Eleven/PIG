#!/bin/bash
#SBATCH --job-name=PIG_gen_masks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/gen_masks_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/gen_masks_%j.err

set -eo pipefail

module purge
module load MLDL/miniconda3
conda activate pig_env

cd /home/harisa_iitp/PIG

python3 preprocess/generate_masks.py \
    --boxes_csv BCS-DBT-boxes-train-v2.csv \
    --dicom_dir /scratch/harisa_iitp/data/dicom_annotated \
    --img_dir /scratch/harisa_iitp/data/slices_512/img \
    --mask_dir /scratch/harisa_iitp/data/slices_512/mask \
    --sam_checkpoint /scratch/harisa_iitp/sam_vit_h_4b8939.pth \
    --img_size 512 512 \
    --matched_list /scratch/harisa_iitp/data/slices_512/matched_slices.txt

echo "SAM mask generation complete."
