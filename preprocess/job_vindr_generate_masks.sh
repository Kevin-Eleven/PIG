#!/bin/bash
#SBATCH --job-name=PIG_vindr_masks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/vindr_masks_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/vindr_masks_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khalieqahmad@gmail.com

set -eo pipefail

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate pig_env

cd /home/harisa_iitp/PIG

python3 preprocess/vindr_generate_masks.py \
    --annotations_csv VinDr-Mammo/finding_annotations.csv \
    --img_dir /scratch/harisa_iitp/data/vindr/slices_512/img \
    --mask_dir /scratch/harisa_iitp/data/vindr/slices_512/mask \
    --sam_checkpoint /scratch/harisa_iitp/sam_vit_h_4b8939.pth \
    --img_size 512 512

echo "VinDr SAM mask generation complete."
