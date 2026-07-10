#!/bin/bash
#SBATCH --job-name=PIG_vindr_d2j
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/vindr_d2j_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/vindr_d2j_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khalieqahmad@gmail.com

set -eo pipefail

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate pig_env

cd /home/harisa_iitp/PIG

# Mass-only pretraining set (~894 images). To also include normal-tissue images
# with blank masks, raise --include_normals (masks script handles them).
python3 preprocess/vindr_dicom_to_jpg.py \
    --vindr_root /scratch/harisa_iitp/data/vindr/physionet.org/files/vindr-mammo/1.0.0 \
    --out_dir /scratch/harisa_iitp/data/vindr/slices_512/img \
    --annotations_csv VinDr-Mammo/finding_annotations.csv \
    --include_normals 0

echo "VinDr DICOM -> JPG conversion complete."
