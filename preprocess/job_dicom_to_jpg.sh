#!/bin/bash
#SBATCH --job-name=PIG_dicom2jpg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/dicom2jpg_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/dicom2jpg_%j.err

set -eo pipefail

module purge
module load MLDL/miniconda3
conda activate pig_env

cd /home/harisa_iitp/PIG

python3 preprocess/dicom_to_jpg.py \
    --dicom_dir /scratch/harisa_iitp/data/dicom_annotated \
    --out_dir /scratch/harisa_iitp/data/slices_512/img \
    --boxes_csv BCS-DBT-boxes-train-v2.csv

echo "DICOM -> JPG conversion complete."
