#!/bin/bash
#SBATCH --job-name=PIG_vindr_derived
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/vindr_derived_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/vindr_derived_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khalieqahmad@gmail.com

set -eo pipefail

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate pig_env

cd /home/harisa_iitp/PIG

# Reuses the BCS-DBT derived-folder generator unchanged -- it only needs
# img/ + mask/ with matching names.
python3 preprocess/generate_derived.py \
    --data_dir /scratch/harisa_iitp/data/vindr/slices_512 \
    --dilate_kernel 15

echo "VinDr masked/ and multi_mask/ generation complete."
