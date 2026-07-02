#!/bin/bash
#SBATCH --job-name=PIG_gen_derived
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/gen_derived_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/gen_derived_%j.err

set -euo pipefail

module purge
module load MLDL/miniconda3
conda activate pig_env

cd /home/harisa_iitp/PIG

python3 preprocess/generate_derived.py \
    --data_dir /scratch/harisa_iitp/data/slices_512 \
    --dilate_kernel 15

echo "masked/ and multi_mask/ generation complete."
