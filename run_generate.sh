#!/bin/bash
#SBATCH --job-name=PIG_generate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/PIG_outputs/generate_%j.out
#SBATCH --error=/scratch/%u/PIG_outputs/generate_%j.err

set -e

module purge
module load MLDL/miniconda3
conda activate pig_env

cd /home/harisa_iitp/PIG
export PYTHONPATH=$(dirname "$PWD")

echo "===== Job Started ====="
date

python main.py

echo "===== Job Finished Successfully ====="
date
touch /scratch/$USER/PIG_outputs/GENERATION_SUCCESS_${SLURM_JOB_ID}