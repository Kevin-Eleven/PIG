#!/bin/bash
#SBATCH --job-name=PIG_smoketest
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/harisa_iitp/PIG_outputs/smoketest_%j.out
#SBATCH --error=/scratch/harisa_iitp/PIG_outputs/smoketest_%j.err

set -eo pipefail

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate pig_env

cd /home/harisa_iitp/PIG

mkdir -p /scratch/harisa_iitp/PIG_outputs/log
ln -sfn /scratch/harisa_iitp/PIG_outputs/log ./log

# Times a handful of epochs of x-model training so we can extrapolate real
# wall-clock cost before committing to a full run against the 2-day gpu QOS cap.
python3 preprocess/patch_config.py --train_model_type x --n_epochs 3

python main.py

# Restore n_epochs to the real training value so a later real submission of
# train_pig.sh isn't accidentally left at 3.
python3 preprocess/patch_config.py --n_epochs 128

echo "Smoke test complete. Check per-epoch timing above / in the log CSV."
