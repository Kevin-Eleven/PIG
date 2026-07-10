#!/bin/bash
#SBATCH --job-name=PIG_selftest
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=/home/harisa_iitp/selftest_%j.out
#SBATCH --error=/home/harisa_iitp/selftest_%j.err

# Diagnostic only: isolates why the vindr preprocess jobs died in <1s with no
# output file. Writes its log to $HOME (not /scratch) so we can tell a scratch
# write problem apart from a job-launch problem.

echo "=== node: $(hostname) ==="
echo "=== date: $(date) ==="

echo "--- can we write to scratch? ---"
touch /scratch/harisa_iitp/PIG_outputs/selftest_touch_${SLURM_JOB_ID} \
    && echo "scratch write OK" \
    || echo "scratch write FAILED"

echo "--- module load ---"
module purge
module load MLDL/miniconda3
echo "module load rc=$?"

echo "--- conda ---"
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
echo "conda.sh rc=$?"
conda activate pig_env
echo "conda activate rc=$?  python=$(which python3)"

echo "--- imports ---"
python3 -c "import pydicom, numpy, PIL; print('core ok')"
python3 -c "import pylibjpeg, openjpeg; print('jpeg2000 ok')" || echo "jpeg2000 MISSING"

echo "--- vindr root visible from compute node? ---"
ls -d /scratch/harisa_iitp/data/vindr/physionet.org/files/vindr-mammo/1.0.0 \
    && echo "vindr root OK" || echo "vindr root NOT VISIBLE"

echo "=== selftest complete ==="
