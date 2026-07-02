#!/bin/bash
# Submit the full preprocessing + training chain as one command, run once from
# the login node AFTER download_annotated_series.py has finished (that step
# stays outside this chain since it needs internet access and TCIA auth,
# which the sbatch compute nodes may not have).
#
# Each job depends on the previous one via --dependency=afterok, so once this
# script returns you can log out / close the terminal and the whole sequence
# (DICOM->JPG -> SAM masks -> masked/multi_mask -> Phase 1 x-model training ->
# Phase 2 y-model training) runs unattended.
#
# Usage:
#   cd /home/harisa_iitp/PIG
#   bash preprocess/submit_pipeline.sh

set -euo pipefail

mkdir -p /scratch/harisa_iitp/PIG_outputs/log
mkdir -p /scratch/harisa_iitp/PIG_outputs/result

JOB_A=$(sbatch --parsable preprocess/job_dicom_to_jpg.sh)
echo "Job A (dicom_to_jpg)      : $JOB_A"

JOB_B=$(sbatch --parsable --dependency=afterok:$JOB_A preprocess/job_generate_masks.sh)
echo "Job B (generate_masks)    : $JOB_B  (after $JOB_A)"

JOB_C=$(sbatch --parsable --dependency=afterok:$JOB_B preprocess/job_generate_derived.sh)
echo "Job C (generate_derived)  : $JOB_C  (after $JOB_B)"

JOB_D=$(sbatch --parsable --dependency=afterok:$JOB_C train_pig.sh)
echo "Job D (train Phase 1, x)  : $JOB_D  (after $JOB_C)"

JOB_E=$(sbatch --parsable --dependency=afterok:$JOB_D train_pig_phase2.sh)
echo "Job E (train Phase 2, y)  : $JOB_E  (after $JOB_D)"

echo ""
echo "All jobs submitted. Monitor with: squeue --me"
echo "If any job fails, its dependents are cancelled automatically (afterok)."
