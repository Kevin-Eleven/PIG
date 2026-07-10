#!/bin/bash
# Download ONLY the training-split Mass DICOMs from VinDr-Mammo.
#
# Rationale: the full dataset is ~345 GB (20000 images x ~17 MB), which does not
# fit in the 200 GB scratch quota. PIG's y-model pretraining set only uses the
# 894 training-split Mass images (~15 GB), so fetch exactly those.
#
# Resumable: -c skips/continues already-downloaded files, so re-running after an
# interruption is cheap and safe.
#
# Usage (login node, inside screen -- this takes a while):
#     screen -S vindr
#     bash preprocess/vindr_download_mass.sh <physionet_username>
#     # detach: Ctrl+A then D

set -eo pipefail

USER_NAME="${1:?usage: $0 <physionet_username>}"
ROOT=/scratch/harisa_iitp/data/vindr
DEST="$ROOT/physionet.org/files/vindr-mammo/1.0.0"
LIST=/scratch/harisa_iitp/data/vindr_mass_urls.txt

cd /home/harisa_iitp/PIG

echo "=== Building list of missing Mass DICOMs ==="
python3 preprocess/vindr_check_coverage.py \
    --vindr_root "$DEST" \
    --write_missing /scratch/harisa_iitp/data/vindr_mass_missing.txt

if [ ! -s /scratch/harisa_iitp/data/vindr_mass_missing.txt ]; then
    echo "Nothing missing -- all Mass images already present."
    exit 0
fi

# Turn "<study_id>/<image_id>.dicom" into full PhysioNet URLs.
sed 's|^|https://physionet.org/files/vindr-mammo/1.0.0/images/|' \
    /scratch/harisa_iitp/data/vindr_mass_missing.txt > "$LIST"
echo "=== $(wc -l < "$LIST") files to fetch (~$(( $(wc -l < "$LIST") * 17 / 1024 )) GB) ==="

echo "=== Downloading (you will be prompted once for your PhysioNet password) ==="
# -nH drops the host dir and -x recreates files/vindr-mammo/1.0.0/images/<study>/...
# under -P, reproducing exactly the layout the rest of the pipeline expects.
wget -c -i "$LIST" \
    --user "$USER_NAME" --ask-password \
    -x -nH -P "$ROOT/physionet.org" \
    --progress=dot:giga

echo "=== Done. Re-checking coverage ==="
python3 preprocess/vindr_check_coverage.py --vindr_root "$DEST"
