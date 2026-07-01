"""
Rename already-converted slices (series0000_000.jpg style) to PatientID_StudyUID_000.jpg
by re-reading only the DICOM header (fast, no pixel decode) in the same directory-walk
order used by the original dicom_to_jpg.py run. Use this instead of re-running the full
conversion when you only need identifying metadata in the filenames.

Usage:
    python3 preprocess/rename_existing_slices.py \
        --dicom_dir /scratch/harisa_iitp/data/dicom_annotated \
        --img_dir /scratch/harisa_iitp/data/slices_512/img
"""
import argparse
import os
import re

import pydicom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    args = parser.parse_args()

    dcm_files = []
    for root, _, files in os.walk(args.dicom_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, f))

    renamed = 0
    for idx, dcm_path in enumerate(dcm_files):
        old_prefix = f"series{idx:04d}_"
        matches = [f for f in os.listdir(args.img_dir) if f.startswith(old_prefix)]
        if not matches:
            print(f"[{idx}] no files found with prefix {old_prefix}, skipping")
            continue

        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
        study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN"))
        new_prefix = f"{patient_id}_{study_uid}_"

        for fname in matches:
            suffix = fname[len(old_prefix):]
            old_path = os.path.join(args.img_dir, fname)
            new_path = os.path.join(args.img_dir, new_prefix + suffix)
            os.rename(old_path, new_path)
            renamed += 1

        print(f"[{idx + 1}/{len(dcm_files)}] {old_prefix} -> {new_prefix} ({len(matches)} files)")

    print(f"Done. Renamed {renamed} files.")


if __name__ == "__main__":
    main()
