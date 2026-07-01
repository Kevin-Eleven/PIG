"""
Diagnostic: print the (PatientID, view_key) computed from every downloaded
DICOM, and the (PatientID, view_key) computed from every CSV row, to find
why generate_masks.py's join finds 0 matches.

Usage:
    python3 preprocess/debug_match2.py \
        --boxes_csv BCS-DBT-boxes-train-v2.csv \
        --dicom_dir /scratch/harisa_iitp/data/dicom_annotated
"""
import argparse
import csv
import os
import re

import pydicom


def get_view_key(ds) -> str:
    laterality = ""
    try:
        laterality = str(ds.ViewCodeSequence[0].FrameAnatomySequence[0].FrameLaterality)
    except (AttributeError, IndexError):
        laterality = str(getattr(ds, "ImageLaterality", ""))
    view_position = str(getattr(ds, "ViewPosition", ""))
    return (laterality + view_position).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes_csv", required=True)
    parser.add_argument("--dicom_dir", required=True)
    args = parser.parse_args()

    print("=== DICOM-derived keys ===")
    dcm_keys = set()
    for root, _, files in os.walk(args.dicom_dir):
        for f in files:
            if not f.lower().endswith(".dcm"):
                continue
            path = os.path.join(root, f)
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
            view_key = get_view_key(ds)
            dcm_keys.add((patient_id, view_key))
            print(repr((patient_id, view_key)))

    print("\n=== CSV-derived keys (first 40) ===")
    csv_keys = set()
    with open(args.boxes_csv, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            patient_id = row["PatientID"]
            view_key = re.sub(r"\d+$", "", row["View"]).lower()
            csv_keys.add((patient_id, view_key))
            if i < 40:
                print(repr((patient_id, view_key)))

    print(f"\nOverlap: {len(dcm_keys & csv_keys)}")
    print("Sample overlap:", list(dcm_keys & csv_keys)[:10])


if __name__ == "__main__":
    main()
