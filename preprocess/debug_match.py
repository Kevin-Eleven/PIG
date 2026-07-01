"""
Diagnostic: compare DICOM header fields against the boxes CSV to figure out
the correct join key (PatientID/StudyUID in the CSV appear to be short
friendly IDs like DBT-S00163, not the raw DICOM StudyInstanceUID).

Usage:
    python3 preprocess/debug_match.py \
        --boxes_csv BCS-DBT-boxes-train-v2.csv \
        --dicom_dir /scratch/harisa_iitp/data/dicom_annotated
"""
import argparse
import csv
import os

import pydicom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes_csv", required=True)
    parser.add_argument("--dicom_dir", required=True)
    args = parser.parse_args()

    csv_patients = set()
    csv_rows_by_patient = {}
    with open(args.boxes_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_patients.add(row["PatientID"])
            csv_rows_by_patient.setdefault(row["PatientID"], []).append(row)

    dcm_files = []
    for root, _, files in os.walk(args.dicom_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, f))

    print(f"{len(dcm_files)} DICOM files found, {len(csv_patients)} patients in CSV\n")

    for dcm_path in dcm_files[:5]:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
        study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN"))
        study_desc = str(getattr(ds, "StudyDescription", ""))
        series_desc = str(getattr(ds, "SeriesDescription", ""))
        view_pos = str(getattr(ds, "ViewPosition", ""))
        laterality = str(getattr(ds, "ImageLaterality", ""))
        n_frames = str(getattr(ds, "NumberOfFrames", ""))

        print(f"--- {dcm_path}")
        print(f"  PatientID={patient_id}  in_csv={patient_id in csv_patients}")
        print(f"  StudyInstanceUID={study_uid}")
        print(f"  StudyDescription={study_desc!r}  SeriesDescription={series_desc!r}")
        print(f"  ViewPosition={view_pos!r}  ImageLaterality={laterality!r}  NumberOfFrames={n_frames}")

        if patient_id in csv_rows_by_patient:
            print(f"  CSV rows for this patient:")
            for row in csv_rows_by_patient[patient_id][:3]:
                print(f"    StudyUID={row['StudyUID']} View={row['View']} Slice={row['Slice']}")
        print()


if __name__ == "__main__":
    main()
