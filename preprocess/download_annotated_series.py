"""
Download only the exact DICOM series that have box annotations, instead of
whole patients (which pulls in unrelated, unannotated studies) or a blind
prefix of the whole collection.

BCS-DBT-boxes-train-v2.csv and BCS-DBT-file-paths-train-v2.csv share the join
key (PatientID, StudyUID, View) -- StudyUID here is the short TCIA study ID
(e.g. "DBT-S00163"), not the DICOM StudyInstanceUID. The file-paths CSV's
classic_path encodes the real identifiers as:
    .../{PatientID}/{StudyInstanceUID}/{SeriesInstanceUID}/1-1.dcm
so the SeriesInstanceUID is classic_path.split('/')[-2]. Downloading by exact
SeriesInstanceUID (rather than by PatientID) pulls only the ~196 annotated
series instead of every series ever taken of these 101 patients.

Also fetches the SAM ViT-H checkpoint used by generate_masks.py.

Usage:
    python3 preprocess/download_annotated_series.py \
        --boxes_csv BCS-DBT-boxes-train-v2.csv \
        --file_paths_csv BCS-DBT-file-paths-train-v2.csv \
        --dicom_dir /scratch/harisa_iitp/data/dicom_annotated \
        --sam_dir /scratch/harisa_iitp
"""
import argparse
import csv
import os
import re
import subprocess

from tcia_utils import nbia

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def view_key(view: str) -> str:
    # A slice can carry a trailing digit for a second box on the same view
    # (e.g. "lmlo1") -- strip it so it matches the file-paths CSV's plain view.
    return re.sub(r"\d+$", "", view).lower()


def build_series_uid_map(file_paths_csv: str) -> dict:
    """Map (PatientID, StudyUID, view_key) -> SeriesInstanceUID."""
    mapping = {}
    with open(file_paths_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["PatientID"], row["StudyUID"], view_key(row["View"]))
            series_uid = row["classic_path"].split("/")[-2]
            mapping[key] = series_uid
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes_csv", required=True)
    parser.add_argument("--file_paths_csv", required=True)
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--sam_dir", default=None,
                         help="If set, wget the SAM ViT-H checkpoint into this directory")
    args = parser.parse_args()

    series_uid_map = build_series_uid_map(args.file_paths_csv)

    needed_keys = set()
    with open(args.boxes_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            needed_keys.add((row["PatientID"], row["StudyUID"], view_key(row["View"])))

    series_uids = set()
    unjoined = []
    for key in needed_keys:
        if key in series_uid_map:
            series_uids.add(series_uid_map[key])
        else:
            unjoined.append(key)

    print(f"{len(needed_keys)} annotated (PatientID, StudyUID, View) combos in {args.boxes_csv}")
    print(f"{len(series_uids)} unique SeriesInstanceUIDs resolved for download")
    if unjoined:
        print(f"WARNING: {len(unjoined)} combos had no match in {args.file_paths_csv}:")
        for key in unjoined:
            print(f"  {key}")

    os.makedirs(args.dicom_dir, exist_ok=True)
    series_uids = sorted(series_uids)
    print(f"\nDownloading {len(series_uids)} series into {args.dicom_dir} ...")
    nbia.downloadSeries(series_uids, path=args.dicom_dir, input_type="list")
    print("Done downloading series.")

    if args.sam_dir:
        os.makedirs(args.sam_dir, exist_ok=True)
        sam_path = os.path.join(args.sam_dir, "sam_vit_h_4b8939.pth")
        if os.path.exists(sam_path):
            print(f"SAM checkpoint already present at {sam_path}, skipping download")
        else:
            print(f"Downloading SAM checkpoint to {sam_path} ...")
            subprocess.run(["wget", "-O", sam_path, SAM_CHECKPOINT_URL], check=True)


if __name__ == "__main__":
    main()
