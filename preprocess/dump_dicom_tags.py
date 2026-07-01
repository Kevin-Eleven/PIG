"""
Dump all DICOM tags for the first file found, to locate the laterality field
(ImageLaterality came back blank -- need to find where L/R is actually stored).

Usage:
    python3 preprocess/dump_dicom_tags.py --dicom_dir /scratch/harisa_iitp/data/dicom_annotated
"""
import argparse
import os

import pydicom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    dcm_files = []
    for root, _, files in os.walk(args.dicom_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, f))

    for dcm_path in dcm_files[:args.n]:
        print(f"===== {dcm_path} =====")
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        print(ds)
        print()


if __name__ == "__main__":
    main()
