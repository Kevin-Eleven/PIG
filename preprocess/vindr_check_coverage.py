"""
Report how much of VinDr-Mammo actually landed on disk.

The PhysioNet downloader creates the full study-dir tree up front, so counting
directories overstates completeness -- only counting *.dicom files tells the
truth. Prints overall coverage plus, specifically, coverage of the training-split
Mass images (the ones the PIG y-model pretraining set is built from).

Usage:
    python3 preprocess/vindr_check_coverage.py
    python3 preprocess/vindr_check_coverage.py --write_missing missing.txt
"""
import argparse
import ast
import csv
import os

DEFAULT_ROOT = "/scratch/harisa_iitp/data/vindr/physionet.org/files/vindr-mammo/1.0.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindr_root", default=DEFAULT_ROOT)
    parser.add_argument("--annotations_csv", default="VinDr-Mammo/finding_annotations.csv")
    parser.add_argument("--split", default="training")
    parser.add_argument("--write_missing", default=None,
                        help="Write the missing <study_id>/<image_id>.dicom paths here")
    args = parser.parse_args()

    images_root = os.path.join(args.vindr_root, "images")

    total_dicom = 0
    for _, _, files in os.walk(images_root):
        total_dicom += sum(1 for f in files if f.endswith(".dicom"))
    n_studies = len(os.listdir(images_root))
    print(f"study dirs on disk : {n_studies}")
    print(f".dicom files       : {total_dicom}  (complete dataset = 20000)")
    print(f"overall coverage   : {100.0 * total_dicom / 20000:.1f}%\n")

    have, missing = [], []
    for row in csv.DictReader(open(args.annotations_csv)):
        if row["split"] != args.split or not row["xmin"]:
            continue
        try:
            cats = ast.literal_eval(row["finding_categories"])
        except (ValueError, SyntaxError):
            cats = [row["finding_categories"]]
        if "Mass" not in cats:
            continue
        rel = os.path.join(row["study_id"], row["image_id"] + ".dicom")
        (have if os.path.exists(os.path.join(images_root, rel)) else missing).append(rel)

    # An image can carry several Mass rows; dedupe before reporting.
    have, missing = sorted(set(have)), sorted(set(missing))
    total = len(have) + len(missing)
    print(f"Mass ({args.split}) images present : {len(have)}")
    print(f"Mass ({args.split}) images missing : {len(missing)}")
    if total:
        print(f"Mass coverage      : {100.0 * len(have) / total:.1f}%  of {total}")

    if args.write_missing and missing:
        with open(args.write_missing, "w") as f:
            f.write("\n".join(missing) + "\n")
        print(f"\nWrote {len(missing)} missing paths -> {args.write_missing}")


if __name__ == "__main__":
    main()
