"""
Convert BCS-DBT DICOM volumes (multi-frame) into individual 512x512 grayscale JPG slices.

With --boxes_csv, only the annotated (PatientID, view, Slice) frames are written
instead of every frame of every volume -- each volume has 50-90 frames but usually
only 1-2 are annotated, so filtering here avoids writing ~95% unusable JPGs and
keeps generate_masks.py's img/mask pairing trivial (1 img : 1 mask, no pruning step
needed).

Usage:
    python3 preprocess/dicom_to_jpg.py \
        --dicom_dir /scratch/harisa_iitp/data/dicom_annotated \
        --out_dir /scratch/harisa_iitp/data/slices_512/img \
        --boxes_csv BCS-DBT-boxes-train-v2.csv

Requires: pydicom, pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, numpy, Pillow
(BCS-DBT DICOMs are typically JPEG2000-compressed; pylibjpeg is needed to decode them.)
"""
import argparse
import csv
import os
import re

import numpy as np
import pydicom
from PIL import Image


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    lo, hi = np.percentile(frame, 0.5), np.percentile(frame, 99.5)
    if hi <= lo:
        lo, hi = frame.min(), frame.max()
    frame = np.clip(frame, lo, hi)
    frame = (frame - lo) / max(hi - lo, 1e-6)
    return (frame * 255).astype(np.uint8)


def get_view_key(ds) -> str:
    """Return e.g. 'lmlo' / 'rcc' combining laterality + view position.

    ImageLaterality is blank in BCS-DBT DICOMs; the real laterality is
    nested under SharedFunctionalGroupsSequence[0].FrameAnatomySequence[0]
    .FrameLaterality (tag 0020,9072). Kept in sync with generate_masks.py.
    """
    try:
        laterality = str(
            ds.SharedFunctionalGroupsSequence[0].FrameAnatomySequence[0].FrameLaterality
        )
    except (AttributeError, IndexError):
        laterality = str(getattr(ds, "ImageLaterality", ""))
    view_position = str(getattr(ds, "ViewPosition", ""))
    return (laterality + view_position).lower()


def load_wanted_slices(boxes_csv: str) -> set:
    """Return {(PatientID, view_key, slice_idx)} from the boxes CSV."""
    wanted = set()
    with open(boxes_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            view_key = re.sub(r"\d+$", "", row["View"]).lower()
            wanted.add((row["PatientID"], view_key, int(row["Slice"])))
    return wanted


def convert_file(dcm_path: str, out_dir: str, img_size: tuple, wanted_slices: set) -> tuple:
    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array  # shape: (num_frames, H, W) or (H, W)

    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, ...]

    patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
    study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN"))
    file_id = f"{patient_id}_{study_uid}"
    view_key = get_view_key(ds) if wanted_slices is not None else None

    count = 0
    for i, frame in enumerate(pixels):
        if wanted_slices is not None and (patient_id, view_key, i) not in wanted_slices:
            continue
        img8 = normalize_to_uint8(frame)
        im = Image.fromarray(img8, mode="L").resize(img_size, Image.BILINEAR)
        out_path = os.path.join(out_dir, f"{file_id}_{i:03d}.jpg")
        im.save(out_path, quality=95)
        count += 1
    return count, patient_id, study_uid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", default="/scratch/harisa_iitp/data/dicom_annotated")
    parser.add_argument("--out_dir", default="/scratch/harisa_iitp/data/slices_512/img")
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--boxes_csv", default=None,
                         help="If set, only annotated (PatientID, View, Slice) frames are written")
    parser.add_argument("--mapping_csv", default=None,
                         help="Optional path to write a PatientID/StudyUID -> file prefix mapping CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wanted_slices = load_wanted_slices(args.boxes_csv) if args.boxes_csv else None
    if wanted_slices is not None:
        print(f"Filtering to {len(wanted_slices)} annotated (PatientID, View, Slice) combos "
              f"from {args.boxes_csv}")

    dcm_files = []
    for root, _, files in os.walk(args.dicom_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, f))

    print(f"Found {len(dcm_files)} DICOM files under {args.dicom_dir}")

    total_slices = 0
    mapping_rows = []
    for idx, dcm_path in enumerate(dcm_files):
        try:
            n, patient_id, study_uid = convert_file(dcm_path, args.out_dir, tuple(args.img_size), wanted_slices)
            total_slices += n
            mapping_rows.append((patient_id, study_uid, n, dcm_path))
            print(f"[{idx + 1}/{len(dcm_files)}] {dcm_path} -> {n} slices (PatientID={patient_id})")
        except Exception as e:
            print(f"[{idx + 1}/{len(dcm_files)}] FAILED: {dcm_path} ({e})")

    if args.mapping_csv:
        with open(args.mapping_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PatientID", "StudyUID", "NumSlices", "SourceDicom"])
            writer.writerows(mapping_rows)
        print(f"Wrote mapping CSV -> {args.mapping_csv}")

    print(f"Done. Total slices written: {total_slices} -> {args.out_dir}")


if __name__ == "__main__":
    main()
