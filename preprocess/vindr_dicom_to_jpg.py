"""
Convert VinDr-Mammo DICOMs into 512x512 grayscale JPG slices for PIG y-model
pretraining.

Differences from the BCS-DBT converter (dicom_to_jpg.py):
  - Files are single-frame, named images/<study_id>/<image_id>.dicom (not .dcm).
  - Selection is driven by finding_annotations.csv: by default only training-split
    images that carry at least one "Mass" finding are converted (those are the ones
    that will get SAM masks). --include_normals N additionally converts N
    "No Finding" training images, which later get blank masks (teaches normal
    breast tissue appearance).
  - Mammography DICOMs need VOI LUT windowing and are frequently MONOCHROME1
    (inverted: high pixel value = dark) -- both handled here.

Output filenames are <image_id>.jpg (image_id is a unique hash in VinDr).

Usage:
    python3 preprocess/vindr_dicom_to_jpg.py \
        --vindr_root /scratch/harisa_iitp/data/vindr/physionet.org/files/vindr-mammo/1.0.0 \
        --out_dir /scratch/harisa_iitp/data/vindr/slices_512/img \
        --annotations_csv VinDr-Mammo/finding_annotations.csv \
        --include_normals 0

Requires: pydicom, pylibjpeg, pylibjpeg-libjpeg, numpy, Pillow
"""
import argparse
import ast
import csv
import os

import numpy as np
import pydicom
from PIL import Image

try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:  # pydicom >= 3.0 moved it
    from pydicom.pixels import apply_voi_lut


def parse_categories(raw: str) -> list:
    """finding_categories is a stringified python list, e.g. "['Mass', 'Skin Retraction']"."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw]


def select_image_ids(annotations_csv: str, include_normals: int, split: str) -> dict:
    """Return {image_id: study_id} of images to convert."""
    mass_ids, normal_ids = {}, {}
    with open(annotations_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            cats = parse_categories(row["finding_categories"])
            if "Mass" in cats and row["xmin"]:
                mass_ids[row["image_id"]] = row["study_id"]
            elif "No Finding" in cats:
                normal_ids[row["image_id"]] = row["study_id"]

    selected = dict(mass_ids)
    if include_normals > 0:
        # Sort for determinism, drop any overlap with mass images (an image can
        # have several finding rows).
        extra = [(iid, sid) for iid, sid in sorted(normal_ids.items()) if iid not in mass_ids]
        selected.update(dict(extra[:include_normals]))
    print(f"Selected {len(mass_ids)} Mass images + "
          f"{len(selected) - len(mass_ids)} normal images ({split} split)")
    return selected


def dicom_to_uint8(ds) -> np.ndarray:
    """Decode one VinDr mammogram: VOI LUT -> MONOCHROME1 inversion -> percentile
    normalization to uint8 (same 0.5/99.5 window as the BCS-DBT converter)."""
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass  # fall back to raw pixel values
    arr = arr.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return (arr * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindr_root", required=True,
                        help="Dir containing images/<study_id>/<image_id>.dicom")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--annotations_csv", default="VinDr-Mammo/finding_annotations.csv")
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--include_normals", type=int, default=0,
                        help="Also convert this many 'No Finding' images (blank masks later)")
    parser.add_argument("--split", default="training")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    selected = select_image_ids(args.annotations_csv, args.include_normals, args.split)

    written, failed, missing = 0, 0, 0
    for idx, (image_id, study_id) in enumerate(sorted(selected.items())):
        dcm_path = os.path.join(args.vindr_root, "images", study_id, f"{image_id}.dicom")
        if not os.path.exists(dcm_path):
            missing += 1
            print(f"[{idx + 1}/{len(selected)}] MISSING: {dcm_path}")
            continue
        try:
            ds = pydicom.dcmread(dcm_path)
            img8 = dicom_to_uint8(ds)
            im = Image.fromarray(img8, mode="L").resize(tuple(args.img_size), Image.BILINEAR)
            im.save(os.path.join(args.out_dir, f"{image_id}.jpg"), quality=95)
            written += 1
            if written % 50 == 0:
                print(f"[{idx + 1}/{len(selected)}] {written} written")
        except Exception as e:
            failed += 1
            print(f"[{idx + 1}/{len(selected)}] FAILED: {dcm_path} ({e})")

    print(f"Done. written={written}, failed={failed}, missing={missing} -> {args.out_dir}")
    if written == 0:
        raise SystemExit("FATAL: no images written -- check --vindr_root path")


if __name__ == "__main__":
    main()
