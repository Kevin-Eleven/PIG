"""
Generate the masked/ and multi_mask/ folders PIG expects alongside img/ and mask/:
  masked      = img * (mask / 255)               -- lesion region cut out of the slice
  multi_mask  = dilated mask (binary, 0/255)      -- grown lesion region

Also validates that img/, mask/, masked/, multi_mask/ end up with identical file
counts, matching the assert in utils/dataset.py -- fails loudly here instead of
letting a mismatch surface as a cryptic assertion error during training.

Usage:
    python3 preprocess/generate_derived.py \
        --data_dir /scratch/harisa_iitp/data/slices_512 \
        --dilate_kernel 15
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/scratch/harisa_iitp/data/slices_512")
    parser.add_argument("--dilate_kernel", type=int, default=15,
                         help="Diameter (px) of the elliptical structuring element used to dilate mask -> multi_mask")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "img")
    mask_dir = os.path.join(args.data_dir, "mask")
    masked_dir = os.path.join(args.data_dir, "masked")
    multi_mask_dir = os.path.join(args.data_dir, "multi_mask")
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(multi_mask_dir, exist_ok=True)

    mask_names = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".jpg"))
    print(f"Found {len(mask_names)} mask files in {mask_dir}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate_kernel, args.dilate_kernel))

    for i, name in enumerate(mask_names):
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(mask_dir, name)
        if not os.path.exists(img_path):
            print(f"  missing img for mask {name}, skipping")
            continue

        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        masked = (img.astype(np.float32) * (mask.astype(np.float32) / 255.0)).astype(np.uint8)
        Image.fromarray(masked, mode="L").save(os.path.join(masked_dir, name), quality=95)

        dilated = cv2.dilate(mask, kernel)
        Image.fromarray(dilated, mode="L").save(os.path.join(multi_mask_dir, name), quality=95)

        if (i + 1) % 50 == 0 or i + 1 == len(mask_names):
            print(f"[{i + 1}/{len(mask_names)}] {name}")

    counts = {
        "img": len([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]),
        "mask": len([f for f in os.listdir(mask_dir) if f.lower().endswith(".jpg")]),
        "masked": len([f for f in os.listdir(masked_dir) if f.lower().endswith(".jpg")]),
        "multi_mask": len([f for f in os.listdir(multi_mask_dir) if f.lower().endswith(".jpg")]),
    }
    print(f"\nFile counts: {counts}")
    if len(set(counts.values())) != 1:
        raise SystemExit(
            f"FATAL: img/mask/masked/multi_mask counts do not match: {counts}. "
            f"utils/dataset.py asserts these are equal -- training would crash on load."
        )
    print("OK: all four folders have matching file counts.")


if __name__ == "__main__":
    main()
