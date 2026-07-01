"""
Generate pixel-level lesion masks from BCS-DBT bounding-box annotations using SAM.

For each row in the boxes CSV that has a matching converted JPG slice
({PatientID}_{StudyUID}_{Slice:03d}.jpg in img_dir), this:
  1. Reads the original DICOM to get Rows/Columns (box coords are in original
     pixel space, but the JPGs were resized to img_size, e.g. 512x512).
  2. Scales the box into the resized JPG's coordinate space.
  3. Runs SAM with the box as a prompt to get a pixel mask.
  4. Saves a binary (0/255) mask JPG to mask_dir with the same filename as the image.

Only slices with a matching annotation get a mask -- this script also writes
matched_slices.txt listing exactly which img filenames got a mask, so a later
step can prune img/ down to the same set (PIG requires img/, mask/, masked/,
multi_mask/ to all have matching file counts).

Requires a GPU. Run inside an interactive session, e.g.:
    salloc --nodes=1 --time=4:00:00 --partition=gpu --gres=gpu:1
    conda activate pig_env

Requires: pip install segment-anything
Checkpoint:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        -P /scratch/harisa_iitp/

Usage:
    python3 preprocess/generate_masks.py \
        --boxes_csv BCS-DBT-boxes-train-v2.csv \
        --dicom_dir /scratch/harisa_iitp/data/dicom_annotated \
        --img_dir /scratch/harisa_iitp/data/slices_512/img \
        --mask_dir /scratch/harisa_iitp/data/slices_512/mask \
        --sam_checkpoint /scratch/harisa_iitp/sam_vit_h_4b8939.pth \
        --img_size 512 512
"""
import argparse
import csv
import os

import numpy as np
import pydicom
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def build_dicom_index(dicom_dir: str) -> dict:
    """Map (PatientID, StudyUID) -> (dicom_path, orig_rows, orig_cols)."""
    index = {}
    for root, _, files in os.walk(dicom_dir):
        for f in files:
            if not f.lower().endswith(".dcm"):
                continue
            path = os.path.join(root, f)
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
            study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN"))
            index[(patient_id, study_uid)] = (
                path,
                int(ds.Rows),
                int(ds.Columns),
            )
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes_csv", required=True)
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_h")
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--matched_list", default=None,
                         help="Optional path to write list of matched img filenames")
    args = parser.parse_args()

    os.makedirs(args.mask_dir, exist_ok=True)

    print("Indexing DICOM headers...")
    dicom_index = build_dicom_index(args.dicom_dir)
    print(f"Indexed {len(dicom_index)} DICOM series")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({args.sam_model_type}) on {device}...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    out_w, out_h = args.img_size

    matched = []
    total_rows = 0
    with open(args.boxes_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            patient_id = row["PatientID"]
            study_uid = row["StudyUID"]
            slice_idx = int(row["Slice"])

            img_name = f"{patient_id}_{study_uid}_{slice_idx:03d}.jpg"
            img_path = os.path.join(args.img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            key = (patient_id, study_uid)
            if key not in dicom_index:
                print(f"  no DICOM header found for {key}, skipping {img_name}")
                continue
            _, orig_rows, orig_cols = dicom_index[key]

            scale_x = out_w / orig_cols
            scale_y = out_h / orig_rows
            x = float(row["X"]) * scale_x
            y = float(row["Y"]) * scale_y
            w = float(row["Width"]) * scale_x
            h = float(row["Height"]) * scale_y
            box = np.array([x, y, x + w, y + h])

            image = np.array(Image.open(img_path).convert("RGB"))
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            best_mask = masks[int(np.argmax(scores))]

            mask_img = (best_mask.astype(np.uint8) * 255)
            Image.fromarray(mask_img, mode="L").save(
                os.path.join(args.mask_dir, img_name), quality=95
            )
            matched.append(img_name)
            print(f"[{len(matched)}] {img_name} <- box scaled from ({orig_cols}x{orig_rows})")

    print(f"Done. {len(matched)}/{total_rows} annotation rows matched and masked.")

    if args.matched_list:
        with open(args.matched_list, "w") as f:
            f.write("\n".join(matched) + "\n")
        print(f"Wrote matched filename list -> {args.matched_list}")


if __name__ == "__main__":
    main()
