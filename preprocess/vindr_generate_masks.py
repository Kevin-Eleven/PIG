"""
Generate pixel-level lesion masks for VinDr-Mammo images using SAM, from the
xmin/ymin/xmax/ymax boxes in finding_annotations.csv.

Differences from the BCS-DBT version (generate_masks.py):
  - Boxes are already corner coordinates (xmin/ymin/xmax/ymax) in original pixel
    space; the original height/width are columns in the CSV, so no DICOM re-read
    is needed to rescale into the 512x512 JPG space.
  - Matching is by image_id (JPGs are named <image_id>.jpg), no slice/view logic.
  - Multiple Mass boxes on one image are unioned into a single mask, same as the
    BCS-DBT script.
  - After Mass masks are written, any remaining image in img_dir with no mask
    (i.e. 'No Finding' normals converted with --include_normals) gets an all-black
    blank mask, so all four PIG data folders end up with matching file counts.

Requires a GPU (SLURM gpu partition). Requires: pip install segment-anything

Usage:
    python3 preprocess/vindr_generate_masks.py \
        --annotations_csv VinDr-Mammo/finding_annotations.csv \
        --img_dir /scratch/harisa_iitp/data/vindr/slices_512/img \
        --mask_dir /scratch/harisa_iitp/data/vindr/slices_512/mask \
        --sam_checkpoint /scratch/harisa_iitp/sam_vit_h_4b8939.pth
"""
import argparse
import ast
import csv
import os

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def parse_categories(raw: str) -> list:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", default="VinDr-Mammo/finding_annotations.csv")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_h")
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--split", default="training")
    args = parser.parse_args()

    os.makedirs(args.mask_dir, exist_ok=True)
    out_w, out_h = args.img_size

    # Group Mass boxes by image_id so multiple masses on one image get unioned.
    boxes_by_image = {}
    with open(args.annotations_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != args.split:
                continue
            if "Mass" not in parse_categories(row["finding_categories"]) or not row["xmin"]:
                continue
            sx = out_w / float(row["width"])
            sy = out_h / float(row["height"])
            box = np.array([
                float(row["xmin"]) * sx,
                float(row["ymin"]) * sy,
                float(row["xmax"]) * sx,
                float(row["ymax"]) * sy,
            ])
            boxes_by_image.setdefault(row["image_id"], []).append(box)
    print(f"{sum(len(b) for b in boxes_by_image.values())} Mass boxes "
          f"on {len(boxes_by_image)} images ({args.split} split)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({args.sam_model_type}) on {device}...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    masked_names = set()
    skipped = 0
    for image_id, boxes in sorted(boxes_by_image.items()):
        img_name = f"{image_id}.jpg"
        img_path = os.path.join(args.img_dir, img_name)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(image)

        union_mask = np.zeros((out_h, out_w), dtype=bool)
        for box in boxes:
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            union_mask |= masks[int(np.argmax(scores))]

        Image.fromarray(union_mask.astype(np.uint8) * 255, mode="L").save(
            os.path.join(args.mask_dir, img_name), quality=95
        )
        masked_names.add(img_name)
        if len(masked_names) % 50 == 0:
            print(f"[{len(masked_names)}/{len(boxes_by_image)}] {img_name} <- {len(boxes)} box(es)")

    # Blank masks for converted images without a Mass finding (normals).
    blanks = 0
    blank = Image.fromarray(np.zeros((out_h, out_w), dtype=np.uint8), mode="L")
    for img_name in sorted(os.listdir(args.img_dir)):
        if img_name.lower().endswith(".jpg") and img_name not in masked_names:
            blank.save(os.path.join(args.mask_dir, img_name), quality=95)
            blanks += 1

    print(f"Done. {len(masked_names)} SAM masks, {blanks} blank masks, "
          f"{skipped} annotated images had no converted JPG.")


if __name__ == "__main__":
    main()
