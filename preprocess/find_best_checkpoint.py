"""
Locate the best Phase 1 (x-model) checkpoint so train_pig_phase2.sh can patch it
into config.py's Train.x_model_path automatically, without a manual hand-off step
between the two SLURM jobs.

Picks the most recently created log dir matching *_{desc_suffix}, then within it
the checkpoint file "best_epoch*_loss*_{desc_suffix}.pth" with the lowest loss
(multiple best_epoch* files can accumulate since a new one is written each time
loss improves and old ones aren't deleted).

Usage:
    python3 preprocess/find_best_checkpoint.py \
        --log_root /scratch/harisa_iitp/PIG_outputs/log \
        --desc_suffix x_mask_y_img_x_pretrained
Prints the checkpoint path to stdout (nothing else) on success.
"""
import argparse
import glob
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--desc_suffix", required=True)
    args = parser.parse_args()

    run_dirs = sorted(
        glob.glob(os.path.join(args.log_root, f"*_{args.desc_suffix}")),
        key=os.path.getmtime,
    )
    if not run_dirs:
        raise SystemExit(f"No log dir matching *_{args.desc_suffix} under {args.log_root}")
    run_dir = run_dirs[-1]

    pattern = re.compile(rf"best_epoch(\d+)_loss([\d.]+)_{re.escape(args.desc_suffix)}\.pth$")
    best_path, best_loss = None, None
    for fname in os.listdir(run_dir):
        m = pattern.match(fname)
        if not m:
            continue
        loss = float(m.group(2))
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_path = os.path.join(run_dir, fname)

    if best_path is None:
        raise SystemExit(f"No best_epoch*_{args.desc_suffix}.pth checkpoint found in {run_dir}")

    print(best_path)


if __name__ == "__main__":
    main()
