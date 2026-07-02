"""
Patch config.py fields in place for the SLURM job chain (avoids brittle bash `sed`
against config.py's multi-line os.path.join(...) assignments).

Usage:
    python3 preprocess/patch_config.py --train_model_type x
    python3 preprocess/patch_config.py --train_model_type y --x_model_path /path/to/best_epoch120_loss0.2651_x_mask_y_img_x.pth
"""
import argparse
import re

CONFIG_PATH = "config.py"


def patch_train_model_type(text: str, value: str) -> str:
    pattern = re.compile(r"(\n\s*train_model_type = ).+")
    new_text, n = pattern.subn(rf"\g<1>'{value}'", text, count=1)
    if n != 1:
        raise SystemExit("Could not find train_model_type assignment in config.py")
    return new_text


def patch_x_model_path(text: str, value: str) -> str:
    # Matches the first x_model_path assignment (Train.x_model_path), which spans
    # two lines: `x_model_path = os.path.join(...)  # <comment>`.
    pattern = re.compile(r"(\n\s*x_model_path = )os\.path\.join\(.*?\)(  # .*)?", re.DOTALL)
    new_text, n = pattern.subn(rf"\g<1>'{value}'", text, count=1)
    if n != 1:
        raise SystemExit("Could not find Train.x_model_path assignment in config.py")
    return new_text


def patch_n_epochs(text: str, value: int) -> str:
    pattern = re.compile(r"(\n\s*n_epochs = )\d+")
    new_text, n = pattern.subn(rf"\g<1>{value}", text, count=1)
    if n != 1:
        raise SystemExit("Could not find Train.n_epochs assignment in config.py")
    return new_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_model_type", choices=["x", "y"], default=None)
    parser.add_argument("--x_model_path", default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--config_path", default=CONFIG_PATH)
    args = parser.parse_args()

    with open(args.config_path) as f:
        text = f.read()

    if args.train_model_type is not None:
        text = patch_train_model_type(text, args.train_model_type)
        print(f"Set train_model_type = '{args.train_model_type}'")

    if args.x_model_path is not None:
        text = patch_x_model_path(text, args.x_model_path)
        print(f"Set Train.x_model_path = '{args.x_model_path}'")

    if args.n_epochs is not None:
        text = patch_n_epochs(text, args.n_epochs)
        print(f"Set Train.n_epochs = {args.n_epochs}")

    with open(args.config_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
