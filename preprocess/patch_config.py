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


def patch_path_assignment(text: str, field: str, value: str) -> str:
    # Matches the first `<field> = ...` assignment, whether it is still the
    # original multi-line os.path.join(...) form or a plain quoted string left
    # by a previous run of this script (so re-patching keeps working).
    pattern = re.compile(
        rf"(\n\s*{field} = )(os\.path\.join\(.*?\)|'[^']*'|\"[^\"]*\")(  # [^\n]*)?",
        re.DOTALL,
    )
    new_text, n = pattern.subn(rf"\g<1>'{value}'", text, count=1)
    if n != 1:
        raise SystemExit(f"Could not find {field} assignment in config.py")
    return new_text


def patch_simple(text: str, field: str, value: str) -> str:
    # For single-line assignments (strings/booleans) like data_dir, resume_train,
    # finetune, desc, lr.
    pattern = re.compile(rf"(\n\s*{field} = )[^\n]+")
    new_text, n = pattern.subn(rf"\g<1>{value}", text, count=1)
    if n != 1:
        raise SystemExit(f"Could not find {field} assignment in config.py")
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
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--desc", default=None,
                        help="Literal run description (log dir / checkpoint name suffix)")
    parser.add_argument("--resume_train", choices=["true", "false"], default=None)
    parser.add_argument("--finetune", choices=["true", "false"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config_path", default=CONFIG_PATH)
    args = parser.parse_args()

    with open(args.config_path) as f:
        text = f.read()

    if args.train_model_type is not None:
        text = patch_train_model_type(text, args.train_model_type)
        print(f"Set train_model_type = '{args.train_model_type}'")

    if args.x_model_path is not None:
        text = patch_path_assignment(text, "x_model_path", args.x_model_path)
        print(f"Set Train.x_model_path = '{args.x_model_path}'")

    if args.ckpt_path is not None:
        text = patch_path_assignment(text, "ckpt_path", args.ckpt_path)
        print(f"Set Train.ckpt_path = '{args.ckpt_path}'")

    if args.n_epochs is not None:
        text = patch_n_epochs(text, args.n_epochs)
        print(f"Set Train.n_epochs = {args.n_epochs}")

    if args.data_dir is not None:
        text = patch_simple(text, "data_dir", f"'{args.data_dir}'")
        print(f"Set Data.data_dir = '{args.data_dir}'")

    if args.desc is not None:
        text = patch_simple(text, "desc", f"'{args.desc}'")
        print(f"Set Train.desc = '{args.desc}'")

    if args.resume_train is not None:
        text = patch_simple(text, "resume_train", args.resume_train.capitalize())
        print(f"Set Train.resume_train = {args.resume_train.capitalize()}")

    if args.finetune is not None:
        text = patch_simple(text, "finetune", args.finetune.capitalize())
        print(f"Set Train.finetune = {args.finetune.capitalize()}")

    if args.lr is not None:
        text = patch_simple(text, "lr", str(args.lr))
        print(f"Set Optim.lr = {args.lr}")

    with open(args.config_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
