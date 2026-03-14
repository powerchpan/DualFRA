import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FallRiskAssessmentModel
from preprocessing import (
    FallRiskDataset,
    SkeletonAugmentor,
    build_dataset_splits,
    build_windowed_dataset,
    make_synthetic_dataset,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for tug, ftsts, labels in loader:
        tug    = tug.to(device)
        ftsts  = ftsts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(tug, ftsts)
        loss   = model.loss(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        n          += labels.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for tug, ftsts, labels in loader:
        tug    = tug.to(device)
        ftsts  = ftsts.to(device)
        labels = labels.to(device)

        logits = model(tug, ftsts)
        loss   = model.loss(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        n          += labels.size(0)

    return total_loss / n, correct / n



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fall Risk Assessment — Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--data_root", type=str, default=None,
        help="Directory of .npz files. If omitted, synthetic data is used.",
    )
    p.add_argument(
        "--apply_sliding_window", action="store_true",
        help=(
            "If set, --data_root is treated as raw variable-length subject "
            "sequences and the sliding window is applied before splitting. "
            "Each .npz must contain 'tug' [T_raw, V, C], 'ftsts', 'label'."
        ),
    )
    p.add_argument(
        "--num_joints", type=int, default=49,
        help="Number of skeleton joints V (49 for VIBE/SMPL output).",
    )
    p.add_argument(
        "--coord_dim", type=int, default=3,
        help="Joint coordinate dimension C.",
    )
    p.add_argument(
        "--seq_len", type=int, default=400,
        help="Temporal window length T (paper: 400).",
    )
    p.add_argument(
        "--window_stride", type=int, default=1,
        help="Sliding-window stride in frames (paper: 1).",
    )
    p.add_argument(
        "--n_test", type=int, default=100,
        help="Number of samples held out as independent test set (paper: 100).",
    )
    p.add_argument(
        "--train_val_ratio", type=float, default=9.0,
        help="Train-to-val ratio for the non-test portion (paper: 9:1).",
    )
    p.add_argument(
        "--no_augment", action="store_true",
        help="Disable training-set augmentation (augmentation is on by default).",
    )
    p.add_argument(
        "--n_synthetic", type=int, default=300,
        help="Synthetic sample count when --data_root is not supplied.",
    )

    p.add_argument("--noise_std",       type=float, default=0.01,
                   help="Std-dev of Gaussian noise injection.")
    p.add_argument("--rotate_prob",     type=float, default=0.5,
                   help="Probability of applying random spatial rotation.")
    p.add_argument("--joint_drop_prob", type=float, default=0.1,
                   help="Per-joint zeroing probability.")
    p.add_argument("--coord_mask_prob", type=float, default=0.1,
                   help="Per-axis coordinate masking probability.")

    p.add_argument(
        "--channel_schedule", type=int, nargs="+",
        default=[48, 48, 48, 96, 96],
        help="Per-layer output channels (paper: [48, 48, 48, 96, 96]).",
    )
    p.add_argument("--mlp_hidden",  type=int,   default=128)
    p.add_argument("--mlp_out",     type=int,   default=64)
    p.add_argument("--attn_dim",    type=int,   default=64)
    p.add_argument("--cls_hidden",  type=int,   default=256)
    p.add_argument("--num_classes", type=int,   default=3)
    p.add_argument("--dropout",     type=float, default=0.3)

    p.add_argument("--epochs",     type=int,   default=100,
                   help="Training epochs (paper: 100).")
    p.add_argument("--batch_size", type=int,   default=16,
                   help="Mini-batch size (paper: 16).")
    p.add_argument("--lr",         type=float, default=0.01,
                   help="Initial learning rate (paper: 0.01).")
    p.add_argument("--lr_gamma",   type=float, default=0.9,
                   help="Exponential LR decay factor per epoch (paper: 0.9).")
    p.add_argument("--momentum",   type=float, default=0.9,
                   help="Adam β₁ (paper: 0.9).")

    # ── Misc ──────────────────────────────────────────────────────────────
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument(
        "--save_path", type=str, default="best_model.pt",
        help="File path for the best-validation-accuracy checkpoint.",
    )

    return p.parse_args()



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  Fall Risk Assessment — Training")
    print("=" * 65)
    print(f"  Device          : {device}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Initial LR      : {args.lr}  (ExponentialLR γ={args.lr_gamma})")
    print(f"  Backbone        : {len(args.channel_schedule)} layers, "
          f"channels {args.channel_schedule}")
    print(f"  Window length T : {args.seq_len}")
    print(f"  Test set size   : {args.n_test}")
    print(f"  Train:val split : {int(args.train_val_ratio)}:1")
    print(f"  Augmentation    : {'OFF' if args.no_augment else 'ON'}")
    print("=" * 65)

    if args.data_root:
        if args.apply_sliding_window:
            print(f"\nLoading raw subject sequences from: {args.data_root}")
            print(f"  Applying sliding window  (W={args.seq_len}, stride={args.window_stride})")
            from pathlib import Path
            import numpy as _np

            raw_records = []
            for p in sorted(Path(args.data_root).glob("*.npz")):
                d = _np.load(p)
                raw_records.append({
                    "tug":   d["tug"].astype(_np.float32),
                    "ftsts": d["ftsts"].astype(_np.float32),
                    "label": int(d["label"]),
                })
            if not raw_records:
                raise FileNotFoundError(
                    f"No .npz files found in '{args.data_root}'.")
            dataset = build_windowed_dataset(
                subject_records=raw_records,
                window_size=args.seq_len,
                stride=args.window_stride,
                seq_len=args.seq_len,
            )
            print(f"  Windowed samples generated: {len(dataset)}")
        else:
            print(f"\nLoading windowed dataset from: {args.data_root}")
            dataset = FallRiskDataset(
                data_root=args.data_root, seq_len=args.seq_len
            )
    else:
        print(f"\nNo --data_root provided. "
              f"Using {args.n_synthetic} synthetic samples (T={args.seq_len}).")
        dataset = make_synthetic_dataset(
            n_samples=args.n_synthetic,
            num_joints=args.num_joints,
            coord_dim=args.coord_dim,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            seed=args.seed,
        )

    augmentor = SkeletonAugmentor(
        noise_std=args.noise_std,
        rotate_prob=args.rotate_prob,
        joint_drop_prob=args.joint_drop_prob,
        coord_mask_prob=args.coord_mask_prob,
        seed=args.seed,
    )

    train_loader, val_loader, test_loader = build_dataset_splits(
        dataset=dataset,
        n_test=args.n_test,
        train_val_ratio=args.train_val_ratio,
        batch_size=args.batch_size,
        augment_train=not args.no_augment,
        augmentor=augmentor,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"\n  Train samples   : {n_train}  ({len(train_loader)} batches)")
    print(f"  Val   samples   : {n_val}  ({len(val_loader)} batches)")
    print(f"  Test  samples   : {n_test}  ({len(test_loader)} batches)")

    model = FallRiskAssessmentModel(
        num_joints=args.num_joints,
        coord_dim=args.coord_dim,
        channel_schedule=args.channel_schedule,
        mlp_hidden=args.mlp_hidden,
        mlp_out=args.mlp_out,
        attn_dim=args.attn_dim,
        cls_hidden=args.cls_hidden,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {total_params:,}\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.momentum, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_gamma
    )

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"lr={current_lr:.2e}  "
            f"train  loss={tr_loss:.4f}  acc={tr_acc:.3f}  |  "
            f"val  loss={va_loss:.4f}  acc={va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":  va_acc,
                    "val_loss": va_loss,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"  ✓ Best checkpoint saved  (val_acc={va_acc:.3f})")

    
    print(f"\n{'=' * 65}")
    print(f"  Loading best checkpoint from: {args.save_path}")
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    te_loss, te_acc = evaluate(model, test_loader, device)

    print(f"  Best validation accuracy  : {best_val_acc:.4f}")
    print(f"  Test set loss             : {te_loss:.4f}")
    print(f"  Test set accuracy         : {te_acc:.4f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
