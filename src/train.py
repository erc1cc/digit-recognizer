"""Train the digit recognizer on MNIST.

Usage:
    python src/train.py [--epochs 12] [--batch-size 128]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"

# Standard MNIST channel statistics.
MEAN, STD = 0.1307, 0.3081


def build_loaders(batch_size: int, workers: int, augment: bool = True):
    # Augmentation matters here: MNIST is clean and centered, but real
    # handwriting is rotated, off-center and varies in stroke weight.
    # Without this the model scores ~99% on the test set and noticeably
    # worse on anything drawn by hand.
    aug = [
        transforms.RandomAffine(
            degrees=12,
            translate=(0.12, 0.12),
            scale=(0.85, 1.15),
            shear=8,
            fill=0,
        ),
    ]
    post = [transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), value=0)]
    train_tf = transforms.Compose(
        (aug if augment else [])
        + [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
        + (post if augment else [])
    )
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])

    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False, persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true",
                        help="train on unmodified MNIST (for ablation)")
    parser.add_argument("--out", default="digit_cnn.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    train_loader, test_loader = build_loaders(
        args.batch_size, args.workers, augment=not args.no_augment)
    print(f"augmentation: {'off' if args.no_augment else 'on'}")
    print(f"train: {len(train_loader.dataset):,} images | "
          f"test: {len(test_loader.dataset):,} images")

    model = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.25,
    )

    OUT_DIR.mkdir(exist_ok=True)
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * targets.size(0)
            seen += targets.size(0)

        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(),
                 "accuracy": val_acc,
                 "mean": MEAN, "std": STD},
                OUT_DIR / args.out,
            )
            flag = "  <- saved"

        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"train_loss {running / seen:.4f}  "
              f"test_loss {val_loss:.4f}  "
              f"test_acc {val_acc * 100:.2f}%{flag}")

    mins = (time.time() - start) / 60
    print(f"\ndone in {mins:.1f} min | best test accuracy {best_acc * 100:.2f}%")
    print(f"saved to {OUT_DIR / args.out}")


if __name__ == "__main__":
    main()
