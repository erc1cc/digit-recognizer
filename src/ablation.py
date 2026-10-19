"""Does the augmentation actually help on realistic input?

Builds a distorted version of the MNIST test set -- rotated, shifted, scaled,
sheared, the way a digit drawn by a person actually arrives -- and scores the
augmented and non-augmented checkpoints on both the clean and distorted sets.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN
from preprocess import MEAN, STD

ROOT = Path(__file__).resolve().parent.parent

CLEAN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])

# Deliberately harsher than the training augmentation, so neither model has
# seen this exact distribution.
DISTORTED = transforms.Compose([
    transforms.RandomAffine(degrees=18, translate=(0.16, 0.16),
                            scale=(0.75, 1.25), shear=12, fill=0),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])


def load(name):
    path = ROOT / "models" / name
    if not path.exists():
        sys.exit(f"missing checkpoint {path}")
    ckpt = torch.load(path, map_location="cpu")
    m = DigitCNN()
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def score(model, tf, seed):
    torch.manual_seed(seed)
    ds = datasets.MNIST(ROOT / "data", train=False, transform=tf)
    correct = total = 0
    for x, y in DataLoader(ds, batch_size=512):
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    models = {"augmented": load("digit_cnn.pt"),
              "no augmentation": load("digit_cnn_noaug.pt")}

    print(f"{'model':<18}{'clean MNIST':>14}{'distorted':>12}{'drop':>9}")
    print("-" * 53)
    for name, m in models.items():
        clean = score(m, CLEAN, 0)
        dist = score(m, DISTORTED, 1234)   # same seed => same distortions
        print(f"{name:<18}{clean * 100:>13.2f}%{dist * 100:>11.2f}%"
              f"{(clean - dist) * 100:>8.2f}pp")


if __name__ == "__main__":
    main()
