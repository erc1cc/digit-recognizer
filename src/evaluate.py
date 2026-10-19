"""Per-class accuracy and the most-confused pairs on the MNIST test set."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN
from preprocess import MEAN, STD

ROOT = Path(__file__).resolve().parent.parent


def main():
    ckpt = torch.load(ROOT / "models" / "digit_cnn.pt", map_location="cpu")
    model = DigitCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((MEAN,), (STD,))])
    loader = DataLoader(datasets.MNIST(ROOT / "data", train=False, transform=tf),
                        batch_size=512)

    cm = np.zeros((10, 10), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            for t, p in zip(y.numpy(), pred.numpy()):
                cm[t, p] += 1

    print(f"overall accuracy: {cm.trace() / cm.sum() * 100:.2f}%  "
          f"({cm.sum() - cm.trace()} errors in {cm.sum():,} images)\n")

    print("digit  support  correct  accuracy")
    for d in range(10):
        print(f"  {d}      {cm[d].sum():5d}    {cm[d, d]:5d}    "
              f"{cm[d, d] / cm[d].sum() * 100:6.2f}%")

    errs = [(cm[t, p], t, p) for t in range(10) for p in range(10) if t != p and cm[t, p]]
    errs.sort(reverse=True)
    print("\nmost common confusions:")
    for n, t, p in errs[:6]:
        print(f"  {t} read as {p}: {n} time{'s' if n > 1 else ''}")


if __name__ == "__main__":
    main()
