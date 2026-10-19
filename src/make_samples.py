"""Render sample digit images that mimic real-world photos, for testing.

Takes MNIST test digits and makes them *unlike* MNIST on purpose: dark ink on
white paper, off-center, rotated, at an arbitrary resolution. If predict.py
handles these, the preprocessing pipeline is doing its job.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"


def main():
    random.seed(7)
    ds = datasets.MNIST(ROOT / "data", train=False, download=True)
    OUT.mkdir(exist_ok=True)

    # One example of each digit, first occurrence in the test set.
    picked: dict[int, Image.Image] = {}
    for img, label in ds:
        if label not in picked:
            picked[label] = img
        if len(picked) == 10:
            break

    for digit, img in sorted(picked.items()):
        big = img.resize((180, 180), Image.LANCZOS)
        big = big.rotate(random.uniform(-14, 14), resample=Image.BICUBIC, fillcolor=0)
        big = ImageOps.invert(big)                      # dark ink on white paper

        # Paste off-center onto a larger sheet, like a cropped photo would be.
        sheet = Image.new("L", (400, 320), 255)
        sheet.paste(big, (random.randint(10, 200), random.randint(10, 120)))

        # A little sensor noise so it isn't a clean synthetic image.
        arr = np.asarray(sheet, dtype=np.int16)
        arr = np.clip(arr + np.random.normal(0, 6, arr.shape), 0, 255)
        Image.fromarray(arr.astype(np.uint8)).save(OUT / f"{digit}.png")

    print(f"wrote {len(picked)} images to {OUT}")


if __name__ == "__main__":
    main()
