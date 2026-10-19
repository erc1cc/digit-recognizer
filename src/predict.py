"""Predict digits from image files.

Usage:
    python src/predict.py path/to/digit.png [more.png ...] [--show]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import DigitCNN
from preprocess import image_to_mnist, normalize, as_ascii

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "digit_cnn.pt"


def load_model(path: Path = CHECKPOINT, device: str = "cpu") -> DigitCNN:
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run: python src/train.py")
    ckpt = torch.load(path, map_location=device)
    model = DigitCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_array(model: DigitCNN, arr: np.ndarray) -> tuple[int, np.ndarray]:
    """Classify one 28x28 [0,1] array. Returns (digit, probabilities)."""
    x = torch.from_numpy(normalize(arr)).float().view(1, 1, 28, 28)
    probs = F.softmax(model(x), dim=1)[0].numpy()
    return int(probs.argmax()), probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--show", action="store_true",
                        help="print the preprocessed 28x28 image as ASCII")
    args = parser.parse_args()

    model = load_model()

    for path in args.images:
        if not path.exists():
            print(f"{path}: not found")
            continue
        arr = image_to_mnist(Image.open(path))
        digit, probs = predict_array(model, arr)

        if args.show:
            print(as_ascii(arr))

        runner_up = int(np.argsort(probs)[-2])
        print(f"{path.name}: {digit}   "
              f"(confidence {probs[digit] * 100:.1f}%, "
              f"next best {runner_up} at {probs[runner_up] * 100:.1f}%)")


if __name__ == "__main__":
    main()
