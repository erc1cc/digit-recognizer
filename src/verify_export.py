"""Cross-check: does the exported (BN-folded) model match the PyTorch model?

Writes fixtures/parity.json so the browser demo's JavaScript forward pass can
be compared against PyTorch on the same inputs.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from model import DigitCNN
from preprocess import MEAN, STD

ROOT = Path(__file__).resolve().parent.parent
N = 8


def main():
    ckpt = torch.load(ROOT / "models" / "digit_cnn.pt", map_location="cpu")
    model = DigitCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = datasets.MNIST(ROOT / "data", train=False, download=True)
    cases = []
    for i in range(N):
        img, label = ds[i]
        raw = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy((raw - MEAN) / STD).view(1, 1, 28, 28)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0].numpy()
        cases.append({
            "label": int(label),
            "input": [round(float(v), 6) for v in raw.flatten()],
            "probs": [round(float(p), 6) for p in probs],
            "pred": int(probs.argmax()),
        })

    out = ROOT / "fixtures" / "parity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cases))
    acc = sum(c["pred"] == c["label"] for c in cases)
    print(f"wrote {out} ({N} cases, pytorch got {acc}/{N} right)")


if __name__ == "__main__":
    main()
