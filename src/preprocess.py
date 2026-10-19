"""Turn an arbitrary handwriting image into an MNIST-style 28x28 tensor.

This is the step most home-grown digit recognizers get wrong. MNIST digits
are not just "28x28 grayscale" -- they were built by size-normalizing each
digit into a 20x20 box and then centering it in a 28x28 field by its center
of mass. A model trained on that distribution does badly on a raw resized
photo, so we reproduce the same normalization here.
"""

import numpy as np
from PIL import Image, ImageOps

MEAN, STD = 0.1307, 0.3081


def _center_of_mass(a: np.ndarray) -> tuple[float, float]:
    total = a.sum()
    if total == 0:
        return (a.shape[1] - 1) / 2, (a.shape[0] - 1) / 2
    ys, xs = np.indices(a.shape)
    return float((xs * a).sum() / total), float((ys * a).sum() / total)


def image_to_mnist(img: Image.Image, invert: bool | None = None) -> np.ndarray:
    """Return a float32 array of shape (28, 28), ink = high values, unnormalized [0, 1]."""
    gray = img.convert("L")

    # MNIST is white ink on a black field. Photos and scans are usually the
    # opposite, so flip when the border of the image is light.
    a = np.asarray(gray, dtype=np.float32)
    if invert is None:
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        invert = border.mean() > 127
    if invert:
        gray = ImageOps.invert(gray)
        a = np.asarray(gray, dtype=np.float32)

    # Drop faint background noise so the bounding box tracks real ink.
    a = np.where(a < 40, 0.0, a)
    rows = np.where(a.sum(axis=1) > 0)[0]
    cols = np.where(a.sum(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((28, 28), dtype=np.float32)  # blank image
    a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Scale the longest side to 20px, preserving the aspect ratio.
    h, w = a.shape
    if h > w:
        new_h, new_w = 20, max(1, round(w * 20 / h))
    else:
        new_w, new_h = 20, max(1, round(h * 20 / w))
    digit = Image.fromarray(a.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    digit = np.asarray(digit, dtype=np.float32)

    # Place it in a 28x28 field so its center of mass sits at the center.
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit

    cx, cy = _center_of_mass(canvas)
    shift_x, shift_y = int(round(13.5 - cx)), int(round(13.5 - cy))
    canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))
    if shift_y > 0:
        canvas[:shift_y] = 0
    elif shift_y < 0:
        canvas[shift_y:] = 0
    if shift_x > 0:
        canvas[:, :shift_x] = 0
    elif shift_x < 0:
        canvas[:, shift_x:] = 0

    return canvas / 255.0


def normalize(arr: np.ndarray) -> np.ndarray:
    """Apply the MNIST channel statistics the model was trained with."""
    return (arr - MEAN) / STD


def as_ascii(arr: np.ndarray) -> str:
    """Render a 28x28 array as text, for sanity-checking preprocessing."""
    ramp = " .:-=+*#%@"
    return "\n".join(
        "".join(ramp[min(int(v * len(ramp)), len(ramp) - 1)] for v in row)
        for row in arr
    )
