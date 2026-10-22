# Handwritten digit recognizer

A convolutional neural network that reads handwritten digits (0–9), trained
from scratch on MNIST in PyTorch. Includes a browser demo that runs the
trained network client-side, with no server required.

## Layout

```
src/model.py             CNN architecture (~468k parameters)
src/train.py             training loop -- downloads MNIST, trains, saves best checkpoint
src/evaluate.py          per-class accuracy and the most-confused pairs
src/preprocess.py        turns an arbitrary image into an MNIST-style 28x28 input
src/predict.py           classify image files from the command line
src/make_samples.py      render realistic test images from MNIST digits
src/export_web.py        fold batch-norm, export weights as web/weights.js
src/build_standalone.py  assemble the one-file standalone page
src/verify_export.py     fixtures for checking JS output against PyTorch
src/ablation.py          augmented vs non-augmented, clean vs distorted

web/index.html           minimal dev demo (loads weights.js separately)
web/standalone/          shell.html + runtime.js, sources for the standalone page
web/digit-reader.html    built standalone page -- network embedded, one file

models/digit_cnn.pt      trained checkpoint
samples/                 generated test images
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision numpy pillow
```

## Train

```bash
.venv/bin/python src/train.py --epochs 14
```

Downloads MNIST on first run (~11 MB into `data/`), uses Apple MPS or CUDA when
available and falls back to CPU. The checkpoint with the best test accuracy is
written to `models/digit_cnn.pt`.

The trained run took **2.5 minutes** and reached **99.62%** on the 10,000-image
test set -- 38 errors. `src/evaluate.py` prints the per-class breakdown; the
most common mistake is a 9 read as a 4 (6 times).

## Predict on an image

```bash
.venv/bin/python src/predict.py path/to/digit.png --show
```

Accepts any image: dark ink on light paper or the reverse, any size. `--show`
prints the preprocessed 28×28 input as ASCII so you can see what the model
actually received.

## Browser demo

```bash
.venv/bin/python src/export_web.py && .venv/bin/python src/build_standalone.py
```

`web/digit-reader.html` is then a single self-contained file -- draw with a
mouse or finger and it classifies live. Nothing is sent anywhere; the network
runs in the page. Verified to match PyTorch to within 6e-7 and to classify at
about 45 ms per inference on desktop.

## How it works

**Architecture.** Two convolutional blocks (32 and 64 channels), each with two
3×3 convolutions, batch norm, ReLU, 2×2 max pooling and dropout, then a
128-unit fully connected layer and a 10-way output. Small enough to train on a
laptop CPU, big enough to saturate MNIST.

**Training.** AdamW with a one-cycle learning rate schedule, label smoothing of
0.05, and aggressive augmentation — random rotation, translation, scaling,
shear and small random erasures.

The augmentation costs a little benchmark accuracy and buys a great deal of
robustness. `src/ablation.py` trains the same architecture with augmentation
off and scores both on clean MNIST and on a harsher distorted version of the
test set:

| model | clean MNIST | distorted | drop |
|---|---|---|---|
| augmented | 99.62% | 98.75% | 0.87 pp |
| no augmentation | **99.70%** | 83.48% | 16.22 pp |

The un-augmented model is the better one on the leaderboard and the far worse
one in practice — it fails on roughly one distorted digit in six, against one
in eighty. Reproduce with:

```bash
.venv/bin/python src/train.py --epochs 14 --no-augment --out digit_cnn_noaug.pt
.venv/bin/python src/ablation.py
```

**Preprocessing is the part that usually breaks.** MNIST digits were not simply
resized to 28×28 — each digit was cropped to its bounding box, scaled so its
longest side is 20 pixels, and placed in a 28×28 field centered by its *center
of mass*. Feeding a naively resized image to a model trained on that
distribution gives bad predictions that look like a model bug. `preprocess.py`
and the JavaScript in the demo both reproduce that exact normalization.

**Browser export.** `export_web.py` folds each batch-norm layer into the weights
of the convolution feeding it, which is mathematically equivalent at inference
time and leaves only conv, ReLU, max-pool and matrix multiply to reimplement in
JavaScript. Weights ship as a base64 float32 blob inside the page.

The JavaScript convolution pre-pads its input by one pixel and unrolls the 3x3
kernel, so the inner loop has no bounds checks. That is worth about 11x -- the
straightforward version ran at 484 ms per inference, this one at 45 ms, which
is the difference between a live readout and a page that stutters while you
draw. `verify_export.py` writes fixtures that pin the JavaScript output to
PyTorch's; the two agree to 6e-7, which is float32 rounding.

## Limits

- One digit per image. Reading multi-digit numbers needs a segmentation step
  first (connected components, or a CTC/detection model) — this classifier
  looks at a single centered glyph.
- Trained on MNIST, which is American handwriting from the 1990s. Crossed
  sevens and European-style ones are underrepresented and get less reliable
  predictions.
- Digits only. Letters produce a confident but meaningless answer, since the
  softmax is forced to pick one of ten classes.
