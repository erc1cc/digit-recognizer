
/* ---------------------------------------------------------------
   Unpack the weight blob. Batch-norm was folded into the preceding
   layer at export time, so only conv / relu / maxpool / matmul remain.
   --------------------------------------------------------------- */
const _raw = Uint8Array.from(atob(MODEL_B64), c => c.charCodeAt(0));
const FLOATS = new Float32Array(_raw.buffer, _raw.byteOffset, _raw.byteLength / 4);
const L = MODEL_META.map(m => ({
  shape: m.shape,
  w: FLOATS.subarray(m.w[0], m.w[0] + m.w[1]),
  b: FLOATS.subarray(m.b[0], m.b[0] + m.b[1]),
}));

/* Zero-pad by one so the convolution needs no per-tap bounds checks. */
function pad1(inp, C, H, W) {
  const PH = H + 2, PW = W + 2;
  const out = new Float32Array(C * PH * PW);
  for (let c = 0; c < C; c++) {
    const si = c * H * W, di = c * PH * PW;
    for (let y = 0; y < H; y++) {
      out.set(inp.subarray(si + y * W, si + y * W + W), di + (y + 1) * PW + 1);
    }
  }
  return out;
}

/* 3x3 stride-1 same-padding convolution, fused ReLU, unrolled kernel. */
function convRelu(inp, Cin, H, W, layer) {
  const Cout = layer.shape[0], w = layer.w, b = layer.b;
  const p = pad1(inp, Cin, H, W);
  const PW = W + 2, PH = H + 2, HW = H * W;
  const out = new Float32Array(Cout * HW);

  for (let oc = 0; oc < Cout; oc++) {
    const oo = oc * HW, wo = oc * Cin * 9;
    out.fill(b[oc], oo, oo + HW);

    for (let ic = 0; ic < Cin; ic++) {
      const ib = ic * PH * PW, wi = wo + ic * 9;
      const w0 = w[wi],     w1 = w[wi + 1], w2 = w[wi + 2],
            w3 = w[wi + 3], w4 = w[wi + 4], w5 = w[wi + 5],
            w6 = w[wi + 6], w7 = w[wi + 7], w8 = w[wi + 8];

      for (let y = 0; y < H; y++) {
        const r0 = ib + y * PW, r1 = r0 + PW, r2 = r1 + PW, o = oo + y * W;
        for (let x = 0; x < W; x++) {
          out[o + x] +=
            p[r0 + x] * w0 + p[r0 + x + 1] * w1 + p[r0 + x + 2] * w2 +
            p[r1 + x] * w3 + p[r1 + x + 1] * w4 + p[r1 + x + 2] * w5 +
            p[r2 + x] * w6 + p[r2 + x + 1] * w7 + p[r2 + x + 2] * w8;
        }
      }
    }
    for (let i = oo; i < oo + HW; i++) if (out[i] < 0) out[i] = 0;
  }
  return out;
}

function maxPool2(inp, C, H, W) {
  const oh = H >> 1, ow = W >> 1;
  const out = new Float32Array(C * oh * ow);
  for (let c = 0; c < C; c++) {
    const ib = c * H * W, ob = c * oh * ow;
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        const a = ib + 2 * y * W + 2 * x, d = a + W;
        out[ob + y * ow + x] = Math.max(inp[a], inp[a + 1], inp[d], inp[d + 1]);
      }
    }
  }
  return out;
}

function linear(inp, layer, relu) {
  const Cout = layer.shape[0], Cin = layer.shape[1], w = layer.w, b = layer.b;
  const out = new Float32Array(Cout);
  for (let o = 0; o < Cout; o++) {
    let s = b[o];
    const base = o * Cin;
    for (let i = 0; i < Cin; i++) s += inp[i] * w[base + i];
    out[o] = (relu && s < 0) ? 0 : s;
  }
  return out;
}

function softmax(v) {
  let m = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
  const e = [], n = v.length;
  let sum = 0;
  for (let i = 0; i < n; i++) { const x = Math.exp(v[i] - m); e.push(x); sum += x; }
  for (let i = 0; i < n; i++) e[i] /= sum;
  return e;
}

function predict(x) {
  let t = convRelu(x, 1, 28, 28, L[0]);
  t = convRelu(t, 32, 28, 28, L[1]);
  t = maxPool2(t, 32, 28, 28);
  t = convRelu(t, 32, 14, 14, L[2]);
  t = convRelu(t, 64, 14, 14, L[3]);
  t = maxPool2(t, 64, 14, 14);
  t = linear(t, L[4], true);
  return softmax(linear(t, L[5], false));
}

/* ---------------------------------------------------------------
   MNIST-style normalization: crop to the ink, scale the longest side
   to 20px, center the result in a 28x28 field by center of mass.
   --------------------------------------------------------------- */
const MEAN = 0.1307, STD = 0.3081;

function toMnist(ctx, W, H) {
  const data = ctx.getImageData(0, 0, W, H).data;
  const ink = new Float32Array(W * H);
  let minX = W, minY = H, maxX = -1, maxY = -1;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = data[(y * W + x) * 4 + 3] / 255;   // alpha = stroke coverage
      ink[y * W + x] = v;
      if (v > 0.15) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;

  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const nw = bw >= bh ? 20 : Math.max(1, Math.round(bw * 20 / bh));
  const nh = bh >= bw ? 20 : Math.max(1, Math.round(bh * 20 / bw));
  const small = new Float32Array(nw * nh);

  for (let y = 0; y < nh; y++) {
    const y0 = minY + y * bh / nh, y1 = minY + (y + 1) * bh / nh;
    const sy0 = Math.floor(y0), sy1 = Math.max(Math.ceil(y1), sy0 + 1);
    for (let x = 0; x < nw; x++) {
      const x0 = minX + x * bw / nw, x1 = minX + (x + 1) * bw / nw;
      const sx0 = Math.floor(x0), sx1 = Math.max(Math.ceil(x1), sx0 + 1);
      let sum = 0, n = 0;
      for (let sy = sy0; sy < sy1 && sy < H; sy++) {
        for (let sx = sx0; sx < sx1 && sx < W; sx++) { sum += ink[sy * W + sx]; n++; }
      }
      small[y * nw + x] = n ? sum / n : 0;
    }
  }

  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const v = small[y * nw + x]; mass += v; cx += x * v; cy += y * v;
    }
  }
  cx = mass ? cx / mass : (nw - 1) / 2;
  cy = mass ? cy / mass : (nh - 1) / 2;
  const offX = Math.round(13.5 - cx), offY = Math.round(13.5 - cy);

  const out = new Float32Array(784);
  for (let y = 0; y < nh; y++) {
    const ty = y + offY;
    if (ty < 0 || ty > 27) continue;
    for (let x = 0; x < nw; x++) {
      const tx = x + offX;
      if (tx < 0 || tx > 27) continue;
      out[ty * 28 + tx] = small[y * nw + x];
    }
  }
  return out;
}

/* ---------------------------------------------------------------
   Pad, readout, and the live loop.
   --------------------------------------------------------------- */
const SIZE = 300;
const DPR = Math.min(window.devicePixelRatio || 1, 2);

const pad = document.getElementById('pad');
const ctx = pad.getContext('2d', { willReadFrequently: true });
// Backing store is DPR-scaled; the CSS box is fluid. Drawing coordinates
// stay in a fixed 0..SIZE logical space regardless of either.
pad.width = SIZE * DPR;
pad.height = SIZE * DPR;
ctx.scale(DPR, DPR);
ctx.lineWidth = 24;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#fff';

const thumb = document.getElementById('thumb');
const tctx = thumb.getContext('2d');

const glyphEl = document.getElementById('glyph');
const confEl = document.getElementById('conf');
const noteEl = document.getElementById('note');
const distEl = document.getElementById('dist');

const rows = [];
for (let d = 0; d < 10; d++) {
  const row = document.createElement('div');
  row.className = 'bar';
  row.innerHTML = '<span>' + d + '</span>' +
                  '<span class="track"><span class="fill"></span></span>' +
                  '<span class="val">0.00</span>';
  distEl.appendChild(row);
  rows.push({ row: row, fill: row.querySelector('.fill'), val: row.querySelector('.val') });
}

let drawing = false, dirty = false, hasInk = false;
const timings = [];

function pos(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * SIZE / r.width,
          (e.clientY - r.top) * SIZE / r.height];
}

pad.addEventListener('pointerdown', e => {
  e.preventDefault();
  drawing = true;
  try { pad.setPointerCapture(e.pointerId); } catch (_) {}
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(p[0], p[1]);
  ctx.stroke();
  dirty = true;
  hasInk = true;
});

pad.addEventListener('pointermove', e => {
  if (!drawing) return;
  const p = pos(e);
  ctx.lineTo(p[0], p[1]);
  ctx.stroke();
  dirty = true;
});

window.addEventListener('pointerup', () => { drawing = false; });
window.addEventListener('pointercancel', () => { drawing = false; });

document.getElementById('clear').addEventListener('click', () => {
  ctx.clearRect(0, 0, SIZE, SIZE);
  tctx.clearRect(0, 0, 28, 28);
  hasInk = false;
  dirty = false;
  glyphEl.textContent = '—';
  glyphEl.className = 'glyph idle';
  confEl.textContent = '—';
  noteEl.textContent = 'Awaiting input';
  noteEl.className = 'note';
  for (let d = 0; d < 10; d++) {
    rows[d].fill.style.width = '0';
    rows[d].val.textContent = '0.00';
    rows[d].row.className = 'bar';
  }
});

function tick() {
  if (!dirty || !hasInk) return;
  dirty = false;

  const arr = toMnist(ctx, pad.width, pad.height);
  if (!arr) return;

  const im = tctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.round(Math.min(arr[i], 1) * 255);
    im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v;
    im.data[i * 4 + 3] = 255;
  }
  tctx.putImageData(im, 0, 0);

  const norm = new Float32Array(784);
  for (let i = 0; i < 784; i++) norm[i] = (arr[i] - MEAN) / STD;

  const t0 = performance.now();
  const probs = predict(norm);
  timings.push(performance.now() - t0);
  if (timings.length > 25) timings.shift();

  let best = 0;
  for (let d = 1; d < 10; d++) if (probs[d] > probs[best]) best = d;
  const unsure = probs[best] < 0.55;

  glyphEl.textContent = best;
  glyphEl.className = 'glyph' + (unsure ? ' unsure' : '');
  confEl.textContent = (probs[best] * 100).toFixed(1) + '%';

  if (unsure) {
    let second = best === 0 ? 1 : 0;
    for (let d = 0; d < 10; d++) if (d !== best && probs[d] > probs[second]) second = d;
    noteEl.textContent = 'Uncertain — could be ' + second;
    noteEl.className = 'note unsure';
  } else {
    noteEl.textContent = 'Confident';
    noteEl.className = 'note';
  }

  for (let d = 0; d < 10; d++) {
    rows[d].fill.style.width = (probs[d] * 100).toFixed(2) + '%';
    rows[d].val.textContent = (probs[d] * 100).toFixed(2);
    rows[d].row.className = 'bar' + (d === best ? ' lead' + (unsure ? ' unsure' : '') : '');
  }

  const sorted = timings.slice().sort((a, b) => a - b);
  document.getElementById('specMs').textContent =
    Math.round(sorted[sorted.length >> 1]) + ' ms';
}

setInterval(tick, 110);

document.getElementById('specAcc').textContent = (MODEL_ACCURACY * 100).toFixed(2) + '%';
