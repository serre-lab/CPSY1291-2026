(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ContrastiveViewsCore = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function createPRNG(seed) {
    let s = (seed >>> 0) || 1;
    return function rand() {
      s |= 0;
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randn(rand) {
    let u = 0;
    let v = 0;
    while (u <= 1e-12) u = rand();
    while (v <= 1e-12) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }

  function rgbToHsv(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    const s = max === 0 ? 0 : d / max;
    const v = max;
    if (d !== 0) {
      if (max === r) {
        h = ((g - b) / d) % 6;
      } else if (max === g) {
        h = (b - r) / d + 2;
      } else {
        h = (r - g) / d + 4;
      }
      h /= 6;
      if (h < 0) h += 1;
    }
    return [h, s, v];
  }

  function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
      case 0:
        return [v, t, p];
      case 1:
        return [q, v, p];
      case 2:
        return [p, v, t];
      case 3:
        return [p, q, v];
      case 4:
        return [t, p, v];
      default:
        return [v, p, q];
    }
  }

  function relu(x) {
    return x > 0 ? x : 0;
  }

  function buildDataset(imageSize, exemplarsPerClass, seed) {
    const rand = createPRNG(seed);
    const colors = [
      [0.88, 0.16, 0.22],
      [0.14, 0.65, 0.30],
      [0.20, 0.34, 0.88],
      [0.90, 0.74, 0.18],
    ];
    const shapeCount = 4;
    const colorCount = colors.length;
    const pixelCount = imageSize * imageSize;
    const inputDim = pixelCount * 3;
    const total = shapeCount * colorCount * exemplarsPerClass;
    const images = new Array(total);
    const shapeLabels = new Int32Array(total);
    const colorLabels = new Int32Array(total);
    let idx = 0;

    function paintShape(shapeId, cx, cy, size, x, y) {
      const dx = x - cx;
      const dy = y - cy;
      if (shapeId === 0) {
        return dx * dx + dy * dy <= size * size;
      }
      if (shapeId === 1) {
        return Math.abs(dx) <= size && Math.abs(dy) <= size;
      }
      if (shapeId === 2) {
        const ny = (dy + size) / (2 * size);
        if (ny < 0 || ny > 1) return false;
        const width = (1 - ny) * size * 1.8;
        return Math.abs(dx) <= width && dy >= -size && dy <= size;
      }
      const thick = Math.max(1.05, size * 0.36);
      const arm = size * 1.15;
      const bar = (Math.abs(dx) <= thick && Math.abs(dy) <= arm) || (Math.abs(dy) <= thick && Math.abs(dx) <= arm);
      return bar;
    }

    for (let s = 0; s < shapeCount; s++) {
      for (let c = 0; c < colorCount; c++) {
        for (let e = 0; e < exemplarsPerClass; e++) {
          const img = new Float32Array(inputDim);
          const cx = imageSize * 0.5 + randn(rand) * 0.9;
          const cy = imageSize * 0.5 + randn(rand) * 0.9;
          const size = 3.2 + rand() * 2.1;
          const hueScale = 0.85 + 0.3 * rand();
          for (let y = 0; y < imageSize; y++) {
            for (let x = 0; x < imageSize; x++) {
              const k = (y * imageSize + x) * 3;
              const base = 0.04 + rand() * 0.02;
              let r = base;
              let g = base;
              let b = base;
              if (paintShape(s, cx, cy, size, x + 0.5, y + 0.5)) {
                const shade = 0.68 + 0.32 * rand();
                r += colors[c][0] * shade * hueScale;
                g += colors[c][1] * shade * hueScale;
                b += colors[c][2] * shade * hueScale;
              }
              img[k] = clamp01(r);
              img[k + 1] = clamp01(g);
              img[k + 2] = clamp01(b);
            }
          }
          images[idx] = img;
          shapeLabels[idx] = s;
          colorLabels[idx] = c;
          idx++;
        }
      }
    }

    return { images, shapeLabels, colorLabels, total, imageSize, inputDim, shapeCount, colorCount };
  }

  function shiftAndCrop(src, dst, imageSize, rand) {
    const crop = 13 + Math.floor(rand() * 4);
    const x0 = Math.floor(rand() * (imageSize - crop + 1));
    const y0 = Math.floor(rand() * (imageSize - crop + 1));
    const sx = Math.floor(rand() * 5) - 2;
    const sy = Math.floor(rand() * 5) - 2;
    for (let y = 0; y < imageSize; y++) {
      const yy = Math.min(crop - 1, Math.max(0, Math.floor((y * crop) / imageSize)));
      for (let x = 0; x < imageSize; x++) {
        const xx = Math.min(crop - 1, Math.max(0, Math.floor((x * crop) / imageSize)));
        const ox = x0 + xx + sx;
        const oy = y0 + yy + sy;
        const d = (y * imageSize + x) * 3;
        if (ox < 0 || oy < 0 || ox >= imageSize || oy >= imageSize) {
          dst[d] = 0.03;
          dst[d + 1] = 0.03;
          dst[d + 2] = 0.03;
        } else {
          const s = (oy * imageSize + ox) * 3;
          dst[d] = src[s];
          dst[d + 1] = src[s + 1];
          dst[d + 2] = src[s + 2];
        }
      }
    }
  }

  function colorJitter(src, dst, imageSize, rand) {
    const hueShift = rand() * 1.0;
    const satScale = 0.65 + 0.9 * rand();
    const valScale = 0.65 + 0.7 * rand();
    const jitter = 0.03;
    const n = imageSize * imageSize * 3;
    for (let i = 0; i < n; i += 3) {
      const hsv = rgbToHsv(src[i], src[i + 1], src[i + 2]);
      let h = hsv[0] + hueShift;
      h -= Math.floor(h);
      let s = hsv[1] * satScale;
      let v = hsv[2] * valScale;
      s = s > 1 ? 1 : s;
      v = v > 1 ? 1 : v;
      const rgb = hsvToRgb(h, s, v);
      dst[i] = clamp01(rgb[0] + (rand() - 0.5) * jitter);
      dst[i + 1] = clamp01(rgb[1] + (rand() - 0.5) * jitter);
      dst[i + 2] = clamp01(rgb[2] + (rand() - 0.5) * jitter);
    }
  }

  function blur3x3(src, dst, imageSize) {
    const tmp = new Float32Array(src.length);
    const w = imageSize;
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < 3; c++) {
          const i = (y * w + x) * 3 + c;
          const i0 = (y * w + Math.max(0, x - 1)) * 3 + c;
          const i1 = i;
          const i2 = (y * w + Math.min(w - 1, x + 1)) * 3 + c;
          tmp[i] = (src[i0] + 2 * src[i1] + src[i2]) * 0.25;
        }
      }
    }
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < 3; c++) {
          const i = (y * w + x) * 3 + c;
          const i0 = (Math.max(0, y - 1) * w + x) * 3 + c;
          const i1 = i;
          const i2 = (Math.min(w - 1, y + 1) * w + x) * 3 + c;
          dst[i] = (tmp[i0] + 2 * tmp[i1] + tmp[i2]) * 0.25;
        }
      }
    }
  }

  function createContrastiveViewsCore(options) {
    const opt = Object.assign(
      {
        seed: 7,
        imageSize: 16,
        exemplarsPerClass: 10,
        hiddenSize: 64,
        batchSize: 24,
        learningRate: 0.008,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        temperature: 0.20,
        repulsion: true,
        augment: { shift: true, colorJitter: true, blur: false },
      },
      options || {}
    );

    const dataset = buildDataset(opt.imageSize, opt.exemplarsPerClass, opt.seed ^ 0x7117a1);
    const inputDim = dataset.inputDim;
    const hiddenSize = opt.hiddenSize;
    const outputDim = 2;
    const rand = createPRNG(opt.seed ^ 0x13579bdf);
    const augmentRand = createPRNG(opt.seed ^ 0x2468ace0);

    let W1 = new Float32Array(hiddenSize * inputDim);
    let b1 = new Float32Array(hiddenSize);
    let W2 = new Float32Array(outputDim * hiddenSize);
    let b2 = new Float32Array(outputDim);

    let mW1 = new Float32Array(W1.length);
    let vW1 = new Float32Array(W1.length);
    let mb1 = new Float32Array(b1.length);
    let vb1 = new Float32Array(b1.length);
    let mW2 = new Float32Array(W2.length);
    let vW2 = new Float32Array(W2.length);
    let mb2 = new Float32Array(b2.length);
    let vb2 = new Float32Array(b2.length);

    let step = 0;
    let lastLoss = 0;
    let temperature = opt.temperature;
    let repulsion = !!opt.repulsion;
    const augment = {
      shift: !!opt.augment.shift,
      colorJitter: !!opt.augment.colorJitter,
      blur: !!opt.augment.blur,
    };

    function initWeights() {
      const s1 = Math.sqrt(2 / inputDim);
      const s2 = Math.sqrt(2 / hiddenSize);
      for (let i = 0; i < W1.length; i++) W1[i] = s1 * randn(rand);
      for (let i = 0; i < b1.length; i++) b1[i] = 0;
      for (let i = 0; i < W2.length; i++) W2[i] = s2 * randn(rand);
      for (let i = 0; i < b2.length; i++) b2[i] = 0;
      mW1.fill(0);
      vW1.fill(0);
      mb1.fill(0);
      vb1.fill(0);
      mW2.fill(0);
      vW2.fill(0);
      mb2.fill(0);
      vb2.fill(0);
      step = 0;
      lastLoss = 0;
    }

    function setAugmentations(next) {
      if (next && typeof next === "object") {
        if (typeof next.shift === "boolean") augment.shift = next.shift;
        if (typeof next.colorJitter === "boolean") augment.colorJitter = next.colorJitter;
        if (typeof next.blur === "boolean") augment.blur = next.blur;
      }
    }

    function setRepulsion(flag) {
      repulsion = !!flag;
    }

    function setTemperature(t) {
      temperature = Math.max(0.03, Math.min(1.2, +t || 0.2));
    }

    function augmentImage(src, out, scratch) {
      let cur = src;
      let dst = out;
      if (augment.shift) {
        shiftAndCrop(cur, dst, dataset.imageSize, augmentRand);
        cur = dst;
        dst = scratch;
      } else {
        dst.set(cur);
        cur = dst;
        dst = scratch;
      }
      if (augment.colorJitter) {
        colorJitter(cur, dst, dataset.imageSize, augmentRand);
        cur = dst;
        dst = cur === out ? scratch : out;
      }
      if (augment.blur) {
        blur3x3(cur, dst, dataset.imageSize);
        cur = dst;
      }
      if (cur !== out) out.set(cur);
    }

    function forwardBatch(inputs, M, cache) {
      const hPre = cache.hPre;
      const hAct = cache.hAct;
      const yRaw = cache.yRaw;
      const zNorm = cache.zNorm;
      const yNorm = cache.yNorm;
      for (let i = 0; i < M; i++) {
        const xoff = i * inputDim;
        const hoff = i * hiddenSize;
        for (let h = 0; h < hiddenSize; h++) {
          let s = b1[h];
          const woff = h * inputDim;
          for (let k = 0; k < inputDim; k++) s += W1[woff + k] * inputs[xoff + k];
          hPre[hoff + h] = s;
          hAct[hoff + h] = relu(s);
        }
        const yoff = i * outputDim;
        for (let o = 0; o < outputDim; o++) {
          let s = b2[o];
          const woff = o * hiddenSize;
          for (let h = 0; h < hiddenSize; h++) s += W2[woff + h] * hAct[hoff + h];
          yRaw[yoff + o] = s;
        }
        const a = yRaw[yoff];
        const b = yRaw[yoff + 1];
        const n = Math.sqrt(a * a + b * b) + 1e-8;
        yNorm[i] = n;
        zNorm[yoff] = a / n;
        zNorm[yoff + 1] = b / n;
      }
    }

    function computeLossAndGrad(zNorm, M, batchSize, dZ, work) {
      dZ.fill(0);
      if (!repulsion) {
        let loss = 0;
        for (let i = 0; i < batchSize; i++) {
          const a0 = zNorm[2 * i];
          const a1 = zNorm[2 * i + 1];
          const b0 = zNorm[2 * (i + batchSize)];
          const b1 = zNorm[2 * (i + batchSize) + 1];
          const dot = a0 * b0 + a1 * b1;
          loss += 1 - dot;
          const g = -1 / batchSize;
          dZ[2 * i] += g * b0;
          dZ[2 * i + 1] += g * b1;
          dZ[2 * (i + batchSize)] += g * a0;
          dZ[2 * (i + batchSize) + 1] += g * a1;
        }
        return loss / batchSize;
      }

      let loss = 0;
      const temp = temperature;
      const logits = work.logits;
      const probs = work.probs;
      for (let i = 0; i < M; i++) {
        const pos = i < batchSize ? i + batchSize : i - batchSize;
        const zi0 = zNorm[2 * i];
        const zi1 = zNorm[2 * i + 1];
        let maxLogit = -1e9;
        for (let k = 0; k < M; k++) {
          if (k === i) {
            logits[k] = -1e9;
            continue;
          }
          const dot = zi0 * zNorm[2 * k] + zi1 * zNorm[2 * k + 1];
          const l = dot / temp;
          logits[k] = l;
          if (l > maxLogit) maxLogit = l;
        }
        let den = 0;
        for (let k = 0; k < M; k++) {
          if (k === i) {
            probs[k] = 0;
            continue;
          }
          const ex = Math.exp(logits[k] - maxLogit);
          probs[k] = ex;
          den += ex;
        }
        const logDen = Math.log(den) + maxLogit;
        loss += -(logits[pos] - logDen);
        for (let k = 0; k < M; k++) {
          if (k === i) continue;
          let g = probs[k] / den;
          if (k === pos) g -= 1;
          g /= M;
          const zk0 = zNorm[2 * k];
          const zk1 = zNorm[2 * k + 1];
          dZ[2 * i] += (g / temp) * zk0;
          dZ[2 * i + 1] += (g / temp) * zk1;
          dZ[2 * k] += (g / temp) * zi0;
          dZ[2 * k + 1] += (g / temp) * zi1;
        }
      }
      return loss / M;
    }

    function backwardAndUpdate(inputs, M, cache, dZ, work) {
      const hPre = cache.hPre;
      const hAct = cache.hAct;
      const yNorm = cache.yNorm;
      const zNorm = cache.zNorm;
      const yRaw = cache.yRaw;

      const gW1 = work.gW1;
      const gb1 = work.gb1;
      const gW2 = work.gW2;
      const gb2 = work.gb2;
      gW1.fill(0);
      gb1.fill(0);
      gW2.fill(0);
      gb2.fill(0);

      const dY = work.dY;
      const dH = work.dH;
      dY.fill(0);
      dH.fill(0);
      for (let i = 0; i < M; i++) {
        const yoff = i * outputDim;
        const gy0 = dZ[yoff];
        const gy1 = dZ[yoff + 1];
        const z0 = zNorm[yoff];
        const z1 = zNorm[yoff + 1];
        const dot = gy0 * z0 + gy1 * z1;
        const inv = 1 / yNorm[i];
        dY[yoff] = (gy0 - dot * z0) * inv;
        dY[yoff + 1] = (gy1 - dot * z1) * inv;
      }

      for (let i = 0; i < M; i++) {
        const hoff = i * hiddenSize;
        const yoff = i * outputDim;
        for (let o = 0; o < outputDim; o++) {
          const g = dY[yoff + o];
          gb2[o] += g;
          const woff = o * hiddenSize;
          for (let h = 0; h < hiddenSize; h++) {
            gW2[woff + h] += g * hAct[hoff + h];
            dH[hoff + h] += g * W2[woff + h];
          }
        }
      }

      for (let i = 0; i < M; i++) {
        const xoff = i * inputDim;
        const hoff = i * hiddenSize;
        for (let h = 0; h < hiddenSize; h++) {
          let g = dH[hoff + h];
          if (hPre[hoff + h] <= 0) g = 0;
          gb1[h] += g;
          const woff = h * inputDim;
          for (let k = 0; k < inputDim; k++) gW1[woff + k] += g * inputs[xoff + k];
        }
      }

      const scale = 1 / M;
      const lr = opt.learningRate;
      const b1c = opt.beta1;
      const b2c = opt.beta2;
      const eps = opt.eps;
      step += 1;
      const b1Fix = 1 - Math.pow(b1c, step);
      const b2Fix = 1 - Math.pow(b2c, step);

      for (let i = 0; i < W1.length; i++) {
        const g = gW1[i] * scale;
        mW1[i] = b1c * mW1[i] + (1 - b1c) * g;
        vW1[i] = b2c * vW1[i] + (1 - b2c) * g * g;
        const mh = mW1[i] / b1Fix;
        const vh = vW1[i] / b2Fix;
        W1[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
      }
      for (let i = 0; i < b1.length; i++) {
        const g = gb1[i] * scale;
        mb1[i] = b1c * mb1[i] + (1 - b1c) * g;
        vb1[i] = b2c * vb1[i] + (1 - b2c) * g * g;
        const mh = mb1[i] / b1Fix;
        const vh = vb1[i] / b2Fix;
        b1[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
      }
      for (let i = 0; i < W2.length; i++) {
        const g = gW2[i] * scale;
        mW2[i] = b1c * mW2[i] + (1 - b1c) * g;
        vW2[i] = b2c * vW2[i] + (1 - b2c) * g * g;
        const mh = mW2[i] / b1Fix;
        const vh = vW2[i] / b2Fix;
        W2[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
      }
      for (let i = 0; i < b2.length; i++) {
        const g = gb2[i] * scale;
        mb2[i] = b1c * mb2[i] + (1 - b1c) * g;
        vb2[i] = b2c * vb2[i] + (1 - b2c) * g * g;
        const mh = mb2[i] / b1Fix;
        const vh = vb2[i] / b2Fix;
        b2[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
      }
    }

    function sampleBatchIndices(n, k, out) {
      for (let i = 0; i < n; i++) out[i] = i;
      for (let i = 0; i < k; i++) {
        const j = i + Math.floor(augmentRand() * (n - i));
        const t = out[i];
        out[i] = out[j];
        out[j] = t;
      }
      return out;
    }

    function forwardSingleClean(img, out2) {
      const h = work.hClean;
      for (let j = 0; j < hiddenSize; j++) {
        let s = b1[j];
        const woff = j * inputDim;
        for (let k = 0; k < inputDim; k++) s += W1[woff + k] * img[k];
        h[j] = relu(s);
      }
      let y0 = b2[0];
      let y1 = b2[1];
      for (let j = 0; j < hiddenSize; j++) {
        y0 += W2[j] * h[j];
        y1 += W2[hiddenSize + j] * h[j];
      }
      const n = Math.sqrt(y0 * y0 + y1 * y1) + 1e-8;
      out2[0] = y0 / n;
      out2[1] = y1 / n;
    }

    function getEmbeddings() {
      const n = dataset.total;
      const emb = new Float32Array(n * 2);
      const tmp = work.embTmp;
      for (let i = 0; i < n; i++) {
        forwardSingleClean(dataset.images[i], tmp);
        emb[2 * i] = tmp[0];
        emb[2 * i + 1] = tmp[1];
      }
      return emb;
    }

    function computeMetrics() {
      const n = dataset.total;
      const emb = getEmbeddings();
      let shapeCorrect = 0;
      let colorCorrect = 0;
      for (let i = 0; i < n; i++) {
        const x0 = emb[2 * i];
        const x1 = emb[2 * i + 1];
        let best = -1;
        let bestSim = -1e9;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const sim = x0 * emb[2 * j] + x1 * emb[2 * j + 1];
          if (sim > bestSim) {
            bestSim = sim;
            best = j;
          }
        }
        if (dataset.shapeLabels[i] === dataset.shapeLabels[best]) shapeCorrect++;
        if (dataset.colorLabels[i] === dataset.colorLabels[best]) colorCorrect++;
      }
      let spread = 0;
      let pairs = 0;
      for (let i = 0; i < n; i++) {
        const xi0 = emb[2 * i];
        const xi1 = emb[2 * i + 1];
        for (let j = i + 1; j < n; j++) {
          const dx = xi0 - emb[2 * j];
          const dy = xi1 - emb[2 * j + 1];
          spread += Math.sqrt(dx * dx + dy * dy);
          pairs++;
        }
      }
      return {
        shapeAccuracy: shapeCorrect / n,
        colorAccuracy: colorCorrect / n,
        spread: spread / Math.max(1, pairs),
        step,
        loss: lastLoss,
      };
    }

    const maxBatch = Math.min(opt.batchSize, dataset.total);
    const maxM = maxBatch * 2;
    const work = {
      inputBatch: new Float32Array(maxM * inputDim),
      batchIdx: new Int32Array(dataset.total),
      viewA: new Float32Array(inputDim),
      viewB: new Float32Array(inputDim),
      scratch: new Float32Array(inputDim),
      cache: {
        hPre: new Float32Array(maxM * hiddenSize),
        hAct: new Float32Array(maxM * hiddenSize),
        yRaw: new Float32Array(maxM * outputDim),
        zNorm: new Float32Array(maxM * outputDim),
        yNorm: new Float32Array(maxM),
      },
      dZ: new Float32Array(maxM * outputDim),
      gW1: new Float32Array(W1.length),
      gb1: new Float32Array(b1.length),
      gW2: new Float32Array(W2.length),
      gb2: new Float32Array(b2.length),
      dY: new Float32Array(maxM * outputDim),
      dH: new Float32Array(maxM * hiddenSize),
      embTmp: [0, 0],
      hClean: new Float32Array(hiddenSize),
      logits: new Float32Array(maxM),
      probs: new Float32Array(maxM),
    };

    function trainStep() {
      const batchSize = Math.min(opt.batchSize, dataset.total);
      const M = batchSize * 2;
      const inputBatch = work.inputBatch;
      const batchIdx = work.batchIdx;
      sampleBatchIndices(dataset.total, batchSize, batchIdx);
      const a = work.viewA;
      const b = work.viewB;
      const scratch = work.scratch;
      for (let i = 0; i < batchSize; i++) {
        const img = dataset.images[batchIdx[i]];
        augmentImage(img, a, scratch);
        augmentImage(img, b, scratch);
        inputBatch.set(a, i * inputDim);
        inputBatch.set(b, (i + batchSize) * inputDim);
      }

      const cache = work.cache;
      forwardBatch(inputBatch, M, cache);
      const dZ = work.dZ;
      lastLoss = computeLossAndGrad(cache.zNorm, M, batchSize, dZ, work);
      backwardAndUpdate(inputBatch, M, cache, dZ, work);
      return lastLoss;
    }

    function trainSteps(count) {
      for (let i = 0; i < count; i++) trainStep();
      return computeMetrics();
    }

    function reset(nextSeed) {
      void nextSeed;
      initWeights();
    }

    initWeights();

    return {
      dataset,
      get step() {
        return step;
      },
      get loss() {
        return lastLoss;
      },
      get config() {
        return {
          batchSize: opt.batchSize,
          hiddenSize: opt.hiddenSize,
          learningRate: opt.learningRate,
          temperature,
          repulsion,
          augment: { shift: augment.shift, colorJitter: augment.colorJitter, blur: augment.blur },
        };
      },
      setAugmentations,
      setRepulsion,
      setTemperature,
      trainStep,
      trainSteps,
      computeMetrics,
      getEmbeddings,
      reset,
    };
  }

  function runScenario(config) {
    const core = createContrastiveViewsCore(config);
    if (config && config.augment) core.setAugmentations(config.augment);
    if (config && typeof config.repulsion === "boolean") core.setRepulsion(config.repulsion);
    if (config && typeof config.temperature === "number") core.setTemperature(config.temperature);
    const steps = (config && config.steps) || 320;
    core.trainSteps(steps);
    return core.computeMetrics();
  }

  return {
    createPRNG: createPRNG,
    createContrastiveViewsCore: createContrastiveViewsCore,
    runScenario: runScenario,
  };
});
