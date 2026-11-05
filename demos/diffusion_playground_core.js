"use strict";

function createRng(seed) {
  let s = (seed >>> 0) || 1;
  return function rng() {
    s |= 0;
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeDdpmLinearSchedule(steps, betaStart, betaEnd) {
  const T = Math.max(2, steps | 0);
  const start = betaStart == null ? 1e-4 : betaStart;
  const end = betaEnd == null ? 0.02 : betaEnd;
  const scale = 1000 / T;
  const betas = new Array(T + 1).fill(0);
  const alphas = new Array(T + 1).fill(1);
  const abar = new Array(T + 1).fill(1);
  for (let t = 1; t <= T; t++) {
    const f = T === 1 ? 0 : (t - 1) / (T - 1);
    const beta = Math.min(0.999, scale * (start + f * (end - start)));
    betas[t] = beta;
    alphas[t] = 1 - beta;
    abar[t] = abar[t - 1] * alphas[t];
  }
  return { T, betas, alphas, abar };
}

function forwardPoint(x0, t, schedule, eps) {
  const ti = Math.max(0, Math.min(schedule.T, t | 0));
  const at = schedule.abar[ti];
  const sData = Math.sqrt(at);
  const sNoise = Math.sqrt(Math.max(0, 1 - at));
  return [sData * x0[0] + sNoise * eps[0], sData * x0[1] + sNoise * eps[1]];
}

function forwardPoints(points, t, schedule, rng) {
  const out = new Array(points.length);
  for (let i = 0; i < points.length; i++) {
    const eps = [randn(rng), randn(rng)];
    out[i] = forwardPoint(points[i], t, schedule, eps);
  }
  return out;
}

function idealPosteriorMeanX0(xt, dataPoints, t, schedule) {
  if (!dataPoints.length) return [0, 0];
  const ti = Math.max(1, Math.min(schedule.T, t | 0));
  const at = schedule.abar[ti];
  const m = Math.sqrt(at);
  const varNoise = Math.max(1e-8, 1 - at);

  let maxLogW = -Infinity;
  const logW = new Array(dataPoints.length);
  for (let i = 0; i < dataPoints.length; i++) {
    const dx = xt[0] - m * dataPoints[i][0];
    const dy = xt[1] - m * dataPoints[i][1];
    const lw = -(dx * dx + dy * dy) / (2 * varNoise);
    logW[i] = lw;
    if (lw > maxLogW) maxLogW = lw;
  }

  let z = 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    const w = Math.exp(logW[i] - maxLogW);
    z += w;
    mx += w * dataPoints[i][0];
    my += w * dataPoints[i][1];
  }
  if (z <= 0) return [0, 0];
  return [mx / z, my / z];
}

function estimateEpsilon(xt, x0hat, t, schedule) {
  const ti = Math.max(1, Math.min(schedule.T, t | 0));
  const at = schedule.abar[ti];
  const sData = Math.sqrt(at);
  const sNoise = Math.sqrt(Math.max(1e-8, 1 - at));
  return [(xt[0] - sData * x0hat[0]) / sNoise, (xt[1] - sData * x0hat[1]) / sNoise];
}

function reverseStepDdpm(xt, t, dataPoints, schedule, rng) {
  const ti = Math.max(1, Math.min(schedule.T, t | 0));
  const beta = schedule.betas[ti];
  const alpha = schedule.alphas[ti];
  const abar = schedule.abar[ti];

  const x0hat = idealPosteriorMeanX0(xt, dataPoints, ti, schedule);
  const epsHat = estimateEpsilon(xt, x0hat, ti, schedule);

  const coeff = beta / Math.sqrt(Math.max(1e-8, 1 - abar));
  const meanX = (xt[0] - coeff * epsHat[0]) / Math.sqrt(alpha);
  const meanY = (xt[1] - coeff * epsHat[1]) / Math.sqrt(alpha);

  if (ti === 1) return [meanX, meanY];

  const sigma = Math.sqrt(beta);
  return [meanX + sigma * randn(rng), meanY + sigma * randn(rng)];
}

function reverseSample(initialParticles, dataPoints, schedule, rng, onStep) {
  let particles = initialParticles.map(function copyPoint(p) {
    return [p[0], p[1]];
  });
  for (let t = schedule.T; t >= 1; t--) {
    particles = particles.map(function stepOne(p) {
      return reverseStepDdpm(p, t, dataPoints, schedule, rng);
    });
    if (onStep) onStep(t - 1, particles);
  }
  return particles;
}

function makeInitialNoise(n, rng) {
  const pts = new Array(n);
  for (let i = 0; i < n; i++) pts[i] = [randn(rng), randn(rng)];
  return pts;
}

function nearestDistance(point, set) {
  let best = Infinity;
  for (let i = 0; i < set.length; i++) {
    const dx = point[0] - set[i][0];
    const dy = point[1] - set[i][1];
    const d = Math.hypot(dx, dy);
    if (d < best) best = d;
  }
  return best;
}

function meanNearestDistance(samples, set) {
  if (!samples.length || !set.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += nearestDistance(samples[i], set);
  return sum / samples.length;
}

function dataExtent(points) {
  if (!points.length) return 1;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  return Math.hypot(maxX - minX, maxY - minY);
}

function pathLength(polyline) {
  if (polyline.length < 2) return 0;
  let L = 0;
  for (let i = 1; i < polyline.length; i++) {
    L += Math.hypot(polyline[i][0] - polyline[i - 1][0], polyline[i][1] - polyline[i - 1][1]);
  }
  return L;
}

function resamplePolyline(polyline, nTarget) {
  if (!polyline.length) return [];
  if (polyline.length === 1) return [polyline[0].slice()];
  const total = pathLength(polyline);
  if (total <= 1e-8) return polyline.slice(0, 1);
  const n = Math.max(2, nTarget | 0);
  const step = total / (n - 1);
  const out = [polyline[0].slice()];
  let segIndex = 1;
  let segStart = polyline[0].slice();
  let segEnd = polyline[1].slice();
  let segLen = Math.hypot(segEnd[0] - segStart[0], segEnd[1] - segStart[1]);
  let distInSeg = 0;
  let acc = step;

  while (out.length < n - 1) {
    const remaining = segLen - distInSeg;
    if (acc <= remaining + 1e-8) {
      const frac = (distInSeg + acc) / segLen;
      out.push([
        segStart[0] + frac * (segEnd[0] - segStart[0]),
        segStart[1] + frac * (segEnd[1] - segStart[1]),
      ]);
      distInSeg += acc;
      acc = step;
    } else {
      acc -= remaining;
      segIndex++;
      if (segIndex >= polyline.length) break;
      segStart = polyline[segIndex - 1];
      segEnd = polyline[segIndex];
      segLen = Math.hypot(segEnd[0] - segStart[0], segEnd[1] - segStart[1]);
      distInSeg = 0;
      if (segLen <= 1e-8) segLen = 1e-8;
    }
  }
  out.push(polyline[polyline.length - 1].slice());
  return out;
}

function centerAndScale(points, targetRadius) {
  if (!points.length) return [];
  let mx = 0;
  let my = 0;
  for (let i = 0; i < points.length; i++) {
    mx += points[i][0];
    my += points[i][1];
  }
  mx /= points.length;
  my /= points.length;
  const centered = points.map(function center(p) {
    return [p[0] - mx, p[1] - my];
  });
  let r = 0;
  for (let i = 0; i < centered.length; i++) {
    const ri = Math.hypot(centered[i][0], centered[i][1]);
    if (ri > r) r = ri;
  }
  const s = r > 1e-8 ? targetRadius / r : 1;
  return centered.map(function scale(p) {
    return [p[0] * s, p[1] * s];
  });
}

function makeSpiralPreset(nPoints) {
  const n = Math.max(80, nPoints == null ? 300 : nPoints | 0);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const theta = 4.75 * Math.PI * u;
    const r = 0.12 + 1.05 * u;
    pts.push([r * Math.cos(theta), r * Math.sin(theta)]);
  }
  return centerAndScale(pts, 1.2);
}

function makeTwoMoonsPreset(nPoints) {
  const n = Math.max(80, nPoints == null ? 300 : nPoints | 0);
  const half = Math.floor(n / 2);
  const pts = [];
  for (let i = 0; i < half; i++) {
    const t = (i / Math.max(1, half - 1)) * Math.PI;
    pts.push([Math.cos(t), Math.sin(t)]);
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / Math.max(1, n - half - 1)) * Math.PI;
    pts.push([1 - Math.cos(t), 0.45 - Math.sin(t)]);
  }
  return centerAndScale(pts, 1.2);
}

function makeLetterPreset(nPoints) {
  const stroke = [
    [-1.0, -1.1], [-1.0, 1.1], [-0.2, 1.1], [0.4, 0.9], [0.5, 0.35], [0.2, -0.05], [-1.0, -0.05],
    [0.3, -0.05], [1.0, -1.1],
  ];
  return centerAndScale(resamplePolyline(stroke, nPoints == null ? 300 : nPoints), 1.2);
}

const api = {
  createRng,
  randn,
  makeDdpmLinearSchedule,
  forwardPoint,
  forwardPoints,
  idealPosteriorMeanX0,
  estimateEpsilon,
  reverseStepDdpm,
  reverseSample,
  makeInitialNoise,
  nearestDistance,
  meanNearestDistance,
  dataExtent,
  resamplePolyline,
  centerAndScale,
  makeSpiralPreset,
  makeTwoMoonsPreset,
  makeLetterPreset,
};

if (typeof module !== "undefined" && module.exports) module.exports = api;
if (typeof globalThis !== "undefined") globalThis.DiffusionPlaygroundCore = api;
