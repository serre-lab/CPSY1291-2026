import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const CORE = require("./diffusion_playground_core.js");

function stats2d(points) {
  let mx = 0;
  let my = 0;
  for (const p of points) {
    mx += p[0];
    my += p[1];
  }
  mx /= points.length;
  my /= points.length;
  let vx = 0;
  let vy = 0;
  for (const p of points) {
    vx += (p[0] - mx) ** 2;
    vy += (p[1] - my) ** 2;
  }
  vx /= points.length;
  vy /= points.length;
  return { mx, my, sx: Math.sqrt(vx), sy: Math.sqrt(vy) };
}

function fail(msg) {
  throw new Error(msg);
}

const seed = 12345;
const schedule = CORE.makeDdpmLinearSchedule(100);
const data = CORE.makeSpiralPreset(300);
const noiseRng = CORE.createRng(seed);
const forwardT = CORE.forwardPoints(data, schedule.T, schedule, noiseRng);
const s = stats2d(forwardT);

const meanTol = 0.12;
const stdTol = 0.15;
if (Math.abs(s.mx) > meanTol || Math.abs(s.my) > meanTol) {
  fail(`Forward-at-T mean too far from 0: mx=${s.mx.toFixed(4)} my=${s.my.toFixed(4)}`);
}
if (Math.abs(s.sx - 1) > stdTol || Math.abs(s.sy - 1) > stdTol) {
  fail(`Forward-at-T std too far from 1: sx=${s.sx.toFixed(4)} sy=${s.sy.toFixed(4)}`);
}

const reverseRng = CORE.createRng(seed + 1);
const initial = CORE.makeInitialNoise(500, reverseRng);
const initialDist = CORE.meanNearestDistance(initial, data);
const final = CORE.reverseSample(initial, data, schedule, reverseRng);
const finalDist = CORE.meanNearestDistance(final, data);
const extent = CORE.dataExtent(data);
const fracExtent = finalDist / extent;
const improvement = initialDist / finalDist;

if (!(fracExtent < 0.05)) {
  fail(`Reverse final distance ${finalDist.toFixed(5)} is not below 5% of extent ${extent.toFixed(5)} (ratio=${fracExtent.toFixed(5)})`);
}
if (!(improvement >= 10)) {
  fail(`Reverse improvement ${improvement.toFixed(3)}x is below required 10x (initial=${initialDist.toFixed(5)}, final=${finalDist.toFixed(5)})`);
}

console.log("Diffusion acceptance test passed.");
console.log(`Forward @T: mean=(${s.mx.toFixed(4)}, ${s.my.toFixed(4)}), std=(${s.sx.toFixed(4)}, ${s.sy.toFixed(4)})`);
console.log(`Reverse: initial mean-NN=${initialDist.toFixed(5)}, final mean-NN=${finalDist.toFixed(5)}, extent=${extent.toFixed(5)}`);
console.log(`Final/extent=${fracExtent.toFixed(5)}, improvement=${improvement.toFixed(3)}x`);
