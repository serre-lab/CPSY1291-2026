import { performance } from "node:perf_hooks";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createContrastiveViewsCore } = require("./contrastive_views_core.js");

function runCase(name, setup) {
  const core = createContrastiveViewsCore({
    seed: 1234,
    hiddenSize: 64,
    batchSize: 24,
    learningRate: 0.008,
    temperature: 0.20,
    repulsion: true,
    augment: { shift: true, colorJitter: true, blur: false },
  });
  setup(core);
  const t0 = performance.now();
  core.trainSteps(360);
  const metrics = core.computeMetrics();
  const dt = performance.now() - t0;
  return { name, metrics, ms: dt };
}

const a = runCase("a", (core) => {
  core.setAugmentations({ shift: true, colorJitter: true, blur: false });
  core.setRepulsion(true);
  core.setTemperature(0.20);
});

const b = runCase("b", (core) => {
  core.setAugmentations({ shift: true, colorJitter: false, blur: false });
  core.setRepulsion(true);
  core.setTemperature(0.20);
});

const c = runCase("c", (core) => {
  core.setAugmentations({ shift: true, colorJitter: true, blur: false });
  core.setRepulsion(false);
  core.setTemperature(0.20);
});

function pct(x) {
  return (100 * x).toFixed(1) + "%";
}

function report(x) {
  return `${x.name}: shape=${pct(x.metrics.shapeAccuracy)} color=${pct(x.metrics.colorAccuracy)} spread=${x.metrics.spread.toFixed(4)} step=${x.metrics.step} time=${x.ms.toFixed(1)}ms`;
}

console.log(report(a));
console.log(report(b));
console.log(report(c));

const condA = a.metrics.shapeAccuracy > a.metrics.colorAccuracy + 0.12;
const condB = b.metrics.colorAccuracy > b.metrics.shapeAccuracy + 0.12;
const condC = c.metrics.spread < a.metrics.spread * 0.55;
const condTime = a.ms < 3000 && b.ms < 3000 && c.ms < 3000;

if (!condA || !condB || !condC || !condTime) {
  console.error("FAIL", {
    condA,
    condB,
    condC,
    condTime,
    a: a.metrics,
    b: b.metrics,
    c: c.metrics,
  });
  process.exit(1);
}

console.log("PASS");
