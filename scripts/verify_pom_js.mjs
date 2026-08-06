/* Parity harness for the in-browser OpenPOM implementation.

   Runs docs/pom-worker.js over molecules from the reference prediction set and
   compares against the values produced by the real openpom/DGL stack. This is the
   gate for the whole in-browser path: the featurizer reconstructs RDKit-derived
   quantities (notably hybridization) from commonchem JSON, and the forward pass is
   a hand port of the DGL model, so both need to be held to the reference.

   Prepare the fixtures first (writes to /tmp):
       python3 scripts/make_pom_fixtures.py [n]
   Then:
       node scripts/verify_pom_js.mjs
*/
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const worker = require(path.join(root, "docs/pom-worker.js"));

const FIXTURES = "/tmp/pom_fixtures.json";
if (!fs.existsSync(FIXTURES)) {
  console.error(`missing ${FIXTURES} — run: python3 scripts/make_pom_fixtures.py`);
  process.exit(1);
}

// --- load the exported weights straight off disk (the browser path uses fetch) ---
const manifest = JSON.parse(fs.readFileSync(path.join(root, "docs/data/pom_weights.json")));
const raw = fs.readFileSync(path.join(root, "docs/data/pom_weights.bin"));
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const W = {};
for (const t of manifest.tensors) {
  const size = t.shape.reduce((a, b) => a * b, 1);
  if (manifest.dtype === "fp16") {
    const u16 = new Uint16Array(buf, t.offset, size);
    const a = new Float32Array(size);
    for (let i = 0; i < size; i++) a[i] = worker.halfToFloat(u16[i]);
    W[t.name] = a;
  } else {
    W[t.name] = new Float32Array(buf.slice(t.offset, t.offset + size * 4));
  }
}
worker.setWeights(W);

const fx = JSON.parse(fs.readFileSync(FIXTURES));
console.log(`weights: ${manifest.dtype}, ${manifest.tensors.length} tensors`);
console.log(`fixtures: ${fx.cases.length} molecules\n`);

// --- 1. featurizer parity: exact match against RDKit-derived features ---------
let featBad = 0, atomsChecked = 0, bondsChecked = 0, featExample = null;
for (const c of fx.cases) {
  if (!c.atom_features) continue;
  const g = worker.buildGraph(c.json);
  const { X, EF } = worker.featurize(g);
  for (let i = 0; i < c.atom_features.length; i++) {
    atomsChecked++;
    const want = c.atom_features[i];
    for (let k = 0; k < 134; k++) {
      if (X[i * 134 + k] !== want[k]) {
        featBad++;
        featExample ??= { smiles: c.smiles, atom: i, k };
        break;
      }
    }
  }
  // Edge features are duplicated per bond direction; check the first of each pair.
  for (let b = 0; b < c.bond_features.length; b++) {
    bondsChecked++;
    const want = c.bond_features[b];
    for (let k = 0; k < 6; k++) {
      if (EF[2 * b * 6 + k] !== want[k]) {
        featBad++;
        featExample ??= { smiles: c.smiles, bond: b, k };
        break;
      }
    }
  }
}
console.log(`featurizer: ${atomsChecked} atoms + ${bondsChecked} bonds checked, ` +
            `${featBad} mismatches` + (featExample ? ` e.g. ${JSON.stringify(featExample)}` : ""));

// --- 2. end-to-end parity against the reference predictions ------------------
let worst = 0, worstSmiles = "", reorder = 0, totalMs = 0;
const diffs = [];
for (const c of fx.cases) {
  const t0 = performance.now();
  const { probs } = worker.forward(c.json);
  totalMs += performance.now() - t0;
  let d = 0;
  for (let i = 0; i < probs.length; i++) d = Math.max(d, Math.abs(probs[i] - c.expected[i]));
  diffs.push(d);
  if (d > worst) { worst = d; worstSmiles = c.smiles; }
  const top = (a) => Array.from(a.keys ? a.keys() : a.map((_, i) => i))
    .sort((x, y) => a[y] - a[x]).slice(0, 8).join(",");
  const topJs = Array.from({ length: probs.length }, (_, i) => i)
    .sort((x, y) => probs[y] - probs[x]).slice(0, 8).join(",");
  const topRef = Array.from({ length: c.expected.length }, (_, i) => i)
    .sort((x, y) => c.expected[y] - c.expected[x]).slice(0, 8).join(",");
  if (topJs !== topRef) reorder++;
  void top;
}
diffs.sort((a, b) => a - b);
const median = diffs[Math.floor(diffs.length / 2)];
console.log(`predictions: max abs diff ${worst.toExponential(2)} (worst: ${worstSmiles})`);
console.log(`             median abs diff ${median.toExponential(2)}`);
console.log(`             top-8 order differs for ${reorder}/${fx.cases.length} molecules`);
console.log(`speed:       ${(totalMs / fx.cases.length).toFixed(1)} ms/molecule\n`);

// fp16 weights cost ~7e-4; fp32 should land near 1e-6. Gate accordingly.
const TOL = manifest.dtype === "fp16" ? 3e-3 : 1e-4;
const ok = featBad === 0 && worst < TOL;
console.log(ok ? `PASS (featurizer exact, predictions within ${TOL})`
                : `FAIL (featurizer mismatches: ${featBad}, max diff ${worst.toExponential(2)}, tol ${TOL})`);
process.exit(ok ? 0 : 1);
