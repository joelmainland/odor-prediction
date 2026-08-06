/* OpenPOM inference in the browser.
   Runs the MPNN-POM forward pass (Lee et al., Science 2023; openpom implementation)
   on a molecular graph, off the main thread.

   The main thread parses the molecule with RDKit.js and posts the commonchem JSON
   from mol.get_json(); this worker derives the 134-dim atom / 6-dim bond features
   from it and runs the network. Deriving the features here (rather than shipping a
   second RDKit wasm instance into the worker) is what keeps this cheap — every
   quantity openpom needs is recoverable from commonchem, including hybridization,
   which is reconstructed from RDKit's own orbital-counting rules below.

   Feature parity with RDKit and output parity with the reference implementation are
   both enforced by scripts/verify_pom_js.mjs. */

/* eslint-env worker */
"use strict";

// ---- Periodic-table constants, indexed by atomic number ---------------------
// Verbatim from RDKit's PeriodicTable (GetNOuterElecs / GetDefaultValence) for
// Z = 0..103, so metal-containing species behave as RDKit does. -1 default valence
// means "no default", which the conjugation code treats as non-conjugatable.
const NOUTER = [0,1,2,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,9,10,11,2,3,4,
  5,6,7,8,1,2,3,4,5,6,7,8,9,10,11,2,3,4,5,6,7,8,1,2,3,4,3,4,5,6,7,8,9,10,11,12,13,14,
  15,4,5,6,7,8,9,10,11,2,3,4,5,6,7,8,1,2,3,4,3,4,5,6,7,8,9,10,11,12,13,14,15];
const DEFVAL = [-1,1,0,1,2,3,4,3,2,1,0,1,2,3,4,3,2,1,0,1,2,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,3,4,3,2,1,0,1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,2,3,2,1,0,1,2,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,2,3,2,1,0,1,2,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];

const VALENCE = [0,1,2,3,4,5,6], DEGREE = [0,1,2,3,4,5], NUM_HS = [0,1,2,3,4];
const CHARGE = [-1,-2,1,2,0];
const HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"];
const ATOM_FDIM = 134, BOND_FDIM = 6;
const NODE = 100, EDGE_OUT = 100, S2S_IN = 200, S2S_OUT = 400;
const MP_STEPS = 5, S2S_STEPS = 3, N_TASKS = 138;

// ---- Graph derived from commonchem ------------------------------------------
function buildGraph(doc) {
  const m = doc.molecules[0];
  const ad = (doc.defaults && doc.defaults.atom) || {};
  const bd = (doc.defaults && doc.defaults.bond) || {};
  const atoms = m.atoms.map((a) => Object.assign({ z: 6, chg: 0, impHs: 0, nRad: 0 }, ad, a));
  const bonds = m.bonds.map((b) => Object.assign({ bo: 1 }, bd, b));
  const ext = (m.extensions || []).find((e) => e.name === "rdkitRepresentation") || {};
  const aromBonds = new Set(ext.aromaticBonds || []);

  const n = atoms.length, nb = bonds.length;
  // Ring membership per bond, from the ring atom lists.
  const ringPairs = new Set();
  for (const r of ext.atomRings || []) {
    for (let i = 0; i < r.length; i++) {
      const a = r[i], b = r[(i + 1) % r.length];
      ringPairs.add(a < b ? `${a},${b}` : `${b},${a}`);
    }
  }
  const inRing = bonds.map((b) => {
    const [u, v] = b.atoms;
    return ringPairs.has(u < v ? `${u},${v}` : `${v},${u}`);
  });

  const abonds = Array.from({ length: n }, () => []);
  bonds.forEach((b, i) => { abonds[b.atoms[0]].push(i); abonds[b.atoms[1]].push(i); });

  // Two notions of bond order, both of which RDKit uses:
  //  contrib — getValenceContrib(), aromatic counts 1.5; used by the conjugation code.
  //  kek     — the kekulized order in commonchem; summing these gives explicit valence
  //            (furan O is 2, not the 3.0 that summing 1.5s would imply).
  const contrib = bonds.map((b, i) => (aromBonds.has(i) ? 1.5 : b.bo));
  const nH = atoms.map((a) => a.impHs);
  const deg = abonds.map((l) => l.length);
  const tdeg = deg.map((d, i) => d + nH[i]);
  const expval = abonds.map((l) => l.reduce((s, bi) => s + bonds[bi].bo, 0));
  const tval = expval.map((v, i) => v + nH[i]);

  const g = { atoms, bonds, abonds, aromBonds, inRing, contrib, nH, deg, tdeg,
              expval, tval, n, nb };

  // --- conjugation (RDKit ConjugHybrid.cpp setConjugation) --------------------
  const conj = bonds.map((_, i) => aromBonds.has(i));
  const countAtomElec = (i) => {
    const z = atoms[i].z, dv = DEFVAL[z];
    if (dv === undefined || dv <= 1) return -1;
    let degree = tdeg[i];
    for (const bi of abonds[i]) if (Math.round(contrib[bi]) === 0) degree--;
    if (degree > 3) return -1;
    const nlp = Math.max((NOUTER[z] || 0) - dv - atoms[i].chg, 0);
    let res = (dv - degree) + nlp - atoms[i].nRad;
    if (res > 1 && expval[i] - deg[i] > 1) res = 1;
    return res;
  };
  const isConjCand = (i) => {
    const z = atoms[i].z, nouter = NOUTER[z] || 0;
    return (z <= 10 || (nouter !== 5 && nouter !== 6) || (nouter === 6 && tdeg[i] < 2))
      && countAtomElec(i) > 0;
  };
  for (let i = 0; i < n; i++) {
    if (!isConjCand(i) || tdeg[i] < 2 || tdeg[i] > 3) continue;
    for (const b1 of abonds[i]) {
      if (contrib[b1] < 1.5) continue;
      for (const b2 of abonds[i]) {
        if (b1 === b2) continue;
        const [x, y] = bonds[b2].atoms;
        const j = x === i ? y : x;
        if (tdeg[j] > 3) continue;
        if (isConjCand(j)) { conj[b1] = true; conj[b2] = true; }
      }
    }
  }
  g.conj = conj;

  // --- hybridization (RDKit setHybridization) --------------------------------
  g.hyb = atoms.map((a, i) => {
    const z = a.z;
    if (z === 0) return "UNSPECIFIED";
    let norbs;
    if (z >= 89) {
      norbs = tdeg[i];
    } else if (z <= 1) {
      norbs = tdeg[i];
    } else {
      const nouter = NOUTER[z];
      if (nouter === undefined) return "UNSPECIFIED";
      const free = nouter - (tval[i] + a.chg);
      norbs = (tval[i] + nouter - a.chg < 8)
        ? tdeg[i] + Math.floor((free - a.nRad) / 2) + a.nRad
        : tdeg[i] + Math.floor(free / 2);
    }
    switch (norbs) {
      case 0: case 1: return "S";
      case 2: return "SP";
      case 3: return "SP2";
      // SP3 unless a lone pair is conjugated (the second O in O=CO), but never for
      // atoms of degree > 3 — RDKit issue 276.
      case 4: return (tdeg[i] > 3 || !abonds[i].some((b) => conj[b])) ? "SP3" : "SP2";
      case 5: return "SP3D";
      case 6: return "SP3D2";
      default: return "UNSPECIFIED";
    }
  });
  return g;
}

function oneHot(out, base, val, allowed) {
  const k = allowed.indexOf(val);
  out[base + (k < 0 ? allowed.length : k)] = 1;
}

function featurize(g) {
  const X = new Float32Array(g.n * ATOM_FDIM);
  for (let i = 0; i < g.n; i++) {
    const o = i * ATOM_FDIM, a = g.atoms[i];
    oneHot(X, o, g.tval[i], VALENCE);
    oneHot(X, o + 8, g.tdeg[i], DEGREE);
    oneHot(X, o + 15, g.nH[i], NUM_HS);
    oneHot(X, o + 21, a.chg, CHARGE);
    // atomic number is one-hot over (Z - 1), per openpom's get_atomic_num_one_hot
    X[o + 27 + (a.z - 1 >= 0 && a.z - 1 < 100 ? a.z - 1 : 100)] = 1;
    oneHot(X, o + 128, g.hyb[i], HYB);
  }
  // Each bond becomes two directed edges sharing one feature vector.
  const E = g.nb * 2;
  const src = new Int32Array(E), dst = new Int32Array(E);
  const EF = new Float32Array(E * BOND_FDIM);
  for (let b = 0; b < g.nb; b++) {
    const [u, v] = g.bonds[b].atoms;
    src[2 * b] = u; dst[2 * b] = v;
    src[2 * b + 1] = v; dst[2 * b + 1] = u;
    const arom = g.aromBonds.has(b), bo = g.bonds[b].bo;
    for (const e of [2 * b, 2 * b + 1]) {
      const o = e * BOND_FDIM;
      EF[o + 1] = !arom && bo === 1 ? 1 : 0;
      EF[o + 2] = !arom && bo === 2 ? 1 : 0;
      EF[o + 3] = !arom && bo === 3 ? 1 : 0;
      EF[o + 4] = arom ? 1 : 0;
      EF[o + 5] = g.inRing[b] ? 1 : 0;
    }
  }
  return { X, src, dst, EF, E };
}

// ---- Weights ----------------------------------------------------------------
let W = null;

function halfToFloat(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

async function loadWeights(base) {
  const [manifest, buf] = await Promise.all([
    fetch(base + "pom_weights.json").then((r) => r.json()),
    fetch(base + "pom_weights.bin").then((r) => r.arrayBuffer()),
  ]);
  const out = {};
  for (const t of manifest.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (manifest.dtype === "fp16") {
      const u16 = new Uint16Array(buf, t.offset, size);
      const a = new Float32Array(size);
      for (let i = 0; i < size; i++) a[i] = halfToFloat(u16[i]);
      out[t.name] = a;
    } else {
      out[t.name] = new Float32Array(buf.slice(t.offset, t.offset + size * 4));
    }
    out[t.name].shape = t.shape;
  }
  return out;
}

// ---- Dense ops (row-major, y = x @ Wᵀ + b) ----------------------------------
function linear(x, rows, w, b, inDim, outDim) {
  const y = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outDim; o++) {
      let s = b ? b[o] : 0;
      const wo = o * inDim, xo = r * inDim;
      for (let i = 0; i < inDim; i++) s += x[xo + i] * w[wo + i];
      y[r * outDim + o] = s;
    }
  }
  return y;
}

const relu = (a) => { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; };
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function gru(x, h, n) {
  // PyTorch GRU gate order [r, z, n]
  const gi = linear(x, n, W["gru.w_ih"], W["gru.b_ih"], NODE, 3 * NODE);
  const gh = linear(h, n, W["gru.w_hh"], W["gru.b_hh"], NODE, 3 * NODE);
  const out = new Float32Array(n * NODE);
  for (let r = 0; r < n; r++) {
    const g = r * 3 * NODE, o = r * NODE;
    for (let i = 0; i < NODE; i++) {
      const rr = sigmoid(gi[g + i] + gh[g + i]);
      const zz = sigmoid(gi[g + NODE + i] + gh[g + NODE + i]);
      const nn = Math.tanh(gi[g + 2 * NODE + i] + rr * gh[g + 2 * NODE + i]);
      out[o + i] = (1 - zz) * nn + zz * h[o + i];
    }
  }
  return out;
}

function lstmCell(x, h, c, L, inDim) {
  // PyTorch LSTM gate order [i, f, g, o]
  const H = S2S_IN;
  const gi = linear(x, 1, W[`lstm.${L}.weight_ih`], W[`lstm.${L}.bias_ih`], inDim, 4 * H);
  const gh = linear(h, 1, W[`lstm.${L}.weight_hh`], W[`lstm.${L}.bias_hh`], H, 4 * H);
  const nh = new Float32Array(H), nc = new Float32Array(H);
  for (let i = 0; i < H; i++) {
    const ii = sigmoid(gi[i] + gh[i]);
    const ff = sigmoid(gi[H + i] + gh[H + i]);
    const gg = Math.tanh(gi[2 * H + i] + gh[2 * H + i]);
    const oo = sigmoid(gi[3 * H + i] + gh[3 * H + i]);
    nc[i] = ff * c[i] + ii * gg;
    nh[i] = oo * Math.tanh(nc[i]);
  }
  return [nh, nc];
}

function forward(doc) {
  const g = buildGraph(doc);
  const { X, src, dst, EF, E } = featurize(g);
  const n = g.n;

  // --- message passing: project -> 5x (NNConv + ReLU + GRU) ------------------
  let h = relu(linear(X, n, W["proj_node.w"], W["proj_node.b"], ATOM_FDIM, NODE));
  let hidden = h.slice();

  // Edge-conditioned NODE x NODE matrices. Edge features don't change between
  // steps and the conv layer is shared, so these are built once.
  let EW = null;
  if (E) {
    const eh = relu(linear(EF, E, W["edge_func.0.w"], W["edge_func.0.b"], BOND_FDIM, 75));
    EW = linear(eh, E, W["edge_func.2.w"], W["edge_func.2.b"], 75, NODE * NODE);
  }
  const convBias = W["conv.bias"];
  for (let step = 0; step < MP_STEPS; step++) {
    const neigh = new Float32Array(n * NODE);
    for (let e = 0; e < E; e++) {
      const hs = src[e] * NODE, no = dst[e] * NODE, wo = e * NODE * NODE;
      for (let i = 0; i < NODE; i++) {
        const hv = h[hs + i];
        if (hv === 0) continue;
        const row = wo + i * NODE;
        for (let o = 0; o < NODE; o++) neigh[no + o] += hv * EW[row + o];
      }
    }
    // NNConv residual is Identity here (in_feats == out_feats), so h passes through.
    for (let i = 0; i < n * NODE; i++) {
      const v = neigh[i] + h[i] + convBias[i % NODE];
      neigh[i] = v > 0 ? v : 0;
    }
    h = gru(neigh, hidden, n);
    hidden = h;
  }

  // --- readout: radius-0 fold of atom + bond embeddings, then set2set --------
  const feat = new Float32Array(n * S2S_IN);
  if (E) {
    const ee = relu(linear(EF, E, W["proj_edge.w"], W["proj_edge.b"], BOND_FDIM, EDGE_OUT));
    for (let e = 0; e < E; e++) {
      const o = dst[e] * S2S_IN, hs = src[e] * NODE, es = e * EDGE_OUT;
      for (let i = 0; i < NODE; i++) feat[o + i] += h[hs + i];
      for (let i = 0; i < EDGE_OUT; i++) feat[o + NODE + i] += ee[es + i];
    }
  }

  let hs0 = new Float32Array(S2S_IN), cs0 = new Float32Array(S2S_IN);
  let hs1 = new Float32Array(S2S_IN), cs1 = new Float32Array(S2S_IN);
  let qStar = new Float32Array(S2S_OUT);
  for (let t = 0; t < S2S_STEPS; t++) {
    [hs0, cs0] = lstmCell(qStar, hs0, cs0, 0, S2S_OUT);
    [hs1, cs1] = lstmCell(hs0, hs1, cs1, 1, S2S_IN);
    const q = hs1;
    const e = new Float32Array(n);
    let mx = -Infinity;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let d = 0; d < S2S_IN; d++) s += feat[i * S2S_IN + d] * q[d];
      e[i] = s; if (s > mx) mx = s;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) { e[i] = Math.exp(e[i] - mx); sum += e[i]; }
    const readout = new Float32Array(S2S_IN);
    for (let i = 0; i < n; i++) {
      const a = e[i] / sum;
      for (let d = 0; d < S2S_IN; d++) readout[d] += a * feat[i * S2S_IN + d];
    }
    qStar = new Float32Array(S2S_OUT);
    qStar.set(q, 0); qStar.set(readout, S2S_IN);
  }

  // --- FFN (batch-norms 0 and 1 are folded into the linears by the exporter) --
  let x = relu(linear(qStar, 1, W["ffn.0.w"], W["ffn.0.b"], S2S_OUT, 392));
  x = relu(linear(x, 1, W["ffn.1.w"], W["ffn.1.b"], 392, 392));
  const emb = linear(x, 1, W["ffn.2.w"], W["ffn.2.b"], 392, 256);
  const post = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const v = emb[i] * W["bn2.scale"][i] + W["bn2.shift"][i];
    post[i] = v > 0 ? v : 0;
  }
  const out = linear(post, 1, W["ffn.3.w"], W["ffn.3.b"], 256, N_TASKS);
  const probs = new Float32Array(N_TASKS);
  for (let i = 0; i < N_TASKS; i++) probs[i] = sigmoid(out[i]);
  return { probs, embedding: emb };
}

// ---- Worker protocol --------------------------------------------------------
let ready = null;

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = async (ev) => {
    const { json, seq, base } = ev.data;
    try {
      if (!ready) ready = loadWeights(base || "data/").then((w) => { W = w; });
      await ready;
      const t0 = (self.performance || Date).now();
      const { probs } = forward(typeof json === "string" ? JSON.parse(json) : json);
      self.postMessage({ seq, probs, ms: (self.performance || Date).now() - t0 });
    } catch (err) {
      ready = null;  // let a later request retry a failed weight download
      self.postMessage({ seq, error: String((err && err.message) || err) });
    }
  };
}

// Exposed for the Node parity harness (scripts/verify_pom_js.mjs).
if (typeof module !== "undefined" && module.exports) {
  module.exports = { buildGraph, featurize, forward, loadWeights,
                     halfToFloat, setWeights: (w) => { W = w; } };
}
