#!/usr/bin/env python3
"""Build docs/data/openpom.json: predicted odor-character descriptors from OpenPOM
(the Principal Odor Map MPNN), keyed by InChIKey.

Input is the prediction table produced by running the OpenPOM checkpoint over the
pyrfume molecule set (see the OpenPOM project folder: code/Matej.py). Predictions are
uncalibrated sigmoid outputs and per-label base rates vary by ~3 orders of magnitude,
so a per-label quantile table is emitted alongside the scores; the client turns a raw
score into a library percentile by interpolating it."""
import argparse, json, os
from datetime import date

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DEFAULT_INPUT = os.path.expanduser(
    "~/Monell Dropbox/Mainland Lab Team Folder/Projects/OpenPOM/"
    "data/processed/df_odor_prediction.csv")

TOP_K = 15      # descriptors kept per molecule (~1.9 MB total, 0.5 MB gzipped)
N_QUANT = 101   # 0th..100th percentile per label

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=DEFAULT_INPUT)
ap.add_argument("--output", default="docs/data/openpom.json")
args = ap.parse_args()

df = pd.read_csv(args.input, sep=";", index_col=0)
labels = list(df.columns)
V = df.values.astype(np.float32)
print(f"read {V.shape[0]} predictions x {V.shape[1]} descriptors")

# --- resolve each input SMILES to an InChIKey ---------------------------------
# A handful of InChIKeys collide across tautomer / zwitterion / salt variants of the
# same input set. Prefer the neutral, single-fragment form; tie-break on first seen.
def preference(canon):
    """Lower is better."""
    frags = canon.count(".")
    charged = 1 if ("+" in canon or "-" in canon) else 0
    return (frags, charged)

best = {}       # ikey -> (preference, row index, canonical SMILES)
by_canon = {}   # canonical SMILES -> row index (every distinct form is kept)
skipped = 0
for i, smi in enumerate(df.index):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        skipped += 1
        continue
    ikey = Chem.MolToInchiKey(m)
    canon = Chem.MolToSmiles(m)
    by_canon.setdefault(canon, i)
    p = preference(canon)
    if ikey not in best or p < best[ikey][0]:
        best[ikey] = (p, i, canon)

print(f"{len(best)} unique InChIKeys ({skipped} SMILES unparseable, "
      f"{len(df.index) - skipped - len(best)} collapsed by InChIKey)")

# --- per-label quantile table over the deduplicated library --------------------
rows = np.array(sorted(i for _, i, _ in best.values()))
pct = np.linspace(0, 100, N_QUANT)
quant = [[int(round(v * 1000)) for v in np.percentile(V[rows, j], pct)]
         for j in range(len(labels))]

# --- assemble ------------------------------------------------------------------
mols = []
for ikey, (_, i, canon) in best.items():
    order = np.argsort(-V[i])[:TOP_K]
    mols.append({
        "i": ikey,
        "c": canon,
        "p": [[int(j), int(round(float(V[i, j]) * 1000))] for j in order],
    })
mols.sort(key=lambda r: r["i"])

# Alternate canonical forms of a collapsed InChIKey, so an exact structural match
# still resolves even when it lost the InChIKey tie-break above.
kept_canon = {r["c"] for r in mols}
alt = {}
for canon, i in by_canon.items():
    if canon in kept_canon:
        continue
    order = np.argsort(-V[i])[:TOP_K]
    alt[canon] = [[int(j), int(round(float(V[i, j]) * 1000))] for j in order]

out = {
    "labels": labels,
    "q": quant,
    "mols": mols,
    "alt": alt,
    "meta": {
        "n": len(mols),
        "model": "OpenPOM MPNN-POM (pom_checkpoint.pt)",
        "input": os.path.basename(args.input),
        "top_k": TOP_K,
        "built": date.today().isoformat(),
    },
}
with open(args.output, "w") as f:
    json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    f.write("\n")

size = os.path.getsize(args.output) / 1e6
above = (V[rows] > 0.5).sum(1)
print(f"wrote {len(mols)} molecules + {len(alt)} alternate forms to {args.output} "
      f"({size:.2f} MB)")
print(f"mean descriptors above 0.5: {above.mean():.1f}; "
      f"{int((above == 0).sum())} molecules with none")
ex = max(mols, key=lambda r: r["p"][0][1])
print("strongest single prediction:", ex["c"],
      [(labels[j], v / 1000) for j, v in ex["p"][:5]])
