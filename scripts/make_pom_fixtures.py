#!/usr/bin/env python3
"""Build the fixture set for scripts/verify_pom_js.mjs.

For a random sample of the reference molecules, emit:
  - the commonchem JSON that RDKit.js get_json() would hand the worker,
  - the RDKit-derived atom/bond feature vectors the worker must reproduce exactly,
  - the reference probabilities from the real openpom/DGL stack.
"""
import json, os, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

CSV = os.path.expanduser("~/Monell Dropbox/Mainland Lab Team Folder/Projects/OpenPOM/"
                         "data/processed/df_odor_prediction.csv")
OUT = "/tmp/pom_fixtures.json"

VALENCE = [0, 1, 2, 3, 4, 5, 6]; DEGREE = [0, 1, 2, 3, 4, 5]
NUM_HS = [0, 1, 2, 3, 4]; CHARGE = [-1, -2, 1, 2, 0]
ATOMIC = list(range(100)); HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]


def onehot(val, allowed):
    v = [0] * (len(allowed) + 1)
    v[allowed.index(val) if val in allowed else len(allowed)] = 1
    return v


def atom_features(a):
    return (onehot(a.GetTotalValence(), VALENCE) + onehot(a.GetTotalDegree(), DEGREE)
            + onehot(a.GetTotalNumHs(), NUM_HS) + onehot(a.GetFormalCharge(), CHARGE)
            + onehot(a.GetAtomicNum() - 1, ATOMIC)
            + onehot(str(a.GetHybridization()), HYB))


def bond_features(b):
    bt = b.GetBondType()
    return [0, int(bt == Chem.BondType.SINGLE), int(bt == Chem.BondType.DOUBLE),
            int(bt == Chem.BondType.TRIPLE), int(bt == Chem.BondType.AROMATIC),
            int(b.IsInRing())]


n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
df = pd.read_csv(CSV, sep=";", index_col=0)
rng = np.random.default_rng(7)
idx = list(rng.choice(len(df), min(n, len(df)), replace=False))

# A random sample is nearly all C/H/N/O/S, but the featurizer's periodic-table and
# conjugation paths differ for metals and heavier elements. Force those in.
COMMON = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
odd = []
for i, smi in enumerate(df.index):
    m = Chem.MolFromSmiles(str(smi))
    if m and any(a.GetAtomicNum() not in COMMON for a in m.GetAtoms()):
        odd.append(i)
rng.shuffle(odd)
idx = list(dict.fromkeys(idx + odd[:60]))
print(f"{len(odd)} molecules contain uncommon elements; including "
      f"{min(60, len(odd))} of them")

cases = []
for i in idx:
    smi = str(df.index[i])
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    cases.append({
        "smiles": smi,
        "json": json.loads(Chem.MolToJSON(mol)),
        "atom_features": [atom_features(a) for a in mol.GetAtoms()],
        "bond_features": [bond_features(b) for b in mol.GetBonds()],
        "expected": [float(v) for v in df.values[i]],
    })

with open(OUT, "w") as f:
    json.dump({"labels": list(df.columns), "cases": cases}, f)
print(f"wrote {len(cases)} fixtures to {OUT}")
