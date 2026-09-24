#!/usr/bin/env python3
"""Refresh scripts/data/site_molecules.csv: the union of every molecule a site table can
show (Mayhew molecules.json, quality.json, openpom.json, intensity.json), by InChIKey.

It is the input list for the per-molecule pipelines (run_toxtree.sh, the CompTox and EPI
Suite fetchers). Existing rows keep their SMILES and order; new molecules are appended, so
downstream caches stay valid. Derived tables (toxicity, physchem) are not sources.

After running it:
  scripts/run_toxtree.sh && python3 scripts/build_toxicity.py
  python3 scripts/fetch_comptox_physchem.py scripts/data/site_molecules.csv \
      --output scripts/data/comptox_physchem_site.csv
  python3 scripts/fetch_episuite_physchem.py scripts/data/site_molecules.csv \
      --output scripts/data/episuite_physchem_site.csv
  python3 scripts/build_physchem.py"""
import json

import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

OUT = "scripts/data/site_molecules.csv"


def ikey(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m) if m else None


def rows(path, key, smi):
    d = json.load(open(path))
    return [(r[key], r[smi]) for r in (d["mols"] if isinstance(d, dict) else d) if r.get(smi)]


site = pd.read_csv(OUT)
have = {ikey(s) for s in site.smiles} - {None}
new, added = [], {}
for name, key, smi in (("molecules", "ikey", "smiles"), ("quality", "ikey", "can"),
                       ("openpom", "i", "c"), ("intensity", "i", "c")):
    n = 0
    for k, s in rows(f"docs/data/{name}.json", key, smi):
        if k and k not in have:
            have.add(k)
            new.append(s)
            n += 1
    added[name] = n

pd.concat([site, pd.DataFrame({"smiles": new})]).to_csv(OUT, index=False)
print(f"{OUT}: {len(site)} existing + {len(new)} new = {len(site) + len(new)} ({added})")
