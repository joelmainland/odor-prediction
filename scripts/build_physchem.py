#!/usr/bin/env python3
"""Build docs/data/physchem.json: vapor pressure and logP for every molecule the site knows.

One shared table so the odor (transport), intensity and toxicity modules all read the
same physical properties instead of each carrying its own copy.

Sources: MixInt curated VPs (EPI Suite best_vp, the 62 intensity training odorants) win;
everything else is ranked by scripts/physchem_sources.py (CompTox OPERA + EPI Suite,
measured before predicted, own record before a stereoisomer's).
The Mayhew et al. dataset values (molecules.json) are NOT folded in here; the client
prefers them itself because the transport boundaries were fit on them.

Inputs, for the site tables (site_molecules.csv) and the intensity library (AllDragon):
  python3 scripts/fetch_comptox_physchem.py scripts/data/site_molecules.csv \
      --output scripts/data/comptox_physchem_site.csv
  python3 scripts/fetch_episuite_physchem.py scripts/data/site_molecules.csv \
      --output scripts/data/episuite_physchem_site.csv
  (and scripts/data/{comptox,episuite}_physchem.csv from alldragon_molecules.csv)
Output record: {"i": inchikey, "v": vp_mmHg, "vs": src, "st": 1 if the VP is a
stereoisomer's, "l": logp, "ls": src}."""
import json

import pandas as pd

from physchem_sources import SRC, resolve

SRC = {"t": "MixInt curated (EPI Suite)", **SRC}

ct = pd.concat([pd.read_csv("scripts/data/comptox_physchem_site.csv"),
                pd.read_csv("scripts/data/comptox_physchem.csv")], ignore_index=True)
ep = pd.concat([pd.read_csv("scripts/data/episuite_physchem_site.csv"),
                pd.read_csv("scripts/data/episuite_physchem.csv")], ignore_index=True)
intensity = json.load(open("docs/data/intensity.json"))
curated = {r["i"]: r["vp"] for r in intensity["mols"] if r.get("vs") == "t"}


def sig(x, n=4):
    return float(f"{x:.{n}g}")


keys = sorted(set(ct.inchikey.dropna()) | set(ep.inchikey.dropna()) | set(curated))
res = resolve(keys, ct, ep)
mols = {}
for k in keys:
    r, rec = res.get(k, {}), {"i": k}
    if k in curated:
        rec["v"], rec["vs"] = sig(curated[k]), "t"
    elif "vp" in r:
        rec["v"], rec["vs"] = sig(r["vp"]), r["vs"]
        if r["st"]:
            rec["st"] = 1
    if "logp" in r:
        rec["l"], rec["ls"] = round(r["logp"], 2), r["ls"]
    if len(rec) > 1:
        mols[k] = rec

out = {"meta": {"n": len(mols), "src": SRC,
                "pred": sorted(c for c in SRC if c in "pmk"),
                "note": "vp in mmHg at 25 C; logp is octanol-water (OPERA or KOWWIN)"},
       "mols": [mols[k] for k in sorted(mols)]}
json.dump(out, open("docs/data/physchem.json", "w"), separators=(",", ":"))
vs = pd.Series([r.get("vs") for r in mols.values()]).value_counts().to_dict()
ls = pd.Series([r.get("ls") for r in mols.values()]).value_counts().to_dict()
print(f"wrote docs/data/physchem.json: {len(mols)} molecules; VP sources {vs}; logP sources {ls}; "
      f"{sum('st' in r for r in mols.values())} VPs from a stereoisomer")
