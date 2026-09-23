#!/usr/bin/env python3
"""Build docs/data/physchem.json: vapor pressure and logP for every molecule the site knows.

One shared table so the odor (transport), intensity and toxicity modules all read the
same physical properties instead of each carrying its own copy.

Sources, best first:
  VP   : MixInt curated (EPI Suite best_vp, the 62 intensity training odorants)
         > CompTox OPERA experimental > CompTox OPERA prediction
  logP : CompTox OPERA experimental > CompTox OPERA prediction
The Mayhew et al. dataset values (molecules.json) are NOT folded in here; the client
prefers them itself because the transport boundaries were fit on them.

Input: scripts/data/comptox_physchem_site.csv from
  python3 scripts/fetch_comptox_physchem.py scripts/data/site_molecules.csv \
      --output scripts/data/comptox_physchem_site.csv
Output record: {"i": inchikey, "v": vp_mmHg, "vs": src, "l": logp, "ls": src}."""
import json, math
import pandas as pd

SRC = {"t": "MixInt curated (EPI Suite)",
       "e": "CompTox experimental",
       "p": "CompTox OPERA prediction"}

df = pd.read_csv("scripts/data/comptox_physchem_site.csv")
intensity = json.load(open("docs/data/intensity.json"))
curated = {r["i"]: r["vp"] for r in intensity["mols"] if r.get("vs") == "t"}


def sig(x, n=4):
    return float(f"{x:.{n}g}")


mols = {}
for r in df.to_dict("records"):
    k = r["inchikey"]
    if not isinstance(k, str) or k in mols:
        continue
    rec: dict = {"i": k}
    if k in curated:
        rec["v"], rec["vs"] = sig(curated[k]), "t"
    elif isinstance(r["vp_source"], str) and r["vp_mmHg"] > 0:
        rec["v"], rec["vs"] = sig(r["vp_mmHg"]), r["vp_source"][0]
    if isinstance(r["logp_source"], str) and math.isfinite(r["logp"]):
        rec["l"], rec["ls"] = round(r["logp"], 2), r["logp_source"][0]
    if len(rec) > 1:
        mols[k] = rec
for k, vp in curated.items():          # curated odorants CompTox didn't match
    mols.setdefault(k, {"i": k, "v": sig(vp), "vs": "t"})

out = {"meta": {"n": len(mols), "src": SRC,
                "note": "vp in mmHg at 25 C; logp is octanol-water (OPERA)"},
       "mols": sorted(mols.values(), key=lambda r: r["i"])}
json.dump(out, open("docs/data/physchem.json", "w"), separators=(",", ":"))
nv = sum("v" in r for r in mols.values()); nl = sum("l" in r for r in mols.values())
print(f"wrote docs/data/physchem.json: {len(mols)} molecules, {nv} with VP, {nl} with logP")
