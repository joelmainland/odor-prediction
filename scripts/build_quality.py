#!/usr/bin/env python3
"""Build docs/data/quality.json: per-molecule odor-quality descriptors aggregated
from the published pyrfume datasets, keyed by InChIKey. Modeled on the
PyrfumeDashboardPlus app.R aggregation (DB_Perception / RATA excluded — internal;
OpenPOM predictions excluded — not available)."""
import json, re
import pyrfume
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

def squish(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s))).strip()

sources = {}  # cid(int) -> {SourceName: description}

def add(cid, name, desc):
    if desc is None: return
    desc = str(desc).strip()
    if not desc or desc.lower() == "nan": return
    try: cid = int(cid)
    except Exception: return
    if cid <= 0: return
    sources.setdefault(cid, {})[name] = desc

# --- Leffingwell: one-hot -> comma list of active descriptors ---
lef = pyrfume.load_data("leffingwell/behavior.csv")
for cid, row in lef.iterrows():
    active = [c for c in lef.columns if row[c] == 1]
    if active: add(cid, "Leffingwell", ", ".join(active))

# --- Goodscents: Description (odor) joined to CID via opl table ---
gs_odor = pyrfume.load_data("goodscents/data_rw_odor.csv")[["Description"]]
gs_opl = pyrfume.load_data("goodscents/data_rw_opl.csv")[["CID"]]
gs = gs_odor.join(gs_opl, how="inner").dropna(subset=["CID"])
for _, r in gs.iterrows():
    add(r["CID"], "Goodscents", r["Description"])

# --- AromaDB: filtered descriptors (index = CID) ---
ar = pyrfume.load_data("aromadb/behavior.csv")
for cid, r in ar.iterrows():
    add(cid, "AromaDB", r["Filtered Descriptors"])

# --- IFRA: up to 3 descriptors, CID via stimuli ---
ifra_b = pyrfume.load_data("ifra_2019/behavior.csv")
ifra_s = pyrfume.load_data("ifra_2019/stimuli.csv")
ifra = ifra_b.join(ifra_s, how="inner")
for _, r in ifra.iterrows():
    ds = [str(r[c]) for c in ["Descriptor 1", "Descriptor 2", "Descriptor 3"]
          if pd.notna(r[c]) and str(r[c]).strip()]
    add(r["CID"], "IFRA", ", ".join(ds))

# --- Arctander: Labels (punct->space), CID via stimuli new_CID ---
arc_b = pyrfume.load_data("arctander_1960/behavior_1_sparse.csv")
arc_s = pyrfume.load_data("arctander_1960/stimuli.csv")[["new_CID"]]
arc = arc_b.join(arc_s, how="inner")
for _, r in arc.iterrows():
    if pd.notna(r["Labels"]):
        add(r["new_CID"], "Arctander", squish(r["Labels"]))

# --- Dravnieks: top-5 usage descriptors at high concentration ---
dr_b = pyrfume.load_data("dravnieks_1985/behavior_2.csv")
dr_s = pyrfume.load_data("dravnieks_1985/stimuli.csv")[["CID", "Conc"]]
dr = dr_b.join(dr_s, how="inner")
dr = dr[dr["Conc"] == "high"]
desc_cols = [c for c in dr_b.columns]
for _, r in dr.iterrows():
    vals = r[desc_cols].astype(float)
    top = vals.sort_values(ascending=False).head(5)
    top = top[top > 0]
    if len(top):
        add(r["CID"], "Dravnieks",
            ", ".join(f"{d.lower()} ({round(v, 1)})" for d, v in top.items()))

# --- Sigma-Aldrich: descriptors (punct->space), index = CID ---
sig = pyrfume.load_data("sigma_2014/behavior.csv")
for cid, r in sig.iterrows():
    add(cid, "Sigma", squish(r["descriptors"]))

# --- names + SMILES for keying ---
mol = pyrfume.load_data("molecules/molecules.csv")
names = mol["name"].to_dict()
smiles = mol["IsomericSMILES"].to_dict()

# --- assemble, key by InChIKey ---
out, skipped = [], 0
for cid, srcs in sources.items():
    smi = smiles.get(cid)
    if not smi or not isinstance(smi, str):
        skipped += 1; continue
    m = Chem.MolFromSmiles(smi)
    if not m:
        skipped += 1; continue
    rec = {
        "ikey": Chem.MolToInchiKey(m),
        "can": Chem.MolToSmiles(m),
        "cid": cid,
        "name": (str(names.get(cid)) if pd.notna(names.get(cid)) else ""),
        "sources": srcs,
        "n": len(srcs),
    }
    out.append(rec)

out.sort(key=lambda r: -r["n"])
with open("docs/data/quality.json", "w") as f:
    json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    f.write("\n")

print(f"wrote {len(out)} molecules (skipped {skipped} without usable structure)")
from collections import Counter
c = Counter(s for r in out for s in r["sources"])
print("per-source counts:", dict(c))
print("example (most sources):")
ex = out[0]
print(" ", ex["name"], ex["cid"], ex["ikey"])
for s, d in ex["sources"].items():
    print(f"    {s}: {d[:70]}")
