#!/usr/bin/env python3
"""Build docs/data/intensity.json: predicted concentration-intensity curves from the
MixInt single-molecule intensity network (Pellegrino et al., bioRxiv 2025), keyed by
InChIKey.

The network (net_127_features.onnx, trained by K. Samoilova in MATLAB) is an MLP:
127 inputs -> 4 x (300, BatchNorm, ReLU) -> 1. Inputs are 126 molecular features
(Dragon 6 descriptors, EPI-style VP/BP, MATLAB alpha-shape volume/area; the ones flagged
in descriptor_list_127.csv are natural-logged, non-finite -> 0) followed by log10 of the
vapor-phase concentration. Output is perceived intensity on the 0-100 rating scale.

Dragon 3D descriptors cannot be computed in the browser, so the curves are evaluated
here, over a fine log-concentration grid, for every molecule that has Dragon features;
the client interpolates. Because the network is piecewise-linear in log C (ReLU), linear
interpolation on the grid is near-exact (the build reports the maximum error).

Feature provenance, in priority order:
  Dragon   : the MixInt training table for the 62 training odorants (exactly what the
             network saw), otherwise AllDragon_251125.csv.
  VP / BP  : the curated training values for the 62, otherwise CompTox OPERA_VP/OPERA_BP
             (scripts/fetch_comptox_physchem.py). Molecules with no VP are left out --
             VP is the one physchem input the predictions are sensitive to.
  alpha-*  : the MATLAB alpha-shape values from the earlier features.csv export,
             otherwise imputed by a linear fit on Dragon size descriptors. The network
             is insensitive to them.

Only the alpha columns are taken from that features.csv: its VP/BP columns are
misaligned with the molecules, best_vp there is unlogged, and MLOGP is logged where
positive, so everything else is rebuilt from the sources above.

--export-features writes the finished (transformed) 126-column matrix for a list of
CIDs, in the layout of that features.csv, plus a provenance table."""
import argparse, json, os
from datetime import date

import numpy as np
import onnxruntime as ort
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

GDRIVE = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-jmainland@monell.org/.shortcut-targets-by-id/"
    "1u-9Ji-bhJx47CVZcSzAe89bu3e6e2XKU/MixInt/main/data/processed")
DROPBOX = os.path.expanduser("~/Monell Dropbox/Mainland Lab Team Folder/Projects")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{GDRIVE}/final_model/net_127_features.onnx")
ap.add_argument("--descriptors", default=f"{GDRIVE}/final_model/descriptor_list_127.csv")
ap.add_argument("--alpha-features", default=f"{GDRIVE}/final_model/features.csv")
ap.add_argument("--training", default=f"{GDRIVE}/mixint/features/mixint_dragon_tranport.csv")
ap.add_argument("--observed", default=f"{GDRIVE}/mixint/mixint-single-behavior.csv",
                help="panel ratings, used only to validate the build (never published)")
ap.add_argument("--dragon", default=f"{DROPBOX}/Google Mixtures/data/processed/chemoinfo/AllDragon_251125.csv")
ap.add_argument("--physchem", default="scripts/data/comptox_physchem.csv")
ap.add_argument("--output", default="docs/data/intensity.json")
ap.add_argument("--export-features", metavar="PREFIX",
                help="also write PREFIX.csv (model-ready features) and PREFIX_provenance.csv "
                     "for the CIDs in --alpha-features")
args = ap.parse_args()

LO, HI, STEP = -12.0, 0.0, 0.25     # log10 concentration grid
TRAIN_LO, TRAIN_HI = -10.4, -3.5    # range of concentrations the panel rated

# --- model inputs ---------------------------------------------------------------
spec = pd.read_csv(args.descriptors, encoding="utf-8-sig")
names = [n.strip("'") for n in spec.descriptor]
use_log = spec.apply_log.astype(bool).values
DRAGON = [n for n in names if n not in ("best_vp", "best_bp", "alpha-vol", "alpha-area")]
assert len(names) == 126

dr = pd.read_csv(args.dragon, low_memory=False)
dr["cid"] = dr.NAME.str.split(":").str[0].astype(int)
dr["smiles"] = dr.NAME.str.split(": ", n=1).str[1]
dr = dr.set_index("cid")

tr = pd.read_csv(args.training, low_memory=False).rename(columns={"NAME": "cid"}).set_index("cid")
kf = pd.read_csv(args.alpha_features).set_index("cid")
pc = pd.read_csv(args.physchem).set_index("cid")
print(f"AllDragon {len(dr)} molecules; training {len(tr)}; CompTox rows {len(pc)}")

X = dr[DRAGON].apply(pd.to_numeric, errors="coerce")
X.loc[X.index.intersection(tr.index)] = tr[DRAGON].apply(pd.to_numeric, errors="coerce")

# VP / BP
vp = pc.vp_mmHg.reindex(X.index)
bp = pc.bp_C.reindex(X.index)
vsrc = pc.vp_source.reindex(X.index).map({"experimental": "e", "predicted": "p"})
t_ids = X.index.intersection(tr.index)
vp.loc[t_ids], bp.loc[t_ids], vsrc.loc[t_ids] = tr.best_vp[t_ids], tr.best_bp[t_ids], "t"
X["best_vp"], X["best_bp"] = vp, bp

# alpha shapes: exported MATLAB values where finite, else a linear fit on size descriptors
SIZE = ["MW", "Vx", "SAtot", "nSK", "nBT", "RBN", "nAB"]
S = dr[SIZE].apply(pd.to_numeric, errors="coerce").fillna(0)
S = np.column_stack([np.ones(len(S)), np.log(S.MW), np.log(S.Vx.clip(lower=1e-3)), S.drop(columns=["MW", "Vx"]).values])
S = pd.DataFrame(S, index=dr.index)
for a in ("alpha-vol", "alpha-area"):
    known = np.log(np.exp(kf[a]).reindex(X.index))       # kf holds ln(alpha); -inf where alpha=0
    known = known[np.isfinite(known)]
    coef, *_ = np.linalg.lstsq(S.loc[known.index].values, known.values, rcond=None)
    fit = S.values @ coef
    r2 = 1 - ((S.loc[known.index].values @ coef - known) ** 2).sum() / ((known - known.mean()) ** 2).sum()
    col = pd.Series(np.exp(fit), index=X.index)          # stored raw; logged below
    col.loc[known.index] = np.exp(known)
    X[a] = col
    print(f"{a}: {len(known)} from features.csv, {len(X) - len(known)} imputed (fit R2={r2:.2f})")

X = X[names]
keep = X.best_vp.notna() & (X.best_vp > 0)
print(f"dropping {(~keep).sum()} molecules with no vapor pressure")
X, vsrc = X[keep], vsrc[keep]

F = X.values.astype(np.float64)
with np.errstate(divide="ignore", invalid="ignore"):
    F[:, use_log] = np.log(F[:, use_log])
F[~np.isfinite(F)] = 0.0                                  # the training convention

if args.export_features:
    want = kf.index
    ok = want[want.isin(X.index)]
    pd.DataFrame(F[X.index.get_indexer(ok)], index=pd.Index(ok, name="cid"),
                 columns=names).to_csv(args.export_features + ".csv")
    prov = pd.DataFrame({"cid": want, "in_model_features": want.isin(X.index),
                         "vp_mmHg": vp.reindex(want).values, "bp_C": bp.reindex(want).values,
                         "vp_source": pc.vp_source.reindex(want).values,
                         "dtxsid": pc.dtxsid.reindex(want).values})
    prov.loc[prov.cid.isin(tr.index), "vp_source"] = "MixInt curated (EPI Suite)"
    prov.to_csv(args.export_features + "_provenance.csv", index=False)
    print(f"exported features for {len(ok)}/{len(want)} CIDs -> {args.export_features}.csv")

# --- evaluate ---------------------------------------------------------------------
sess = ort.InferenceSession(args.model)
def predict(feats, logc):
    """feats (n,126), logc (m,) -> (n,m) intensities"""
    n, m = len(feats), len(logc)
    inp = np.concatenate([np.repeat(feats, m, 0), np.tile(logc, n)[:, None]], 1).astype(np.float32)
    out = np.concatenate([sess.run(None, {"input": inp[i:i + 200000]})[0]
                          for i in range(0, len(inp), 200000)])
    return out.reshape(n, m)

grid = np.round(np.arange(LO, HI + STEP / 2, STEP), 6)
Y = predict(F, grid)

# interpolation error vs. a 10x finer grid, over a sample
rng = np.random.default_rng(0)
samp = rng.choice(len(F), size=min(500, len(F)), replace=False)
fine = np.round(np.arange(LO, HI + 1e-9, STEP / 10), 6)
interp = np.array([np.interp(fine, grid, np.round(Y[i], 1)) for i in samp])
err = np.abs(predict(F[samp], fine) - interp)
print(f"grid interpolation error: max {err.max():.2f}, 99th pct {np.percentile(err, 99):.2f} intensity units")

# --- validation against the panel (build-time only) -----------------------------
# In-sample: these are the network's training data, so this checks the port (features,
# transforms, units), not generalization. Expect r ~0.99, RMSE ~2.3 on raw Intensity.
obs = (pd.read_csv(args.observed).groupby(["Odor", "Concentration"]).Intensity.mean()
       .rename("mean").reset_index())
odor_cid = tr.reset_index().set_index("Odor").cid
obs["cid"] = obs.Odor.map(odor_cid)
obs = obs[obs.cid.isin(X.index)]
rows = X.index.get_indexer(obs.cid)
p = np.array([predict(F[r:r + 1], np.array([np.log10(c)]))[0, 0]
              for r, c in zip(rows, obs.Concentration)])
y = obs["mean"].values
print(f"validation vs panel means: {obs.Odor.nunique()} odorants, {len(y)} points, "
      f"r={np.corrcoef(p, y)[0, 1]:.3f}, RMSE={np.sqrt(((p - y) ** 2).mean()):.2f}")

# --- write ------------------------------------------------------------------------
mols, seen = [], set()
order = sorted(range(len(X)), key=lambda i: X.index[i] not in tr.index)   # training first
for i in order:
    cid = int(X.index[i])
    m = Chem.MolFromSmiles(str(dr.smiles[cid]))
    if m is None:
        continue
    ik = Chem.MolToInchiKey(m)
    if ik in seen:
        continue
    seen.add(ik)
    rec = {"i": ik, "c": Chem.MolToSmiles(m), "cid": cid,
           "y": [int(v) for v in np.round(np.clip(Y[i], 0, None) * 10)],
           "vp": float(f"{X.best_vp.iat[i]:.4g}"), "vs": vsrc.iat[i]}
    if cid in tr.index:
        rec["t"] = 1
    mols.append(rec)

out = {
    "meta": {
        "n": len(mols), "built": date.today().isoformat(),
        "grid": {"lo": LO, "step": STEP, "n": len(grid)},
        "train": {"lo": TRAIN_LO, "hi": TRAIN_HI, "n": int(len(tr))},
        "scale": "0-100 intensity rating; stored x10",
        "conc": "log10 vapor-phase concentration (v/v in air)",
        "vs": {"t": "MixInt curated (EPI Suite)", "e": "CompTox experimental",
               "p": "CompTox OPERA prediction"},
        "source": "Pellegrino et al. 2025, bioRxiv 10.1101/2025.08.08.668954 (preprint)",
    },
    "mols": mols,
}
with open(args.output, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"wrote {args.output}: {len(mols)} molecules, {os.path.getsize(args.output) / 1e6:.2f} MB; "
      f"VP sources {pd.Series([m['vs'] for m in mols]).value_counts().to_dict()}")
