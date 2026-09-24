"""Shared vapor pressure / boiling point / logP resolution for the build scripts, so the
odor, toxicity and intensity modules all rank sources the same way.

Inputs are the per-molecule CSVs from fetch_comptox_physchem.py (CompTox OPERA) and
fetch_episuite_physchem.py (EPI Suite). Measured beats predicted; within each, the
molecule's own record beats a stereoisomer's (same InChIKey skeleton -- VP and logP are
stereo-insensitive, siblings differ <=0.7 log units):

  VP   : CompTox exp  > EPI exp  > CompTox exp (stereoisomer)
         > OPERA pred > OPERA pred (stereoisomer) > MPBPVP estimate
  logP : CompTox exp  > EPI exp  > OPERA pred > KOWWIN estimate   (stereo-blind)

EPI's own experimental lookup already falls back to a stereoisomer's CAS record; that is
flagged too. OPERA predictions are kept ahead of MPBPVP estimates so values already on the
site only change when a measurement turns up: OPERA reproduces PhysProp for its training
chemicals, and the two predictors disagree by ~0.3-0.8 log units in the odorant range.

Source codes (also the client's): e = CompTox experimental, p = CompTox OPERA prediction,
x = EPI Suite experimental, m = EPI Suite MPBPVP estimate, k = EPI Suite KOWWIN estimate.
"""
import math

import pandas as pd

SRC = {"e": "CompTox experimental",
       "p": "CompTox OPERA prediction",
       "x": "EPI Suite experimental",
       "m": "EPI Suite estimate (MPBPVP)",
       "k": "EPI Suite estimate (KOWWIN)"}
PREDICTED = {"p", "m", "k"}


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _rows(df, val, src, codes, positive=False):
    """inchikey -> (value, code, bp) for rows with a usable value; first row wins."""
    out = {}
    if val not in df or src not in df:
        return out
    for k, v, s, bp in zip(df.inchikey, df[val], df[src], df.get("bp_C", [None] * len(df))):
        if isinstance(k, str) and isinstance(s, str) and _ok(v) and (v > 0 or not positive):
            out.setdefault(k, (float(v), codes[s], bp if _ok(bp) else None))
    return out


def _by_skeleton(d, code):
    """skeleton -> (value, code, bp) among records of the given source code."""
    out = {}
    for k, rec in d.items():
        if rec[1] == code:
            out.setdefault(k[:14], rec)
    return out


def resolve(keys, comptox, episuite):
    """keys: iterable of InChIKeys. Returns {inchikey: {vp, vs, st, bp, logp, ls}} with only
    the fields that were found. st=True when the VP was borrowed from a stereoisomer."""
    ct_vp = _rows(comptox, "vp_mmHg", "vp_source", {"experimental": "e", "predicted": "p"}, True)
    ep_vp = _rows(episuite, "vp_mmHg", "vp_source", {"experimental": "x", "estimated": "m"}, True)
    ep_stereo = {k for k, m in zip(episuite.inchikey, episuite.match) if m == "stereo-blind"}
    ct_sk_e, ct_sk_p = _by_skeleton(ct_vp, "e"), _by_skeleton(ct_vp, "p")
    ct_lp = _rows(comptox, "logp", "logp_source", {"experimental": "e", "predicted": "p"})
    ep_lp = _rows(episuite, "logkow", "logkow_source", {"experimental": "x", "estimated": "k"})
    lp_sk = {}
    for d in (ct_lp, ep_lp):
        for code in ("e", "x", "p", "k"):
            for sk, rec in _by_skeleton(d, code).items():
                lp_sk.setdefault(code, {})[sk] = rec

    ep_bp = {k: b for k, b in zip(episuite.inchikey, episuite.bp_C) if isinstance(k, str) and _ok(b)}
    out = {}
    for k in keys:
        if not isinstance(k, str) or k in out:
            continue
        sk, rec = k[:14], {}
        own_ct, own_ep = ct_vp.get(k), ep_vp.get(k)
        for cand, st in ((own_ct if own_ct and own_ct[1] == "e" else None, False),
                         (own_ep if own_ep and own_ep[1] == "x" else None, k in ep_stereo),
                         (ct_sk_e.get(sk), True),
                         (own_ct if own_ct and own_ct[1] == "p" else None, False),
                         (ct_sk_p.get(sk), True),
                         (own_ep, False)):
            if cand:
                rec["vp"], rec["vs"], rec["st"] = cand[0], cand[1], st
                bp = cand[2] if cand[2] is not None else ep_bp.get(k)
                if bp is not None:
                    rec["bp"] = float(bp)
                break
        for code in ("e", "x", "p", "k"):
            own = (ct_lp if code in "ep" else ep_lp).get(k)
            cand = own if own and own[1] == code else lp_sk.get(code, {}).get(sk)
            if cand:
                rec["logp"], rec["ls"] = cand[0], code
                break
        if rec:
            out[k] = rec
    return out


def load(comptox_csv, episuite_csv):
    return pd.read_csv(comptox_csv), pd.read_csv(episuite_csv)
