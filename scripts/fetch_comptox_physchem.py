#!/usr/bin/env python3
"""Fetch vapor pressure, boiling point and logP from the EPA CompTox (CTX) API.

The intensity model was trained on EPI Suite's `best_vp` / `best_bp` (experimental
where available, MPBPVP estimate otherwise). CTX's OPERA_VP / OPERA_BP records follow
the same convention -- propValue is the curated experimental value when OPERA has one
and its QSAR prediction otherwise -- so we take those, and report which it was.
OPERA_LogP (octanol-water) follows the same rule; it feeds the transport model.

Input: a CSV with `smiles` (and any id columns, passed through). Output: one row per
input with inchikey, dtxsid, vp_mmHg, bp_C, logp and *_source in {experimental, predicted}.

API key: $CTX_API_KEY, else ~/.config/comptox/api_key. It is only ever sent to
comptox.epa.gov in the x-api-key header. Responses are cached (keyed by InChIKey /
DTXSID) so reruns only query what is new."""
import argparse, json, os, sys, time
import urllib.request, urllib.error

import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

BASE = "https://comptox.epa.gov/ctx-api"   # api-ccte.epa.gov no longer resolves
BATCH = 200
MODELS = {"OPERA_VP": "vp", "OPERA_BP": "bp", "OPERA_LogP": "logp"}
WANT = sorted(MODELS.values())   # stored per DTXSID so adding a model refetches old entries

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--output", required=True)
ap.add_argument("--cache", default="scripts/cache/comptox.json")
args = ap.parse_args()

key = os.environ.get("CTX_API_KEY")
if not key:
    p = os.path.expanduser("~/.config/comptox/api_key")
    if not os.path.exists(p):
        sys.exit("no API key: set CTX_API_KEY or write it to ~/.config/comptox/api_key")
    key = open(p).read().strip()


def post(path, body, ctype):
    data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    for attempt in range(5):
        req = urllib.request.Request(BASE + path, data=data, method="POST", headers={
            "x-api-key": key, "Content-Type": ctype, "accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** attempt
            print(f"  {path}: {e}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {path}")


cache = {"dtxsid": {}, "props": {}}
if os.path.exists(args.cache):
    cache = json.load(open(args.cache))
os.makedirs(os.path.dirname(args.cache), exist_ok=True)
save = lambda: json.dump(cache, open(args.cache, "w"))

df = pd.read_csv(args.input)
def ikey(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m) if m else None
df["inchikey"] = df.smiles.map(ikey)
keys = sorted({k for k in df.inchikey.dropna()})

# --- InChIKey -> DTXSID. Exact match first; a stereo-blind retry on the skeleton
# (first block + UHFFFAOYSA) catches inputs whose stereo differs from the DSSTox record.
todo = [k for k in keys if k not in cache["dtxsid"]]
print(f"{len(keys)} InChIKeys, {len(todo)} to resolve")
for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    for hit in post("/chemical/search/equal/", "\n".join(chunk), "text/plain"):
        cache["dtxsid"][hit["searchValue"]] = hit.get("dtxsid")
    for k in chunk:
        cache["dtxsid"].setdefault(k, None)
    save()
    print(f"  resolved {min(i + BATCH, len(todo))}/{len(todo)}")

flat = {k: k.split("-")[0] + "-UHFFFAOYSA-N" for k in keys
        if cache["dtxsid"].get(k) is None and not k.endswith("-UHFFFAOYSA-N")}
retry = [f for f in set(flat.values()) if f not in cache["dtxsid"]]
for i in range(0, len(retry), BATCH):
    for hit in post("/chemical/search/equal/", "\n".join(retry[i:i + BATCH]), "text/plain"):
        cache["dtxsid"][hit["searchValue"]] = hit.get("dtxsid")
    for k in retry[i:i + BATCH]:
        cache["dtxsid"].setdefault(k, None)
    save()

def resolve(k):
    d = cache["dtxsid"].get(k)
    return (d, "exact") if d else ((cache["dtxsid"].get(flat[k]), "stereo-blind")
                                   if k in flat and cache["dtxsid"].get(flat[k]) else (None, None))

# --- DTXSID -> OPERA VP / BP / logP -------------------------------------------------------
ids = sorted({resolve(k)[0] for k in keys} - {None})
todo = [d for d in ids if cache["props"].get(d, {}).get("_m") != WANT]
print(f"{len(ids)} DTXSIDs, {len(todo)} to fetch properties for")
for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    got = {d: {"_m": WANT} for d in chunk}
    for rec in post("/chemical/property/predicted/search/by-dtxsid/", chunk, "application/json"):
        short = MODELS.get(rec.get("modelName"))
        if short and rec["dtxsid"] in got:
            got[rec["dtxsid"]][short] = [rec.get("propValue"), rec.get("propValueExperimental")]
    cache["props"].update(got)
    save()
    print(f"  fetched {min(i + BATCH, len(todo))}/{len(todo)}")

# --- assemble ----------------------------------------------------------------------
out = []
for k in df.inchikey:
    d, how = resolve(k) if k else (None, None)
    row = {"inchikey": k, "dtxsid": d, "match": how}
    p = cache["props"].get(d, {}) if d else {}
    for short in ("vp", "bp", "logp"):
        val, exp = p.get(short, [None, None])
        row[f"{short}"] = val
        row[f"{short}_source"] = None if val is None else (
            "experimental" if exp is not None and abs(exp - val) <= 1e-9 * max(1, abs(val))
            else "predicted")
    out.append(row)
res = pd.concat([df.reset_index(drop=True), pd.DataFrame(out).drop(columns="inchikey")], axis=1)
res = res.rename(columns={"vp": "vp_mmHg", "bp": "bp_C"})
res.to_csv(args.output, index=False)
print(f"wrote {args.output}: {res.dtxsid.notna().sum()}/{len(res)} matched, "
      f"{res.vp_mmHg.notna().sum()} with VP ({(res.vp_source == 'experimental').sum()} experimental)")
