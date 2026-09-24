#!/usr/bin/env python3
"""Vapor pressure, boiling point, melting point and logKow from EPA EPI Suite (MPBPVP +
KOWWIN), run locally from the EPI Suite CLI JAR -- nothing is sent over the network.

EPI Suite is the source the intensity network was trained on (`best_vp` / `best_bp`:
experimental where available, MPBPVP estimate otherwise), and unlike CompTox it estimates
any parseable SMILES, so it fills the molecules CompTox has no record for.

Experimental values live in the JAR's embedded PhysProp-derived database and are only used
when the request carries a CAS number, so each input InChIKey is first mapped to a CAS via
the JAR's own identity table (search.parquet): exact InChIKey, else -- only when that CAS has
measured VP/BP/MP/logKow -- a stereoisomer's (same InChIKey skeleton). Otherwise the input
SMILES is estimated directly. Measured BP/MP also feed MPBPVP's VP estimate.

Setup (the JAR is ~40 MB, gitignored):
  curl -o scripts/vendor/episuite/epi-estimators-cli.jar https://episuite.dev/api/download
  (cd scripts/vendor/episuite && unzip -o epi-estimators-cli.jar 'webroot/*.parquet' \
       webroot/dataset-manifest.json)
Runs on Java 8+.

Input: a CSV with `smiles` (other columns pass through). Output: one row per input with
inchikey, cas, match, vp_mmHg, bp_C, mp_C, logkow and *_source in {experimental, estimated},
plus vp_ref (the citation for measured VPs). Results are cached per request."""
import argparse, json, os, subprocess, sys

import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

VENDOR = "scripts/vendor/episuite"
MODULES = ["physicalProperties", "logKow"]
PROPS = {"vaporPressure": "vp", "boilingPoint": "bp", "meltingPoint": "mp", "logKow": "logkow"}
CHUNK = 2000

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--output", required=True)
ap.add_argument("--jar", default=f"{VENDOR}/epi-estimators-cli.jar")
ap.add_argument("--webroot", default=f"{VENDOR}/webroot")
ap.add_argument("--cache", default="scripts/cache/episuite.json")
ap.add_argument("--java", default="java")
args = ap.parse_args()

for f in (args.jar, f"{args.webroot}/data.parquet", f"{args.webroot}/search.parquet",
          f"{args.webroot}/dataset-manifest.json"):
    if not os.path.exists(f):
        sys.exit(f"missing {f} -- see the setup notes at the top of this script")


def ikey(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m) if m else None


# --- InChIKey -> CAS, from the JAR's identity table (cached; ~1 min to build) -------------
idx_path = os.path.join(os.path.dirname(args.cache), "episuite_cas_index.csv")
os.makedirs(os.path.dirname(args.cache), exist_ok=True)
if os.path.exists(idx_path):
    idx = pd.read_csv(idx_path)
else:
    print("indexing EPI Suite identities by InChIKey ...")
    s = pd.read_parquet(f"{args.webroot}/search.parquet",
                        columns=["cas", "canonical_smiles", "chemical_id"]).dropna()
    s = s.drop_duplicates("cas")
    s["inchikey"] = s.canonical_smiles.map(ikey)
    d = pd.read_parquet(f"{args.webroot}/data.parquet", columns=["chemical_id", "property_code"])
    d = d[d.property_code.isin(["vapor_pressure_mm_hg", "boiling_point_c",
                                "melting_point_c", "log_kow"])]
    s["n_exp"] = s.chemical_id.map(d.groupby("chemical_id").size()).fillna(0).astype(int)
    s["has_vp"] = s.chemical_id.isin(d[d.property_code == "vapor_pressure_mm_hg"].chemical_id)
    idx = s.dropna(subset=["inchikey"])[["inchikey", "cas", "n_exp", "has_vp"]]
    idx.to_csv(idx_path, index=False)


def valid_cas(c):
    """~5% of the identity table are internal placeholders that fail the CAS checksum;
    the CLI rejects them and aborts the whole batch, so drop them."""
    parts = str(c).split("-")
    d = "".join(parts)
    if len(parts) != 3 or not d.isdigit():
        return False
    return sum((i + 1) * int(x) for i, x in enumerate(reversed(d[:-1]))) % 10 == int(d[-1])


idx = idx[idx.cas.map(valid_cas)]
# several CAS can share a structure (CAS for a mixture of isomers, a deleted number...):
# take the one with a measured VP, then the most measured properties
idx = idx.sort_values(["has_vp", "n_exp"], ascending=False)
exact = idx.drop_duplicates("inchikey").set_index("inchikey")
exp_only = idx[idx.n_exp > 0].assign(sk=lambda x: x.inchikey.str[:14])
# stereo-blind: prefer the stereo-unspecified record, then measured VP, then most data
exp_only = exp_only.assign(flat=exp_only.inchikey.str.endswith("-UHFFFAOYSA-N"))
skel = (exp_only.sort_values(["flat", "has_vp", "n_exp"], ascending=False)
        .drop_duplicates("sk").set_index("sk"))

df = pd.read_csv(args.input)
df["inchikey"] = df.smiles.map(ikey)


def request(smiles, k):
    if k in exact.index:
        return {"cas": exact.cas[k], "modules": MODULES}, exact.cas[k], "exact"
    if k and k[:14] in skel.index:
        return {"cas": skel.cas[k[:14]], "modules": MODULES}, skel.cas[k[:14]], "stereo-blind"
    return {"smiles": str(smiles), "modules": MODULES}, None, None


reqs = [request(s, k) for s, k in zip(df.smiles, df.inchikey)]
keyed = [json.dumps(r[0], sort_keys=True) for r in reqs]

# --- run the CLI over whatever isn't cached ----------------------------------------------
cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
todo = sorted(set(keyed) - cache.keys())
print(f"{len(df)} rows, {sum(r[1] is not None for r in reqs)} mapped to a CAS, "
      f"{len(todo)} requests to run")


def measured(sv):
    """The embedded database is not all measurements: VP and logKow records carry a PhysProp
    method code (EXP measured, EXT extrapolated from measurements, EST estimated), and ~5%
    of them cite "EPI SUITE v4.x" -- old EPI estimates filed as data. BP/MP have no code."""
    src = sv.get("source") or ""
    if not src.startswith("experimental:"):
        return False
    if "EPISUITE" in src.upper().replace(" ", ""):
        return False
    table = (src.split(":") + ["", "", ""])[2]     # experimental:EpiUnified:<table>:<ref>
    return table in ("BoilingPoints", "MeltingPoints") or sv.get("method") in ("EXP", "EXT")


def extract(res):
    out = {}
    for name, short in PROPS.items():
        sv = ((res or {}).get(name) or {}).get("selectedValue")
        if sv and sv.get("value") is not None:
            src = sv.get("source") or ""
            out[short] = [sv["value"], "experimental" if measured(sv) else "estimated",
                          src.split(":", 3)[3] if short == "vp" and src.count(":") >= 3 else None]
    return out


for i in range(0, len(todo), CHUNK):
    chunk = todo[i:i + CHUNK]
    p = subprocess.run([args.java, "-jar", args.jar, "--compact", "--jsonl",
                        "--experimental-data", f"{args.webroot}/data.parquet",
                        "--request-json", "-"],
                       input="\n".join(chunk), capture_output=True, text=True)
    lines = p.stdout.strip().split("\n")
    if len(lines) != len(chunk):
        sys.exit(f"CLI returned {len(lines)} lines for {len(chunk)} requests:\n{p.stderr[-2000:]}")
    for k, line in zip(chunk, lines):
        try:
            cache[k] = extract(json.loads(line))
        except json.JSONDecodeError:
            cache[k] = {}
    json.dump(cache, open(args.cache, "w"))
    print(f"  ran {min(i + CHUNK, len(todo))}/{len(todo)}")

# --- assemble ----------------------------------------------------------------------------
def has_carbon(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return m is None or any(a.GetAtomicNum() == 6 for a in m.GetAtoms())


# MPBPVP/KOWWIN are fragment methods for organics; on carbon-free species they are nonsense
# (HCl ~4e-8 mmHg, really ~3.5e4; EPI still flags it "organic"), so keep only measured values.
rows = []
for (req, cas, how), k, smi in zip(reqs, keyed, df.smiles):
    r = {"cas": cas, "match": how}
    got = cache.get(k, {})
    organic = has_carbon(smi)
    for short in PROPS.values():
        val, src, ref = got.get(short, [None, None, None])
        if src == "estimated" and not organic:
            val, src, ref = None, None, None
        r[short], r[f"{short}_source"] = val, src
        if short == "vp":
            r["vp_ref"] = ref
    rows.append(r)
res = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
res = res.rename(columns={"vp": "vp_mmHg", "bp": "bp_C", "mp": "mp_C"})
res.to_csv(args.output, index=False)
print(f"wrote {args.output}: {res.vp_mmHg.notna().sum()}/{len(res)} with VP "
      f"({(res.vp_source == 'experimental').sum()} experimental), "
      f"{res.logkow.notna().sum()} with logKow ({(res.logkow_source == 'experimental').sum()} experimental)")
