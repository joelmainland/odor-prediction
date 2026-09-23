#!/usr/bin/env bash
# Run the three Toxtree decision trees the toxicity module uses over a SMILES list and
# merge them into scripts/data/toxtree_site.csv (input for build_toxicity.py).
#
#   Revised Cramer Decision Tree            toxtree.tree.cramer3.RevisedCramerDecisionTree
#   In vitro mutagenicity (Ames) by ISS     toxtree.plugins.ames.AmesMutagenicityRules
#   Gradient Supplemental Rules             user-defined tree (.tml), via toxtree/TmlHeadless.java
#
# Same trees as the R app (/Users/jmainland/Documents/R/ToxicologyScreen); verified to
# reproduce its toxdata.csv on all 20 molecules.
#
# Toxtree 3.1.0 is NOT vendored in git (69 MB). Get it from
#   https://sourceforge.net/projects/toxtree/files/toxtree/Toxtree-v.3.1.0/
# (the zip, or unpack the -setup.exe with `7zz x`) and put Toxtree-3.1.0.1851.jar, ext/
# and toxtree-plugins.properties in scripts/vendor/toxtree/. Needs Java 8.
#
# usage: scripts/run_toxtree.sh [smiles.csv] [out.csv]
#   smiles.csv: one column with header, default scripts/data/site_molecules.csv
#   GRADIENT_TML overrides the location of Gradient_Supplemental_Tree.tml
set -euo pipefail
cd "$(dirname "$0")/.."

IN="${1:-scripts/data/site_molecules.csv}"
OUT="${2:-scripts/data/toxtree_site.csv}"
TT=scripts/vendor/toxtree
JAR="$TT/Toxtree-3.1.0.1851.jar"
TML="${GRADIENT_TML:-$HOME/Monell Dropbox/Mainland Lab Team Folder/Tox Screening/Gradient_Supplemental_Tree.tml}"
WORK=scripts/cache/toxtree
mkdir -p "$WORK"

[ -f "$JAR" ] || { echo "missing $JAR (see header)"; exit 1; }
[ -f "$TML" ] || { echo "missing Gradient tree: $TML (set GRADIENT_TML)"; exit 1; }

# Toxtree reads the header row as the column name; force it to SMILES.
{ echo SMILES; tail -n +2 "$IN"; } > "$WORK/in.csv"

javac -d "$WORK" -cp "$JAR" scripts/toxtree/TmlHeadless.java

# Toxtree resolves ext/ relative to the working directory, so run from $TT.
ABS_WORK="$(cd "$WORK" && pwd)"
(
  cd "$TT"
  J=java; OPTS=-Djava.awt.headless=true
  $J $OPTS -jar Toxtree-3.1.0.1851.jar -n -i "$ABS_WORK/in.csv" -o "$ABS_WORK/cramer3.csv" \
      -m toxtree.tree.cramer3.RevisedCramerDecisionTree > "$ABS_WORK/cramer3.log" 2>&1 &
  $J $OPTS -jar Toxtree-3.1.0.1851.jar -n -i "$ABS_WORK/in.csv" -o "$ABS_WORK/ames.csv" \
      -m toxtree.plugins.ames.AmesMutagenicityRules > "$ABS_WORK/ames.log" 2>&1 &
  $J $OPTS -cp "Toxtree-3.1.0.1851.jar:$ABS_WORK" TmlHeadless "$TML" \
      "$ABS_WORK/in.csv" "$ABS_WORK/gradient.csv" > "$ABS_WORK/gradient.log" 2>&1 &
  wait
)

python3 - "$WORK" "$OUT" <<'EOF'
import csv, sys
work, out = sys.argv[1], sys.argv[2]
rd = lambda f: {r["SMILES"]: r for r in csv.DictReader(open(f"{work}/{f}"))}
src = [r["SMILES"] for r in csv.DictReader(open(f"{work}/in.csv"))]
cr, am, gr = rd("cramer3.csv"), rd("ames.csv"), rd("gradient.csv")

CLASS = {"Low (Class I)": "I", "Intermediate (Class II)": "II", "High (Class III)": "III"}
def gradient(s):
    v = gr.get(s, {}).get("Gradient Supplemental Rules", "")
    if v == "Is Mutagen": return "mutagen"
    if v.startswith("No Supplemental Alert"): return "none"
    return CLASS.get(v, "")
# Ames alert columns that can fire: SA*_Ames plus the two aromatic-amine sub-alerts.
def ames_alerts(r):
    ids = [k for k, v in r.items() if v == "YES" and (k.endswith("_Ames") or k in ("aN=Na", "ar-N=CH2"))]
    if r.get("Potential S. typhimurium TA100 mutagen based on QSAR") == "YES": ids.append("QSAR")
    return ";".join(sorted(ids))

n = 0
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["smiles", "cramer", "ames_alert", "ames_sa", "gradient"])
    for s in src:
        a = am.get(s)
        # "No alerts for S. typhimurium mutagenicity" == NO means an alert fired.
        alert = "" if a is None else ("1" if a["No alerts for S. typhimurium mutagenicity"] == "NO" else "0")
        w.writerow([s, CLASS.get(cr.get(s, {}).get("RevisedCDT", ""), ""), alert,
                    ames_alerts(a) if a else "", gradient(s)])
        n += 1
print(f"wrote {out}: {n} molecules")
EOF
