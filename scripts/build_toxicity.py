#!/usr/bin/env python3
"""Build docs/data/toxicity.json: Toxtree hazard classes for every molecule the site knows.

Input: scripts/data/toxtree_site.csv from scripts/run_toxtree.sh (Revised Cramer, ISS Ames
alerts and the Gradient Supplemental Rules over scripts/data/site_molecules.csv).

Only the classes are stored. Vapor pressure is NOT baked in: the client reads it from
physchem.json (CompTox) at runtime, so every module shares one VP; MW comes from RDKit.js.

Output record: {"i": inchikey, "c": Revised Cramer "I"|"II"|"III",
                "a": [fired Ames alert ids] (present only if an alert fired),
                "g": Gradient result "m" (mutagen) | "I"|"II"|"III" (present only if assigned)}.

TTC precedence (applied client-side, matches the R app's README/get_ttc):
  Gradient mutagen or Ames alert -> 1.5 > Gradient class > Revised Cramer class.
Several SMILES can share an InChIKey (tautomers, salts written differently); when their
classes disagree, the most conservative (lowest-TTC) record is kept."""
import csv, json
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TTC = {"I": 1800, "II": 540, "III": 90}

# Titles as reported by Toxtree 3.1.0 (AmesMutagenicityRules.getRule(i).getTitle()).
SA_TITLES = {
    "SA1_Ames": "Acyl halides",
    "SA2_Ames": "Alkyl (C<5) or benzyl ester of sulphonic or phosphonic acid",
    "SA3_Ames": "N-methylol derivatives",
    "SA4_Ames": "Monohaloalkene",
    "SA5_Ames": "S or N mustard",
    "SA6_Ames": "Propiolactones and propiosultones",
    "SA7_Ames": "Epoxides and aziridines",
    "SA8_Ames": "Aliphatic halogens",
    "SA9_Ames": "Alkyl nitrite",
    "SA10_Ames": "α,β-unsaturated carbonyls",
    "SA11_Ames": "Simple aldehyde",
    "SA12_Ames": "Quinones",
    "SA13_Ames": "Hydrazine",
    "SA14_Ames": "Aliphatic azo and azoxy",
    "SA15_Ames": "Isocyanate and isothiocyanate groups",
    "SA16_Ames": "Alkyl carbamate and thiocarbamate",
    "SA18_Ames": "Polycyclic aromatic hydrocarbons",
    "SA19_Ames": "Heterocyclic polycyclic aromatic hydrocarbons",
    "SA21_Ames": "Alkyl and aryl N-nitroso groups",
    "SA22_Ames": "Azide and triazene groups",
    "SA23_Ames": "Aliphatic N-nitro",
    "SA24_Ames": "α,β-unsaturated alkoxy",
    "SA25_Ames": "Aromatic nitroso group",
    "SA26_Ames": "Aromatic ring N-oxide",
    "SA27_Ames": "Nitro aromatic",
    "SA28_Ames": "Primary aromatic amine, hydroxylamine or derived ester",
    "SA28bis_Ames": "Aromatic mono- and dialkylamine",
    "SA28ter_Ames": "Aromatic N-acyl amine",
    "SA29_Ames": "Aromatic diazo",
    "SA30_Ames": "Coumarins and furocoumarins",
    "SA37_Ames": "Pyrrolizidine alkaloids",
    "SA38_Ames": "Alkenylbenzenes",
    "SA39_Ames": "Steroidal estrogens",
    "SA57_Ames": "DNA intercalating agents with a basic side chain",
    "SA58_Ames": "Haloalkene cysteine S-conjugates",
    "SA59_Ames": "Xanthones, thioxanthones, acridones",
    "SA60_Ames": "Flavonoids",
    "SA61_Ames": "Alkyl hydroperoxides",
    "SA62_Ames": "N-acyloxy-N-alkoxybenzamides",
    "SA63_Ames": "N-aryl-N-acetoxyacetamides",
    "SA64_Ames": "Hydroxamic acid derivatives",
    "SA65_Ames": "Halofuranones",
    "SA66_Ames": "Anthrones",
    "SA67_Ames": "Triphenylimidazole and related",
    "SA68_Ames": "9,10-dihydrophenanthrenes",
    "SA69_Ames": "Fluorinated quinolines",
    "aN=Na": "Aromatic diazo",
    "ar-N=CH2": "Derived aromatic amines",
    "QSAR": "α,β-unsaturated aliphatic aldehyde (TA100 QSAR)",
}


def ttc(rec):
    if rec.get("g") == "m" or rec.get("a"):
        return 1.5
    return TTC.get(rec.get("g")) or TTC.get(rec.get("c")) or float("inf")


mols, n_bad, n_dup, n_conflict = {}, 0, 0, 0
for r in csv.DictReader(open("scripts/data/toxtree_site.csv")):
    m = Chem.MolFromSmiles(r["smiles"])
    k = Chem.MolToInchiKey(m) if m is not None else ""
    if not k:
        n_bad += 1
        continue
    rec: dict = {"i": k}
    if r["cramer"]:
        rec["c"] = r["cramer"]
    if r["ames_alert"] == "1":
        rec["a"] = [s for s in r["ames_sa"].split(";") if s in SA_TITLES] or ["?"]
    if r["gradient"] == "mutagen":
        rec["g"] = "m"
    elif r["gradient"] in TTC:
        rec["g"] = r["gradient"]
    if len(rec) == 1:              # Toxtree could not process it
        n_bad += 1
        continue
    if k in mols:
        n_dup += 1
        if ttc(rec) != ttc(mols[k]):
            n_conflict += 1
        if ttc(rec) >= ttc(mols[k]):
            continue
    mols[k] = rec

used = sorted({s for r in mols.values() for s in r.get("a", [])} - {"?"})
out = {"meta": {"n": len(mols), "tool": "Toxtree 3.1.0",
                "trees": ["Revised Cramer Decision Tree",
                          "In vitro mutagenicity (Ames test) alerts by ISS",
                          "Gradient Supplemental Rules"],
                "sa": {s: SA_TITLES[s] for s in used}},
       "mols": sorted(mols.values(), key=lambda r: r["i"])}
json.dump(out, open("docs/data/toxicity.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(f"wrote docs/data/toxicity.json: {len(mols)} molecules "
      f"({n_bad} unparsable/unclassified, {n_dup} duplicate InChIKeys, {n_conflict} with conflicting TTC)")
