/* Does it smell? — odor prediction from molecular structure.
   Implements the transport-feature models of Mayhew et al., PNAS 2022.
   All chemistry runs client-side via RDKit.js (WebAssembly). */

(function () {
  "use strict";

  // Cache-busting version tag, taken from this script's own src (app.js?v=...). GitHub
  // Pages lets browsers cache files for 10 min, so without it a visitor right after a
  // deploy can get a new index.html with an old app.js or old data tables.
  const VERSION = (() => {
    try { return new URL(document.currentScript.src).searchParams.get("v") || ""; } catch (e) { return ""; }
  })();
  const asset = (path) => (VERSION ? `${path}?v=${encodeURIComponent(VERSION)}` : path);

  // ---- Model constants (from the paper) ----
  // Rule of three: odorous if 30 <= MW <= 300 Da and heteroatoms < 4.
  const RULE_MW_MIN = 30, RULE_MW_MAX = 300, RULE_NHET_MAX = 4;
  // Transport logistic-regression boundaries in log10(vapor pressure) / logP space.
  // Odorous region lies between the two lines:
  //   low-volatility boundary (liquids/solids): logP > -1.72*log10(VP) - 9.10
  //   high-volatility boundary (gases):         logP < -1.61*log10(VP) + 8.17
  function transportBoundaries(vp) {
    const l = Math.log10(vp);
    return { low: -1.72 * l - 9.10, high: -1.61 * l + 8.17 };
  }

  // ---- Toxicity reference constants ----
  // Number of sniffs of neat headspace whose combined inhaled mass equals the
  // molecule's daily TTC. Ideal-gas headspace mass per sniff: (VP*V/RT)*MW.
  const TOX_R = 62.36367;    // L·mmHg/(K·mol)
  const TOX_T = 298.15;      // K (25 °C)
  const SNIFF_L = 0.5;       // average sniff volume, Laing 1983
  const TTC_BY_CRAMER = { I: 1800, II: 540, III: 90 }; // µg/person/day
  const CRAMER_LABEL = { I: "Class I (low)", II: "Class II (intermediate)", III: "Class III (high)" };

  let RDKit = null;
  let DATA = null;           // array of molecule records
  let byKey = new Map();     // InChIKey -> record
  let byCan = new Map();     // canonical SMILES -> record
  let byFlat = new Map();    // InChIKey skeleton (first block) -> record, stereo-blind fallback
  let TOX = null;            // Toxtree hazard classes { meta, mols }
  let byKeyTox = new Map();  // InChIKey -> { c, a, g }
  let toxPromise = null;     // lazy-load promise for toxicity.json
  let byKeyQ = new Map();    // InChIKey -> odor-quality record
  let byCanQ = new Map();    // canonical SMILES -> odor-quality record
  let byNameQ = new Map();   // lower-case name -> odor-quality record (name-lookup fallback)
  let qualPromise = null;    // lazy-load promise for quality.json (1.1 MB)
  let qSeq = 0;              // guards async quality renders against races
  let POM = null;            // OpenPOM prediction table { labels, q, mols, alt }
  let byKeyPOM = new Map();  // InChIKey -> [[labelIdx, score0_1000], ...]
  let byCanPOM = new Map();  // canonical SMILES -> same
  let pomPromise = null;     // lazy-load promise for openpom.json (2.2 MB)
  let pSeq = 0;              // guards async OpenPOM renders against races
  let INT = null;            // intensity table { meta, mols }
  let byKeyInt = new Map();  // InChIKey -> intensity record
  let byCanInt = new Map();  // canonical SMILES -> same
  let intPromise = null;     // lazy-load promise for intensity.json
  let iSeq = 0;              // guards async intensity renders against races
  let PHYS = null;           // shared VP / logP table { meta, mols }
  let byKeyPhys = new Map(); // InChIKey -> { v, vs, l, ls }
  let byFlatPhys = new Map(); // InChIKey skeleton -> best record, stereo-blind fallback
  let physPromise = null;    // lazy-load promise for physchem.json
  let aSeq = 0;              // analysis counter
  let current = null;        // evidence for the molecule on screen (see analyze)

  const $ = (id) => document.getElementById(id);
  const statusEl = () => $("status");

  function setStatus(msg, isError) {
    const s = statusEl();
    s.textContent = msg || "";
    s.classList.toggle("error", !!isError);
  }

  // ---- Boot ----
  Promise.all([
    window.initRDKitModule().then((m) => { RDKit = m; }),
    fetch(asset("data/molecules.json")).then((r) => r.json()).then((d) => {
      DATA = d;
      for (const rec of d) {
        if (rec.ikey) byKey.set(rec.ikey, rec);
        if (rec.can) byCan.set(rec.can, rec);
        // Prefer the stereo-free entry for a skeleton. No skeleton in the dataset has
        // stereoisomers with conflicting odor labels, so this fallback is safe.
        if (rec.ikey) {
          const f = rec.ikey.split("-")[0];
          if (!byFlat.has(f) || rec.ikey.endsWith("-UHFFFAOYSA-N")) byFlat.set(f, rec);
        }
      }
    }),
  ]).then(() => {
    const btn = $("predict");
    btn.disabled = false;
    btn.textContent = "Predict";
    setStatus("Ready. Enter a SMILES string or a chemical name.");
    // Shareable links: index.html#q=vanillin runs automatically.
    const h = decodeURIComponent((location.hash || "").replace(/^#q=/, ""));
    if (h && location.hash.startsWith("#q=")) { $("query").value = h; run(); }
  }).catch((e) => {
    setStatus("Failed to load chemistry engine: " + e.message, true);
  });

  // ---- Input handling ----
  $("predict").addEventListener("click", run);
  $("query").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  document.querySelectorAll(".ex").forEach((b) =>
    b.addEventListener("click", () => { $("query").value = b.dataset.q; run(); })
  );
  $("man-run").addEventListener("click", () => {
    if (!current) return;
    const mv = parseFloat($("man-vp").value), ml = parseFloat($("man-logp").value);
    current.manual = { vp: mv > 0 ? mv : null, logp: isNaN(ml) ? null : ml };
    renderProps(current);
    renderOdor(current);
    renderToxicity(current);
  });

  async function run() {
    const raw = $("query").value.trim();
    if (!raw) { setStatus("Please enter a molecule."); return; }
    if (!RDKit) { setStatus("Engine still loading…"); return; }

    setStatus("Resolving structure…");
    $("result").classList.add("hidden");

    let smiles = raw, displayName = "";
    let mol = safeMol(raw);

    // If the raw text isn't a valid SMILES, treat it as a chemical name (PubChem).
    if (!mol) {
      try {
        const res = await lookupName(raw);
        if (res && res.busy) {
          setStatus("PubChem is busy right now, so the name couldn't be looked up. Try again in a moment, or enter a SMILES string.", true);
          return;
        }
        if (!res) { setStatus(`Couldn't parse "${raw}" as SMILES or find it by name.`, true); return; }
        smiles = res.smiles;
        displayName = res.name;
        mol = safeMol(smiles);
      } catch (err) {
        setStatus("Name lookup failed (network?). Try entering a SMILES string instead.", true);
        return;
      }
    }
    if (!mol) { setStatus("Could not build a molecule from that input.", true); return; }

    try {
      analyze(mol, smiles, displayName);
    } finally {
      mol.delete();
    }
    setStatus("");
  }

  function safeMol(smi) {
    try {
      const m = RDKit.get_mol(smi);
      if (!m) return null;
      if (!m.is_valid || !m.is_valid()) { m.delete(); return null; }
      return m;
    } catch (e) { return null; }
  }

  // Name -> { smiles, name }. PubChem first; if it doesn't know the name, fall back to
  // the names in the site's own atlas table (it has trade names PubChem lacks, e.g.
  // "Tween 20"). Returns { busy: true } if PubChem is throttling and nothing local matches.
  async function lookupName(name) {
    const base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/";
    // PubChem name search ignores case; normalizing gives every spelling one URL (and one
    // cache entry). A throttled 503 was seen to stick to one capitalization's URL.
    const url = base + encodeURIComponent(name.trim().toLowerCase()) + "/property/IsomericSMILES,Title/JSON";
    let r = null;
    for (let attempt = 0; attempt < 3; attempt++) {
      r = await fetch(url);
      if (r.status !== 503) break;             // PUGREST.ServerBusy: back off and retry
      await new Promise((ok) => setTimeout(ok, 800 * (attempt + 1)));
    }
    if (!r.ok) {
      const local = await localName(name);
      return local || (r.status === 503 ? { busy: true } : null);
    }
    const j = await r.json();
    const p = j && j.PropertyTable && j.PropertyTable.Properties && j.PropertyTable.Properties[0];
    if (!p) return null;
    // PubChem has changed this column name over time; accept any SMILES variant.
    const smi = p.IsomericSMILES || p.SMILES || p.ConnectivitySMILES || p.CanonicalSMILES;
    if (!smi) return null;
    return { smiles: smi, name: p.Title || name };
  }

  async function localName(name) {
    await ensureQuality();
    const rec = byNameQ.get(name.trim().toLowerCase());
    return rec ? { smiles: rec.can, name: rec.name } : null;
  }

  // ---- Descriptor extraction ----
  function descriptors(mol) {
    let d = {};
    try { d = JSON.parse(mol.get_descriptors()); } catch (e) {}
    const mw = num(d.amw) ?? num(d.exactmw);
    const logp = num(d.CrippenClogP);
    let nhet = num(d.NumHeteroatoms);
    if (nhet == null) nhet = countHetero(mol);
    const heavy = num(d.NumHeavyAtoms);
    return { mw, logp, nhet, heavy };
  }

  function countHetero(mol) {
    try {
      const j = JSON.parse(mol.get_json());
      const atoms = j.molecules[0].atoms;
      let n = 0;
      for (const a of atoms) { const z = a.z == null ? 6 : a.z; if (z !== 1 && z !== 6) n++; }
      return n;
    } catch (e) { return null; }
  }

  // Detect species where the transport rule of three does not apply:
  // salts / mixtures (multiple fragments) and inorganic / metal-containing molecules.
  // Organic-relevant elements from the study's chemical space (GDB-17 + P).
  const ORGANIC_Z = new Set([1, 6, 7, 8, 9, 15, 16, 17, 35, 53]);
  function structuralFlags(mol, canon) {
    const salt = (canon || "").indexOf(".") >= 0;
    let inorganic = false, hasCarbon = false;
    try {
      const atoms = JSON.parse(mol.get_json()).molecules[0].atoms;
      for (const a of atoms) {
        const z = a.z == null ? 6 : a.z;
        if (z === 6) hasCarbon = true;
        if (!ORGANIC_Z.has(z)) inorganic = true;
      }
      if (!hasCarbon) inorganic = true; // no carbon => inorganic
    } catch (e) {}
    return { salt, inorganic };
  }

  function inchiKey(mol) {
    try {
      const inchi = mol.get_inchi();
      if (inchi && RDKit.get_inchikey_for_inchi) return RDKit.get_inchikey_for_inchi(inchi);
    } catch (e) {}
    return null;
  }

  // ---- Main analysis ----
  function analyze(mol, smiles, displayName) {
    const desc = descriptors(mol);
    const canon = (() => { try { return mol.get_smiles(); } catch (e) { return smiles; } })();
    const ikey = inchiKey(mol);

    // Dataset lookup (paper's curated 1,924 molecules).
    // The dataset often stores molecules without stereo (sucrose, for one), while PubChem
    // returns the stereo form -- so fall back to the InChIKey skeleton.
    let hit = (ikey && byKey.get(ikey)) || byCan.get(canon) || null;
    let hitStereoBlind = false;
    if (!hit && ikey && (hit = byFlat.get(ikey.split("-")[0]) || null)) hitStereoBlind = true;

    // Structure depiction
    try { $("structure-svg").innerHTML = mol.get_svg(230, 190); } catch (e) { $("structure-svg").innerHTML = ""; }
    $("mol-name").textContent = displayName ? displayName : canon;

    const flags = structuralFlags(mol, canon);
    const rule = ruleOfThree(desc);

    // Everything the odor panel weighs. Physical properties and the published
    // atlases arrive asynchronously; OpenPOM later still (it may run live).
    current = { seq: ++aSeq, ikey, canon, desc, hit, hitStereoBlind, flags, rule,
                phys: undefined, atlas: undefined, manual: null };
    $("man-vp").value = ""; $("man-logp").value = "";
    $("verdict").className = "verdict";
    $("verdict").innerHTML = `<p class="hint" style="margin:0">Weighing the evidence…</p>`;
    $("evidence").innerHTML = "";
    $("transport-box").innerHTML = "";
    renderProps(current);
    $("result").classList.remove("hidden");

    const ctx = current;
    Promise.all([ensurePhyschem(), ensureQuality()]).then(() => {
      if (ctx !== current) return;
      ctx.phys = lookupPhys(ikey);
      const q = (ikey && byKeyQ.get(ikey)) || byCanQ.get(canon) || null;
      ctx.atlas = atlasEvidence(q);
      renderProps(ctx);
      renderOdor(ctx);
    });

    renderToxicity(ctx);
    renderIntensity(ikey, canon);
    // The mol is deleted as soon as analyze() returns, so capture the graph now —
    // the OpenPOM worker may need it after an async weight download.
    let molJson = null;
    try { molJson = mol.get_json(); } catch (e) {}
    renderOpenPOM(ikey, canon, molJson);
    renderQuality(ikey, canon);
  }

  function renderProps(ctx) {
    const { desc, canon } = ctx;
    const t = transportInputs(ctx);
    const props = [
      ["Molecular weight", fmt(desc.mw, 2) + " Da"],
      ["Heteroatoms", desc.nhet == null ? "—" : String(desc.nhet)],
      ["Heavy atoms", desc.heavy == null ? "—" : String(desc.heavy)],
      ["Vapor pressure (25 °C)", t.vp != null
        ? `${fmtSci(t.vp)} mmHg<small class="src">${escapeHtml(t.vpSource)}</small>`
        : (ctx.phys === undefined ? "…" : "—")],
      ["logP", t.logp != null
        ? `${fmt(t.logp, 2)}<small class="src">${escapeHtml(t.logpSource)}</small>` : "—"],
    ];
    if (t.logpSource !== LOGP_CRIPPEN && desc.logp != null)
      props.push(["logP (computed)", `${fmt(desc.logp, 2)}<small class="src">${LOGP_CRIPPEN}</small>`]);
    props.push(["Canonical SMILES", `<code>${escapeHtml(canon)}</code>`]);
    $("props").innerHTML = props.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  }

  // ---- Shared physical properties (physchem.json: CompTox OPERA + EPI Suite VP, logP) ----
  function ensurePhyschem() {
    if (!physPromise) {
      physPromise = fetch(asset("data/physchem.json")).then((r) => r.json()).then((d) => {
        PHYS = d;
        // CompTox often lists only specific stereoisomers (e.g. (R)-2-nonanol) while
        // the site has the unspecified form. VP and logP are stereo-insensitive
        // (siblings differ by <=0.7 log units), so fall back to the skeleton,
        // preferring measured values.
        const rank = (r) => (r.v == null ? 0 : physPredicted(r.vs) ? 1 : 2);
        for (const rec of d.mols) {
          byKeyPhys.set(rec.i, rec);
          const f = rec.i.split("-")[0], cur = byFlatPhys.get(f);
          if (!cur || rank(rec) > rank(cur)) byFlatPhys.set(f, rec);
        }
      }).catch(() => {});
    }
    return physPromise;
  }

  function lookupPhys(ikey) {
    if (!ikey) return null;
    const exact = byKeyPhys.get(ikey);
    // st: the build already borrowed this VP from a stereoisomer (measured beats own-predicted)
    if (exact) return exact.st ? { ...exact, stereo: true } : exact;
    const sib = byFlatPhys.get(ikey.split("-")[0]);
    return sib ? { ...sib, stereo: true } : null;
  }

  // Whether a physchem.json source code is a model prediction (OPERA, MPBPVP, KOWWIN).
  const physPredicted = (code) => ((PHYS && PHYS.meta.pred) || ["p"]).includes(code);

  // Human-readable source of a physchem.json value, e.g. "CompTox experimental".
  function physSource(phys, key) {
    const s = ((PHYS && PHYS.meta.src) || {})[phys[key]] || "";
    return phys.stereo ? `${s}, other stereoisomer` : s;
  }

  const LOGP_CRIPPEN = "computed from structure (Crippen)";
  // logP units below the low-volatility boundary beyond which physics overrides reports
  // (larger when the vapor pressure is itself a prediction). See odorVerdict.
  const INVOLATILE = 3, INVOLATILE_PRED = 5;

  // Best available vapor pressure and logP, each with where it came from. The
  // transport boundaries were fit on the paper's dataset values, so those win.
  function transportInputs(ctx) {
    const { hit, desc, phys, manual } = ctx;
    const o = { vp: null, vpSource: "", vpPredicted: false, logp: null, logpSource: "", logpFit: false };
    if (manual && manual.vp != null) { o.vp = manual.vp; o.vpSource = "your value"; }
    else if (hit && hit.vp != null) { o.vp = hit.vp; o.vpSource = "Mayhew et al. dataset"; }
    else if (phys && phys.v != null) {
      o.vp = phys.v; o.vpSource = physSource(phys, "vs"); o.vpPredicted = physPredicted(phys.vs);
    }
    if (manual && manual.logp != null) { o.logp = manual.logp; o.logpSource = "your value"; }
    else if (hit && hit.logp != null) {
      o.logp = hit.logp; o.logpSource = "Mayhew et al. dataset (Moriguchi)"; o.logpFit = true;
    } else if (phys && phys.l != null) { o.logp = phys.l; o.logpSource = physSource(phys, "ls"); }
    else if (desc.logp != null) { o.logp = desc.logp; o.logpSource = LOGP_CRIPPEN; }
    return o;
  }

  // ---- Reported odor / odorless (published atlases) ----
  // A source counts as reporting "odorless" only if that is all it says; one that
  // lists "odorless" alongside descriptors still describes an odor. AromaDB's
  // "no aroma" is ignored outright: it is attached to indole, acetophenone, octanoic
  // acid and ~50 other plainly odorous molecules, so it reads as a no-data placeholder.
  const ODORLESS_TERMS = /\b(odou?rless|no (apparent |distinct |perceptible )?(odou?r|smell)|none|bland)\b/gi;
  const isPlaceholder = (src, text) => src === "AromaDB" && String(text).trim().toLowerCase() === "no aroma";
  const DESC_STOP = new Set(["like", "and", "with", "slight", "slightly", "very", "mild", "faint", "odor", "note", "notes"]);

  function atlasEvidence(rec) {
    if (!rec) return null;
    const odor = [], none = [], seen = new Map();
    let ignored = null;
    for (const [src, text] of Object.entries(rec.sources || {})) {
      if (isPlaceholder(src, text)) { ignored = src; continue; }
      const rest = String(text).replace(ODORLESS_TERMS, " ");
      const words = rest.toLowerCase().split(/[^a-z-]+/).filter((w) => w.length > 2 && !DESC_STOP.has(w));
      if (!words.length) { none.push(src); continue; }
      odor.push(src);
      for (const w of new Set(words)) seen.set(w, (seen.get(w) || 0) + 1);
    }
    // Descriptors named by more than one source, most-cited first.
    const common = [...seen].filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]).slice(0, 4);
    return { odor, none, common, ignored, name: rec.name };
  }

  // ---- The integrated odor panel ----
  // Every line of evidence the site has, strongest first. The headline takes the
  // strongest one available; lower lines that disagree are called out, not hidden.
  function odorEvidence(ctx) {
    const { hit, atlas, rule, flags, desc } = ctx;
    const rows = [];

    if (atlas && (atlas.odor.length || atlas.none.length)) {
      const n = atlas.odor.length + atlas.none.length;
      const parts = [];
      if (atlas.odor.length) {
        parts.push(`${atlas.odor.length} of ${n} describe an odor (${atlas.odor.join(", ")})` +
          (atlas.common.length
            ? ` — most often <em>${atlas.common.map(([w]) => escapeHtml(w)).join(", ")}</em>`
            : ""));
      }
      if (atlas.none.length) parts.push(`${atlas.none.length} report${atlas.none.length === 1 ? "s" : ""} it odorless (${atlas.none.join(", ")})`);
      const lab = hit && hit.src === "Lab-Tested";
      rows.push({ tier: "reported", used: !lab,
        vote: atlas.odor.length && atlas.none.length ? "mixed" : atlas.odor.length ? "odor" : "odorless",
        name: "Published odor atlases", detail: parts.join("; ") + "." +
          (atlas.ignored ? `<span class="fine">AromaDB's “no aroma” entry is ignored; it is ` +
            `attached to many clearly odorous molecules.</span>` : "") +
          (lab ? `<span class="fine">Outranked by the lab test below.</span>` : ""),
        nOdor: atlas.odor.length, nNone: atlas.none.length });
    }
    if (hit && hit.odor) {
      const isOdor = hit.odor === "Odor", lab = hit.src === "Lab-Tested";
      rows.push({ tier: "reported", used: true, vote: isOdor ? "odor" : "odorless",
        name: lab ? "Lab-tested (Mayhew et al.)" : "Mayhew et al. curated dataset", lab,
        detail: `Labelled <strong>${escapeHtml(hit.odor.toLowerCase())}</strong>` +
          (hit.src ? ` (${escapeHtml(hit.src)})` : "") + "." +
          (ctx.hitStereoBlind ? `<span class="fine">Matched ignoring stereochemistry.</span>` : ""),
        nOdor: isOdor ? 1 : 0, nNone: isOdor ? 0 : 1 });
    }
    if (hit && hit.p != null) {
      rows.push({ tier: "model", used: true, vote: hit.p >= 0.5 ? "odor" : "odorless",
        name: "Transport-ML model",
        detail: `p(odorous) = ${hit.p.toFixed(3)} — the paper's gradient-boosted model, which uses ` +
          `measured transport features.`, prob: hit.p });
    }

    const t = transportInputs(ctx);
    const tv = transportVerdict(t.vp, t.logp);
    ctx.transport = { t, tv };
    if (tv.verdict) {
      const scaleNote = t.logpFit ? "" :
        ` The boundaries were fit on Moriguchi logP; this uses ${t.logpSource.includes("Crippen") ? "Crippen" : t.logpSource.includes("KOWWIN") ? "KOWWIN" : t.logpSource.includes("OPERA") ? "OPERA" : t.logpSource.includes("experimental") ? "measured octanol–water" : "your"} logP.`;
      rows.push({ tier: "model", used: true, vote: tv.verdict,
        name: "Transport boundaries (vapor pressure &amp; logP)",
        detail: `VP ${fmtSci(t.vp)} mmHg <small class="src">${escapeHtml(t.vpSource)}</small>, ` +
          `logP ${fmt(t.logp, 2)} <small class="src">${escapeHtml(t.logpSource)}</small> — ` +
          `${tv.verdict === "odor" ? "inside" : "outside"} the odorous window ` +
          `(${fmt(tv.low, 1)} &lt; logP &lt; ${fmt(tv.high, 1)} at this volatility)` +
          (tv.lowMargin < 0 ? `, ${fmt(-tv.lowMargin, 1)} logP units below the low-volatility boundary.` : ".") +
          `<span class="fine">${t.vpPredicted ? " The vapor pressure is a model prediction, which can be 10–50× off." : ""}${scaleNote}</span>` });
    } else {
      rows.push({ tier: "model", used: false, vote: null,
        name: "Transport boundaries (vapor pressure &amp; logP)",
        detail: "No vapor pressure is available for this molecule. Enter one under the plot below to run it." });
    }

    const na = flags.salt || flags.inorganic;
    const mw = desc.mw == null ? "?" : fmt(desc.mw, 1), het = desc.nhet == null ? "?" : desc.nhet;
    rows.push({ tier: "rule", used: !na,
      vote: na ? null : rule.odorous ? "odor" : rule.tooHeavyOrPolar ? "odorless" : null,
      name: "Rule of three",
      detail: na
        ? `Not applicable to ${flags.salt ? "salts / multi-component species" : "inorganic or metal-containing compounds"}.`
        : `MW ${mw} Da (need 30–300: ${yn(rule.mwOK)}), heteroatoms ${het} (need &lt;4: ${yn(rule.hetOK)}).`,
      why: "it screens on size and heteroatom count alone, which is useful when nothing else is known " +
        "but coarse: many heavier or more polar molecules still smell" });

    return rows;
  }

  function odorVerdict(ctx, rows) {
    const reported = rows.filter((r) => r.tier === "reported" && r.used);
    const s = (n) => (n === 1 ? "" : "s");
    // A direct lab test outranks everything, including the involatility override
    // (no lab-tested molecule below that line is labelled odorous, so they never clash).
    const lab = reported.find((r) => r.lab);
    if (lab) {
      const atlas = rows.find((r) => r.name === "Published odor atlases");
      const against = atlas ? (lab.vote === "odor" ? atlas.nNone : atlas.nOdor) : 0;
      return { dir: lab.vote, cls: lab.vote === "odor" ? "v-odor" : "v-odorless",
        icon: lab.vote === "odor" ? "👃" : "🚫", label: lab.vote === "odor" ? "Odorous" : "Odorless",
        pill: "Lab-tested",
        sub: `Tested directly in the Mayhew et al. study.` +
          (against ? ` ${against} published atlas source${s(against)} say${against === 1 ? "s" : ""} otherwise; ` +
            `the lab test takes precedence.` : "") };
    }
    const nOdor = reported.reduce((a, r) => a + r.nOdor, 0);
    const nNone = reported.reduce((a, r) => a + r.nNone, 0);
    // Physics outranks reports for molecules far too involatile to reach the nose. In the
    // Mayhew dataset, molecules >3 logP units below the low-volatility boundary are labelled
    // odorless 99 : 34 (the odorous ones mostly reactive inorganics smelled via breakdown
    // products), and every atlas-"odorous" one the dataset also labels is odorless (16/16) --
    // sugars, amino acids, acids, surfactants described by taste ("sweet") or by impurities.
    // A predicted VP can be 10-50x off (~1.7 log units ≈ 3 logP units here), so it needs more.
    const { t, tv } = ctx.transport;
    if (tv.verdict === "odorless" && tv.lowMargin < -(t.vpPredicted ? INVOLATILE_PRED : INVOLATILE)) {
      return { dir: "odorless", cls: "v-odorless", icon: "🚫", label: "Probably odorless", pill: "Too involatile",
        sub: `Its vapor pressure (${fmtSci(t.vp)} mmHg) is far too low for it to reach the nose: it sits ` +
          `${fmt(-tv.lowMargin, 1)} logP units below the transport model's low-volatility boundary.` +
          (nOdor ? ` The ${nOdor === 1 ? "report of an odor more likely reflects" : `${nOdor} reports of an odor more likely reflect`} a taste ` +
            `word such as “sweet”, an impurity, or a breakdown product.` : "") };
    }
    if (nOdor + nNone > 0) {
      if (!nNone) return { dir: "odor", cls: "v-odor", icon: "👃", label: "Odorous", pill: "Reported",
        sub: `Described as having an odor by ${nOdor} published source${s(nOdor)}.` };
      if (!nOdor) return { dir: "odorless", cls: "v-odorless", icon: "🚫", label: "Odorless", pill: "Reported",
        sub: `Reported odorless by ${nNone} published source${s(nNone)}.` };
      return { dir: nOdor > nNone ? "odor" : nNone > nOdor ? "odorless" : null, cls: "v-maybe", icon: "❓",
        label: nOdor > nNone ? "Probably odorous" : nNone > nOdor ? "Probably odorless" : "Reports conflict",
        pill: "Reports disagree",
        sub: `${nOdor} source${s(nOdor)} describe${nOdor === 1 ? "s" : ""} an odor and ${nNone} ` +
          `report${nNone === 1 ? "s" : ""} it odorless — it may be a weak odorant, or an impurity may be what was smelled.` };
    }
    const ml = rows.find((r) => r.prob != null);
    if (ml) {
      const p = ml.prob;
      return { dir: ml.vote, cls: p >= 0.5 ? "v-odor" : "v-odorless", icon: p >= 0.5 ? "👃" : "🚫",
        label: p >= 0.5 ? "Probably odorous" : "Probably odorless", pill: "Transport-ML model",
        sub: `${Math.round(Math.max(p, 1 - p) * 100)}% confidence.` };
    }
    if (ctx.flags.salt || ctx.flags.inorganic) {
      return { dir: null, cls: "v-maybe", icon: "❓", label: "Uncertain", pill: "Outside the models",
        sub: ctx.flags.salt
          ? "Salt / multi-component species — the transport models don't apply cleanly, and no reports were found."
          : "Inorganic / metal-containing — outside the models' chemical space, and no reports were found." };
    }
    const tb = rows.find((r) => r.name.startsWith("Transport boundaries") && r.used);
    if (tb) {
      return { dir: tb.vote, cls: tb.vote === "odor" ? "v-odor" : "v-odorless", icon: tb.vote === "odor" ? "👃" : "🚫",
        label: tb.vote === "odor" ? "Probably odorous" : "Probably odorless", pill: "Transport model",
        sub: `No published reports found; predicted from its vapor pressure and logP` +
          (t.vpPredicted ? " (the vapor pressure itself is predicted)." : ".") };
    }
    const r3 = ctx.rule;
    return { dir: r3.odorous ? "odor" : r3.tooHeavyOrPolar ? "odorless" : null,
      cls: r3.odorous ? "v-odor" : r3.tooHeavyOrPolar ? "v-odorless" : "v-maybe",
      icon: r3.odorous ? "👃" : r3.tooHeavyOrPolar ? "🚫" : "❓",
      label: r3.odorous ? "Possibly odorous" : r3.tooHeavyOrPolar ? "Possibly odorless" : "Uncertain",
      pill: "Rule of three only",
      sub: "No reports or vapor pressure found, so this rests on a structure-only rule of thumb." };
  }

  const VOTE_TAG = {
    odor: `<span class="tag odor">odor</span>`,
    odorless: `<span class="tag odorless">odorless</span>`,
    mixed: `<span class="tag maybe">mixed</span>`,
  };
  const TIER_LABEL = { reported: "What people report", model: "What the models predict", rule: "Rule of thumb" };

  function renderOdor(ctx) {
    if (ctx !== current || ctx.atlas === undefined) return;
    const rows = odorEvidence(ctx);
    const v = odorVerdict(ctx, rows);
    const disagree = v.dir ? rows.filter((r) => r.used && (r.vote === "odor" || r.vote === "odorless") && r.vote !== v.dir) : [];

    const el = $("verdict");
    el.className = "verdict " + v.cls;
    el.innerHTML =
      `<span class="pill">${escapeHtml(v.pill)}</span>` +
      `<p class="big">${v.icon} ${escapeHtml(v.label)}</p>` +
      `<p class="sub">${escapeHtml(v.sub)}</p>` +
      (disagree.length
        ? `<p class="disagree"><strong>Disagrees:</strong> ` +
          disagree.map((r) => `${r.name}${r.why ? ` — ${r.why}` : ""}`).join("; ") + `.</p>`
        : "");

    let html = "", tier = null;
    for (const r of rows) {
      if (r.tier !== tier) { tier = r.tier; html += `<div class="ev-tier">${TIER_LABEL[tier]}</div>`; }
      html += `<div class="ev${r.used ? "" : " ev-off"}">` +
        `<div class="ev-vote">${VOTE_TAG[r.vote] || `<span class="tag na">—</span>`}</div>` +
        `<div><div class="name">${r.name}</div><div class="verdict-line">${r.detail}</div>` +
        (r.prob != null ? `<div class="prob-bar"><div style="width:${Math.round(r.prob * 100)}%"></div></div>` : "") +
        `</div></div>`;
    }
    html += `<p class="hint" style="margin:14px 0 0">The headline uses the strongest evidence available: ` +
      `a lab test › what people report › the paper's transport-ML model › the transport boundaries › the rule of three. ` +
      `The exception: a molecule far below the low-volatility boundary (${INVOLATILE} logP units, or ` +
      `${INVOLATILE_PRED} if its vapor pressure is predicted) is called odorless whatever the atlases say.</p>`;
    $("evidence").innerHTML = html;
    renderTransport(ctx);
  }

  // ---- Odor quality: predicted (OpenPOM, lookup of precomputed predictions) ----
  // Descriptor probabilities from the Principal Odor Map MPNN, precomputed for the
  // pyrfume molecule set. openpom.json is large, so it is fetched on first use.
  function ensureOpenPOM() {
    if (!pomPromise) {
      pomPromise = fetch(asset("data/openpom.json")).then((r) => r.json()).then((d) => {
        POM = d;
        for (const rec of d.mols) {
          byKeyPOM.set(rec.i, rec.p);
          byCanPOM.set(rec.c, rec.p);
        }
        // Alternate canonical forms of molecules whose InChIKeys collide.
        for (const can in d.alt) byCanPOM.set(can, d.alt[can]);
      }).catch(() => {});
    }
    return pomPromise;
  }

  // Where a score falls in the distribution of that descriptor across the whole
  // library, as a "top N%". Scores are uncalibrated and base rates differ by orders
  // of magnitude between descriptors, so the rank matters as much as the value.
  // POM.q[label] holds the 0th..100th percentile of the label, as 0-1000 ints.
  function pomTopPercent(labelIdx, score) {
    const arr = POM && POM.q && POM.q[labelIdx];
    if (!arr) return null;
    let lo = 0, hi = arr.length - 1;
    if (score <= arr[0]) return 100;
    if (score >= arr[hi]) return 0;
    while (lo < hi - 1) {                    // arr is ascending
      const mid = (lo + hi) >> 1;
      if (arr[mid] <= score) lo = mid; else hi = mid;
    }
    const span = arr[hi] - arr[lo];
    return 100 - (lo + (span ? (score - arr[lo]) / span : 0));
  }

  function fmtTopPercent(p) {
    if (p == null) return "";
    if (p < 0.1) return "top 0.1%";
    if (p < 10) return `top ${p.toFixed(1)}%`;
    return `top ${Math.round(p)}%`;
  }

  const POM_SHOW = 8;        // descriptors listed per molecule
  const POM_WEAK = 0.2;      // below this, flag the whole profile as low-confidence
  const POM_KEEP = 15;       // descriptors kept from a live prediction (matches the table)

  // ---- Live inference (pom-worker.js) ----
  // For molecules outside the precomputed table, run the model itself. The weights
  // are ~8 MB, so the worker is only created — and the download only started — when
  // a molecule actually misses the lookup.
  let pomWorker = null, pomJobSeq = 0, pomModelReady = false;
  const pomJobs = new Map();

  function pomPredict(molJson) {
    return new Promise((resolve, reject) => {
      if (!pomWorker) {
        try {
          pomWorker = new Worker(asset("pom-worker.js"));
        } catch (e) { reject(e); return; }
        pomWorker.onmessage = (ev) => {
          const job = pomJobs.get(ev.data.seq);
          if (!job) return;
          pomJobs.delete(ev.data.seq);
          if (ev.data.error) return job.reject(new Error(ev.data.error));
          pomModelReady = true;
          job.resolve(ev.data.probs);
        };
        pomWorker.onerror = (e) => {
          for (const [, job] of pomJobs) job.reject(new Error(e.message || "worker failed"));
          pomJobs.clear();
          pomWorker = null;
        };
      }
      const seq = ++pomJobSeq;
      pomJobs.set(seq, { resolve, reject });
      pomWorker.postMessage({ seq, json: molJson, base: "data/", v: VERSION });
    });
  }

  // Model output (138 probabilities) -> the same [labelIdx, score0-1000] shape the
  // lookup table uses, so both paths render through one code path.
  function pomProbsToEntries(probs) {
    const idx = Array.from({ length: probs.length }, (_, i) => i);
    idx.sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, POM_KEEP).map((j) => [j, Math.round(probs[j] * 1000)]);
  }

  function renderOpenPOM(ikey, canon, molJson) {
    const box = $("openpom-box");
    if (!box) return;
    const seq = ++pSeq;
    box.innerHTML = `<p class="hint" style="margin:0">Loading predicted descriptors…</p>`;
    ensureOpenPOM().then(() => {
      if (seq !== pSeq) return; // a newer molecule was analyzed; skip stale write
      const p = (ikey && byKeyPOM.get(ikey)) || byCanPOM.get(canon) || null;
      if (p && POM) return renderPomEntries(box, seq, p, "lookup");

      if (!POM) {
        box.innerHTML = `<p class="hint" style="margin:0">Odor-quality predictions are ` +
          `unavailable right now.</p>`;
        return;
      }
      if (!molJson) {
        box.innerHTML = `<p class="hint" style="margin:0">Could not read this molecule's ` +
          `structure, so no prediction was run.</p>`;
        return;
      }
      // Not in the precomputed table — run the model in the browser.
      box.innerHTML = `<p class="hint" style="margin:0">` + (pomModelReady
        ? "Running the OpenPOM model…"
        : "This molecule isn't in the precomputed set, so the model is running here in " +
          "your browser. Downloading it (~8 MB, first time only)…") + `</p>`;
      pomPredict(molJson).then((probs) => {
        if (seq !== pSeq) return;
        renderPomEntries(box, seq, pomProbsToEntries(probs), "predicted");
      }).catch((err) => {
        if (seq !== pSeq) return;
        box.innerHTML = `<p class="hint" style="margin:0">Couldn't run the OpenPOM model ` +
          `for this molecule (${escapeHtml(err.message || "error")}).</p>`;
      });
    });
  }

  function renderPomEntries(box, seq, p, source) {
      if (seq !== pSeq) return;
      // "odorless" is a meta-label, not an odor character, and OpenPOM is unreliable at
      // odor/odorless — so it is left out of the list and not used by the odor panel.
      const odorlessIdx = POM.labels.indexOf("odorless");
      const chars = p.filter((e) => e[0] !== odorlessIdx).slice(0, POM_SHOW);

      const rows = chars.map(([j, v]) => {
        const s = v / 1000;
        return `<div class="pom-row">` +
          `<span class="pom-name">${escapeHtml(POM.labels[j])}</span>` +
          `<span class="pom-bar"><i style="width:${(s * 100).toFixed(1)}%"></i></span>` +
          `<span class="pom-val">${s.toFixed(2)}</span>` +
          `<span class="pom-pct">${fmtTopPercent(pomTopPercent(j, v))}</span>` +
        `</div>`;
      }).join("");

      const weak = !chars.length || chars[0][1] / 1000 < POM_WEAK;
      const lead = weak
        ? `<p class="hint" style="margin:0 0 10px">No descriptor scores highly for this ` +
          `molecule — the model gives it no confident odor character.</p>`
        : `<p class="verdict-line" style="margin:0 0 10px">` +
          "Predicted odor character:" +
          `</p>`;
      const list = `<div class="pom-list">${rows}</div>`;
      const n = POM.meta.n.toLocaleString();
      const provenance = source === "predicted"
        ? `<p class="hint" style="margin:12px 0 0"><span class="tag dataset">computed here</span> ` +
          `This molecule isn't in the precomputed set, so the model ran in your browser. ` +
          `Scores are uncalibrated, so each is also ranked against the ${n} molecules ` +
          `OpenPOM was run over.</p>`
        : `<p class="hint" style="margin:12px 0 0">Model scores are uncalibrated, so each is ` +
          `also given as its rank among the ${n} molecules OpenPOM was run over.</p>`;
      box.innerHTML = lead + list + provenance;
  }

  // ---- Odor quality (lookup only, lazy-loaded) ----
  // Published-atlas descriptors aggregated via Pyrfume. quality.json is large,
  // so it is fetched on first use rather than at boot.
  function ensureQuality() {
    if (!qualPromise) {
      qualPromise = fetch(asset("data/quality.json")).then((r) => r.json()).then((d) => {
        for (const rec of d) {
          if (rec.ikey) byKeyQ.set(rec.ikey, rec);
          if (rec.can) byCanQ.set(rec.can, rec);
          if (rec.name && !byNameQ.has(rec.name.toLowerCase())) byNameQ.set(rec.name.toLowerCase(), rec);
        }
      }).catch(() => {});
    }
    return qualPromise;
  }

  // Preferred display order; any other sources are appended.
  const QUALITY_ORDER = ["Leffingwell", "Goodscents", "Arctander", "IFRA", "Dravnieks", "Sigma", "AromaDB"];

  function renderQuality(ikey, canon) {
    const box = $("quality-box");
    if (!box) return;
    const seq = ++qSeq;
    box.innerHTML = `<p class="hint" style="margin:0">Loading odor descriptors…</p>`;
    ensureQuality().then(() => {
      if (seq !== qSeq) return; // a newer molecule was analyzed; skip stale write
      const rec = (ikey && byKeyQ.get(ikey)) || byCanQ.get(canon) || null;
      if (!rec) {
        box.innerHTML =
          `<p class="hint" style="margin:0">This molecule isn't in the published odor-descriptor ` +
          `datasets, so no quality profile is shown.</p>`;
        return;
      }
      const srcs = rec.sources || {};
      const keys = QUALITY_ORDER.filter((k) => srcs[k])
        .concat(Object.keys(srcs).filter((k) => !QUALITY_ORDER.includes(k)));
      const rows = keys.map((k) =>
        `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(srcs[k])}</td></tr>`).join("");
      box.innerHTML =
        `<p class="verdict-line" style="margin:0 0 10px">Odor-character descriptors reported across ` +
        `<strong>${rec.n}</strong> published database${rec.n === 1 ? "" : "s"}:</p>` +
        `<table class="qtable">${rows}</table>`;
    });
  }

  // ---- Intensity (MixInt network, precomputed curves) ----
  // Each record holds the network's output on a fixed log10-concentration grid (x10).
  // The network is piecewise-linear in log C, so linear interpolation is near-exact.
  function ensureIntensity() {
    if (!intPromise) {
      intPromise = fetch(asset("data/intensity.json")).then((r) => r.json()).then((d) => {
        INT = d;
        for (const rec of d.mols) {
          byKeyInt.set(rec.i, rec);
          byCanInt.set(rec.c, rec);
        }
      }).catch(() => {});
    }
    return intPromise;
  }

  const PPM = 1e6;                 // v/v -> ppm
  // The panel rated on the generalized Labeled Magnitude Scale (gLMS, 0-100). Its verbal
  // anchors sit at quasi-logarithmic positions (Green et al. 1996; Bartoshuk et al. 2004).
  const GLMS = [
    [1.4, "barely detectable"], [6, "weak"], [17, "moderate"],
    [35, "strong"], [53, "very strong"], [100, "strongest imaginable"],
  ];
  const INT_REF = 17;              // "moderate" on the gLMS: reference point for the readout

  // Nearest gLMS label for a rating: the anchor itself when close, else the two it sits between.
  function glmsWord(y) {
    if (y < GLMS[0][0] * 0.7) return "below barely detectable";
    for (let i = 0; i < GLMS.length - 1; i++) {
      const [lo, a] = GLMS[i], [hi, b] = GLMS[i + 1];
      if (y > hi) continue;
      const f = (y - lo) / (hi - lo);
      return f < 0.2 ? a : f > 0.8 ? b : `between ${a} and ${b}`;
    }
    return GLMS[GLMS.length - 1][1];
  }

  function intensityAt(rec, logc) {
    const g = INT.meta.grid;
    const t = (logc - g.lo) / g.step;
    if (t <= 0) return rec.y[0] / 10;
    if (t >= g.n - 1) return rec.y[g.n - 1] / 10;
    const k = Math.floor(t), f = t - k;
    return (rec.y[k] * (1 - f) + rec.y[k + 1] * f) / 10;
  }

  // First log10 C at which the curve reaches `level`, or null.
  function concForIntensity(rec, level) {
    const g = INT.meta.grid, L = level * 10;
    if (rec.y[0] >= L) return g.lo;
    for (let k = 1; k < g.n; k++) {
      if (rec.y[k] >= L) {
        const f = (L - rec.y[k - 1]) / (rec.y[k] - rec.y[k - 1]);
        return g.lo + (k - 1 + f) * g.step;
      }
    }
    return null;
  }

  function fmtPpm(logc) { return fmtSci(Math.pow(10, logc) * PPM) + " ppm"; }

  function renderIntensity(ikey, canon) {
    const box = $("intensity-box");
    if (!box) return;
    const seq = ++iSeq;
    box.innerHTML = `<p class="hint" style="margin:0">Loading intensity predictions…</p>`;
    ensureIntensity().then(() => {
      if (seq !== iSeq) return; // a newer molecule was analyzed; skip stale write
      if (!INT) {
        box.innerHTML = `<p class="hint" style="margin:0">Intensity predictions are unavailable right now.</p>`;
        return;
      }
      const rec = (ikey && byKeyInt.get(ikey)) || byCanInt.get(canon) || null;
      if (!rec) {
        box.innerHTML =
          `<p class="hint" style="margin:0">This molecule isn't among the ${INT.meta.n.toLocaleString()} ` +
          `with the Dragon descriptors and vapor pressure the intensity network needs, so no curve is ` +
          `shown.</p>`;
        return;
      }
      const tr = INT.meta.train;
      const sat = Math.log10(rec.vp / 760);          // saturated vapor, v/v at 25 °C
      const g = INT.meta.grid, gHi = g.lo + (g.n - 1) * g.step;
      const satIn = sat >= g.lo && sat <= gHi;
      const iSat = satIn ? intensityAt(rec, sat) : null;
      const cRef = concForIntensity(rec, INT_REF);
      const vpSrc = (INT.meta.vs[rec.vs] || "") + (rec.st ? ", other stereoisomer" : "");

      const lines = [];
      if (iSat != null) {
        lines.push(`Saturated headspace (neat, 25&nbsp;°C, ${fmtPpm(sat)}): predicted intensity ` +
          `<strong>${iSat.toFixed(0)}</strong> (${glmsWord(iSat)}).`);
      } else if (sat < g.lo) {
        lines.push(`Its saturated vapor concentration (${fmtPpm(sat)}) is below the plotted range — ` +
          `it barely evaporates at room temperature.`);
      }
      if (cRef != null && cRef <= sat) {
        lines.push(`Reaches <em>moderate</em> (${INT_REF} on the gLMS) at ≈ <strong>${fmtPpm(cRef)}</strong>.`);
      } else {
        lines.push(`Doesn't reach <em>moderate</em> (${INT_REF} on the gLMS) below saturation.`);
      }
      const outside = (cRef != null && (cRef < tr.lo || cRef > tr.hi));
      box.innerHTML =
        `<div class="model" style="border:none;padding-top:0">` +
        `<div class="name">Predicted concentration–intensity curve ` +
        (rec.t ? `<span class="tag dataset">training odorant</span>` : "") + `</div>` +
        `<div class="verdict-line">${lines.join("<br>")}</div>` +
        intensityPlotSVG(rec, sat) +
        `<div class="int-probe"><label>Concentration (ppm, v/v in air) ` +
        `<input id="int-ppm" type="number" step="any" min="0" placeholder="e.g. 1" /></label>` +
        `<span id="int-out" class="verdict-line"></span></div>` +
        `<div class="verdict-line" style="color:var(--muted)">Vapor pressure ${fmtSci(rec.vp)} mmHg ` +
        `— ${escapeHtml(vpSrc)}. Solid line: the concentration range the panel rated ` +
        `(${fmtPpm(tr.lo)} – ${fmtPpm(tr.hi)}); dashed: extrapolation` +
        (outside ? `, which includes this molecule's moderate-intensity point` : "") +
        `. Beyond saturation (grey) the concentration can't be reached at 25&nbsp;°C.</div>` +
        `<div class="verdict-line" style="color:var(--muted)">Model predictions from a ` +
        `<strong>preprint</strong> (not yet peer reviewed), trained on ${tr.n} odorants — ` +
        `treat values for molecules unlike those as rough estimates.</div>` +
        `</div>`;

      const inp = $("int-ppm"), out = $("int-out");
      inp.addEventListener("input", () => {
        const ppm = parseFloat(inp.value);
        if (!(ppm > 0)) { out.innerHTML = ""; return; }
        const lc = Math.log10(ppm / PPM);
        const yi = intensityAt(rec, lc);
        let msg = `→ predicted intensity <strong>${yi.toFixed(0)}</strong> (${glmsWord(yi)})`;
        if (lc > sat) msg += ` <span style="color:var(--maybe)">(above saturation — not reachable at 25&nbsp;°C)</span>`;
        else if (lc < tr.lo || lc > tr.hi) msg += ` <span style="color:var(--muted)">(extrapolated)</span>`;
        out.innerHTML = msg;
      });
    });
  }

  function intensityPlotSVG(rec, sat) {
    const g = INT.meta.grid, tr = INT.meta.train;
    const xmin = g.lo, xmax = g.lo + (g.n - 1) * g.step;
    const W = 580, H = 320, mL = 42, mR = 118, mT = 14, mB = 42;
    const pw = W - mL - mR, ph = H - mT - mB;
    const X = (x) => mL + (x - xmin) / (xmax - xmin) * pw;
    const Y = (y) => mT + (100 - Math.min(Math.max(y, 0), 100)) / 100 * ph;
    const pts = (lo, hi) => {
      const p = [];
      for (let k = 0; k < g.n; k++) {
        const x = g.lo + k * g.step;
        if (x >= lo - 1e-9 && x <= hi + 1e-9) p.push(`${X(x).toFixed(1)},${Y(rec.y[k] / 10).toFixed(1)}`);
      }
      // close the segment exactly at its end points
      if (lo > xmin) p.unshift(`${X(lo).toFixed(1)},${Y(intensityAt(rec, lo)).toFixed(1)}`);
      if (hi < xmax) p.push(`${X(hi).toFixed(1)},${Y(intensityAt(rec, hi)).toFixed(1)}`);
      return p.join(" ");
    };
    let grid = "", axis = "";
    for (let t = Math.ceil(xmin); t <= xmax; t += 2) {
      const px = X(t);
      grid += `<line x1="${px}" y1="${mT}" x2="${px}" y2="${mT + ph}" stroke="#22303e"/>`;
      // label as ppm: 10^(t+6)
      const e = t + 6;
      const lab = e === 0 ? "1" : e === 1 ? "10" : `10<tspan dy="-4" font-size="8">${e}</tspan>`;
      axis += `<text x="${px}" y="${mT + ph + 15}" fill="#9fb0c0" font-size="10" text-anchor="middle">${lab}</text>`;
    }
    for (let t = 0; t <= 100; t += 25) {
      axis += `<text x="${mL - 6}" y="${Y(t) + 3}" fill="#9fb0c0" font-size="10" text-anchor="end">${t}</text>`;
    }
    // gLMS verbal anchors: a gridline at each, labelled on the right.
    for (const [v, name] of GLMS) {
      const py = Y(v);
      grid += `<line x1="${mL}" y1="${py}" x2="${mL + pw}" y2="${py}" stroke="#2b3a49"/>`;
      axis += `<line x1="${mL + pw}" y1="${py}" x2="${mL + pw + 5}" y2="${py}" stroke="#7f93a8"/>` +
        `<text x="${mL + pw + 8}" y="${py + 3.5}" fill="#9fb0c0" font-size="10">${name}</text>`;
    }
    const satX = Math.min(Math.max(sat, xmin), xmax);
    const satShade = sat < xmax
      ? `<rect x="${X(satX)}" y="${mT}" width="${X(xmax) - X(satX)}" height="${ph}" fill="rgba(159,176,192,.10)"/>` +
        (sat >= xmin ? `<line x1="${X(sat)}" y1="${mT}" x2="${X(sat)}" y2="${mT + ph}" stroke="#9fb0c0" stroke-width="1" stroke-dasharray="3 3"/>` +
          `<text x="${X(sat) - 4}" y="${mT + 12}" fill="#9fb0c0" font-size="10" text-anchor="end">saturation</text>` : "")
      : "";
    const cid = "ipc" + Math.random().toString(36).slice(2, 7);
    const dash = `stroke="var(--accent)" stroke-width="2" fill="none" stroke-dasharray="4 4" opacity=".7"`;
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block;margin-top:12px" ` +
      `font-family="-apple-system,Segoe UI,Roboto,sans-serif" role="img" ` +
      `aria-label="Predicted perceived intensity versus concentration">` +
      `<defs><clipPath id="${cid}"><rect x="${mL}" y="${mT}" width="${pw}" height="${ph}"/></clipPath></defs>` +
      `<rect x="${mL}" y="${mT}" width="${pw}" height="${ph}" fill="#0c141c" stroke="#2b3a49"/>` +
      `<g clip-path="url(#${cid})">${grid}${satShade}` +
      `<polyline points="${pts(xmin, tr.lo)}" ${dash}/>` +
      `<polyline points="${pts(tr.hi, xmax)}" ${dash}/>` +
      `<polyline points="${pts(tr.lo, tr.hi)}" stroke="var(--accent)" stroke-width="2.5" fill="none"/>` +
      `</g>` + axis +
      `<text x="${mL + pw / 2}" y="${H - 5}" fill="#9fb0c0" font-size="11" text-anchor="middle">concentration in air (ppm, log scale)</text>` +
      `<text transform="translate(12,${mT + ph / 2}) rotate(-90)" fill="#9fb0c0" font-size="11" text-anchor="middle">perceived intensity (gLMS)</text>` +
      `</svg>`;
  }

  // ---- Toxicity reference (Toxtree lookup) ----
  // toxicity.json holds Toxtree's classes for every molecule the site knows (see
  // scripts/build_toxicity.py). TTC hierarchy (Kroes/Munro; the R app's README, not
  // its code): a mutagenicity alert -- Ames (ISS) or the Gradient rules -- wins, then a
  // Gradient supplemental class, then the Revised Cramer class.
  let byFlatTox = new Map(); // InChIKey skeleton -> most conservative record
  function ensureTox() {
    if (!toxPromise) {
      toxPromise = fetch(asset("data/toxicity.json")).then((r) => r.json()).then((d) => {
        TOX = d;
        for (const rec of d.mols) {
          byKeyTox.set(rec.i, rec);
          // The trees ignore stereochemistry, so a stereo-specific query can use the
          // unspecified form's classes.
          const f = rec.i.split("-")[0], cur = byFlatTox.get(f);
          if (!cur || toxTTC(rec) < toxTTC(cur)) byFlatTox.set(f, rec);
        }
      }).catch(() => {});
    }
    return toxPromise;
  }

  function toxTTC(rec) {
    if (rec.g === "m" || rec.a) return 1.5;
    if (TTC_BY_CRAMER[rec.g] != null) return TTC_BY_CRAMER[rec.g];
    return TTC_BY_CRAMER[rec.c] != null ? TTC_BY_CRAMER[rec.c] : null;
  }

  function toxBasis(rec) {
    if (rec.a) {
      const names = rec.a.map((id) => (TOX.meta.sa || {})[id] || id);
      return `a structural alert for mutagenicity (Ames, ISS): ${names.map(escapeHtml).join("; ")}`;
    }
    if (rec.g === "m") return "a Gradient supplemental rule flagging it as a mutagen";
    if (rec.g) {
      const cr = rec.c && rec.c !== rec.g ? `; the Revised Cramer tree alone gives ${CRAMER_LABEL[rec.c]}` : "";
      return `a Gradient supplemental rule, ${CRAMER_LABEL[rec.g]}${cr}`;
    }
    return `the Revised Cramer tree, ${CRAMER_LABEL[rec.c]}`;
  }

  // Vapor pressure for the sniff count: your value > physchem.json > Mayhew dataset.
  function toxVP(ctx) {
    if (ctx.manual && ctx.manual.vp != null) return { vp: ctx.manual.vp, src: "your value", pred: false };
    const phys = lookupPhys(ctx.ikey);
    if (phys && phys.v != null) return { vp: phys.v, src: physSource(phys, "vs"), pred: physPredicted(phys.vs) };
    if (ctx.hit && ctx.hit.vp != null) return { vp: ctx.hit.vp, src: "Mayhew et al. dataset", pred: false };
    return null;
  }

  function sniffsToTTC(ttc, vp, mw) {
    if (ttc == null || !(vp > 0) || !(mw > 0)) return null;
    // µg of saturated (neat) headspace inhaled per 0.5 L sniff.
    const massPerSniff = (vp * SNIFF_L / (TOX_R * TOX_T)) * mw * 1e6;
    return { ttc, massPerSniff, sniffs: ttc / massPerSniff };
  }

  function fmtSniffs(n) {
    if (n >= 1000) return Math.round(n).toLocaleString();
    if (n >= 10) return n.toFixed(0);
    if (n >= 1) return n.toFixed(1);
    if (n >= 0.01) return n.toFixed(3);
    return n.toExponential(1);
  }

  // Compact number: scientific for very small / very large, else ~4 sig figs.
  function fmtSci(x) {
    if (x == null || !isFinite(x)) return "—";
    if (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5)) return x.toExponential(2);
    return String(Number(x.toPrecision(4)));
  }

  function renderToxicity(ctx) {
    const box = $("tox-box");
    if (!box) return;
    if (!TOX) box.innerHTML = `<p class="hint" style="margin:0">Looking up hazard classes…</p>`;
    Promise.all([ensurePhyschem(), ensureTox()]).then(() => {
      if (ctx !== current) return;
      if (!TOX) {
        box.innerHTML = `<p class="hint" style="margin:0">The toxicological reference table could not be loaded.</p>`;
        return;
      }
      const { ikey } = ctx;
      const rec = (ikey && (byKeyTox.get(ikey) || byFlatTox.get(ikey.split("-")[0]))) || null;
      if (!rec) {
        box.innerHTML =
          `<p class="hint" style="margin:0">This molecule hasn't been run through Toxtree yet, ` +
          `so no TTC figure is shown. (Lookup only for now — a predictive version is planned.)</p>`;
        return;
      }
      const ttc = toxTTC(rec);
      if (ttc == null) {
        box.innerHTML = `<p class="hint" style="margin:0">Toxtree could not classify this molecule, so no TTC figure is shown.</p>`;
        return;
      }
      const basis = `TTC = ${ttc} µg/person/day, from ${toxBasis(rec)}.`;
      const v = toxVP(ctx);
      const r = v && sniffsToTTC(ttc, v.vp, ctx.desc.mw);
      let lead, detail;
      if (!r) {
        lead = "No vapor pressure available, so the sniff count can't be computed";
        detail = `${basis} Enter a vapor pressure above to compute it.`;
      } else {
        const count = fmtSniffs(r.sniffs);
        lead = r.sniffs < 1
          ? `A single sniff of the neat headspace already exceeds its TTC (equivalent to ~${count} sniffs)`
          : `Reaches its TTC after <strong>${count}</strong> sniff${r.sniffs === 1 ? "" : "s"} of neat headspace`;
        detail = `${basis} Vapor pressure = ${fmtSci(v.vp)} mmHg (${escapeHtml(v.src)}); a 0.5 L sniff ` +
          `of the saturated headspace carries ~${fmtSci(r.massPerSniff)} µg.` +
          (v.pred ? ` Predicted vapor pressures can be off by 10× or more, and the sniff count scales with it.` : "");
      }
      box.innerHTML =
        `<div class="model" style="border:none;padding-top:0">` +
        `<div class="name">${lead} <span class="tag dataset">Toxtree</span></div>` +
        `<div class="verdict-line">${detail}</div>` +
        `<div class="verdict-line" style="color:var(--muted)">A comparative reference point only — it counts how many sniffs of ` +
        `undiluted headspace would together equal the daily Threshold of Toxicological Concern. Not a safety determination, ` +
        `exposure limit, or recommendation.</div>` +
        `<div class="verdict-line" style="color:var(--muted)">The TTC is a generic screening threshold for chemicals ` +
        `without their own toxicity data, and is generally <strong>conservative</strong>. Structural alerts are ` +
        `screening flags, not test results. Where a molecule has substance-specific toxicological data, those data ` +
        `take precedence over this figure — they usually allow more exposure than the TTC, but not always.</div>` +
        `</div>`;
    });
  }

  function ruleOfThree(desc) {
    const mwOK = desc.mw != null && desc.mw >= RULE_MW_MIN && desc.mw <= RULE_MW_MAX;
    const hetOK = desc.nhet != null && desc.nhet < RULE_NHET_MAX;
    const odorous = mwOK && hetOK;
    // If it fails because it is too big or too polar, that is evidence of odorlessness.
    const tooHeavyOrPolar = (desc.mw != null && desc.mw > RULE_MW_MAX) ||
                            (desc.nhet != null && desc.nhet >= RULE_NHET_MAX);
    return { odorous, mwOK, hetOK, tooHeavyOrPolar, desc };
  }

  // Returns { verdict: 'odor'|'odorless'|null, ... } for the transport boundary model.
  function transportVerdict(vp, logp) {
    if (vp == null || !(vp > 0) || logp == null) return { verdict: null };
    const b = transportBoundaries(vp);
    const odorous = logp > b.low && logp < b.high;
    // lowMargin < 0: below the low-volatility boundary (too involatile / hydrophilic).
    return { verdict: odorous ? "odor" : "odorless", low: b.low, high: b.high, logp, vp,
             lowMargin: logp - b.low };
  }

  // ---- Transport plot (inside the odor panel) ----
  function renderTransport(ctx) {
    const box = $("transport-box");
    const { t, tv } = ctx.transport;
    if (t.logp != null && !$("man-logp").value) $("man-logp").placeholder = fmt(t.logp, 2);
    if (t.vp != null && !$("man-vp").value) $("man-vp").placeholder = fmtSci(t.vp);
    if (!tv.verdict) {
      box.innerHTML = `<p class="hint" style="margin:0">No vapor pressure available for this molecule, ` +
        `so there is nothing to plot. Enter a measured value below.</p>`;
      return;
    }
    box.innerHTML = transportPlotSVG(t.vp, t.logp, tv);
  }

  // Scatter of the two logistic boundaries in log10(VP)/logP space, with the
  // molecule plotted as a point inside or outside the odorous window.
  function transportPlotSVG(vp, logp, res) {
    const l = Math.log10(vp);
    const low = (x) => -1.72 * x - 9.10;
    const high = (x) => -1.61 * x + 8.17;
    const xmin = Math.min(-9, l - 1), xmax = Math.max(3, l + 1);
    const ymin = Math.min(-6, logp - 1), ymax = Math.max(10, logp + 1);
    const W = 480, H = 320, mL = 46, mR = 16, mT = 16, mB = 42;
    const pw = W - mL - mR, ph = H - mT - mB;
    const X = (x) => mL + (x - xmin) / (xmax - xmin) * pw;
    const Y = (y) => mT + (ymax - y) / (ymax - ymin) * ph;
    const band =
      `${X(xmin)},${Y(low(xmin))} ${X(xmax)},${Y(low(xmax))} ` +
      `${X(xmax)},${Y(high(xmax))} ${X(xmin)},${Y(high(xmin))}`;
    let grid = "", axis = "";
    for (const t of niceTicks(xmin, xmax, 7)) {
      const px = X(t);
      grid += `<line x1="${px}" y1="${mT}" x2="${px}" y2="${mT + ph}" stroke="#22303e"/>`;
      axis += `<text x="${px}" y="${mT + ph + 15}" fill="#9fb0c0" font-size="10" text-anchor="middle">${t}</text>`;
    }
    for (const t of niceTicks(ymin, ymax, 7)) {
      const py = Y(t);
      grid += `<line x1="${mL}" y1="${py}" x2="${mL + pw}" y2="${py}" stroke="#22303e"/>`;
      axis += `<text x="${mL - 6}" y="${py + 3}" fill="#9fb0c0" font-size="10" text-anchor="end">${t}</text>`;
    }
    const ptColor = res.verdict === "odor" ? "#e8534f" : "#3f8bd8";
    const cid = "tpc" + Math.random().toString(36).slice(2, 7);
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block;margin-top:12px" ` +
      `font-family="-apple-system,Segoe UI,Roboto,sans-serif">` +
      `<defs><clipPath id="${cid}"><rect x="${mL}" y="${mT}" width="${pw}" height="${ph}"/></clipPath></defs>` +
      `<rect x="${mL}" y="${mT}" width="${pw}" height="${ph}" fill="#0c141c" stroke="#2b3a49"/>` +
      `<g clip-path="url(#${cid})">${grid}` +
      `<polygon points="${band}" fill="rgba(63,180,90,.16)"/>` +
      `<line x1="${X(xmin)}" y1="${Y(low(xmin))}" x2="${X(xmax)}" y2="${Y(low(xmax))}" stroke="#7f93a8" stroke-width="1.5" stroke-dasharray="5 4"/>` +
      `<line x1="${X(xmin)}" y1="${Y(high(xmin))}" x2="${X(xmax)}" y2="${Y(high(xmax))}" stroke="#7f93a8" stroke-width="1.5" stroke-dasharray="5 4"/>` +
      `<circle cx="${X(l)}" cy="${Y(logp)}" r="6" fill="${ptColor}" stroke="#fff" stroke-width="1.5"/></g>` +
      axis +
      `<rect x="${mL + 8}" y="${mT + 7}" width="13" height="9" fill="rgba(63,180,90,.16)" stroke="#7f93a8" stroke-dasharray="2 2"/>` +
      `<text x="${mL + 25}" y="${mT + 15}" fill="#9fb0c0" font-size="10.5">odorous window</text>` +
      `<text x="${mL + pw / 2}" y="${H - 5}" fill="#9fb0c0" font-size="11" text-anchor="middle">log₁₀ vapor pressure (mmHg)</text>` +
      `<text transform="translate(12,${mT + ph / 2}) rotate(-90)" fill="#9fb0c0" font-size="11" text-anchor="middle">logP</text>` +
      `</svg>`;
  }

  // ---- utils ----
  // "Nice" axis ticks (1/2/5 × 10^n) spanning [min,max], ~count divisions.
  function niceTicks(min, max, count) {
    const span = max - min;
    if (!(span > 0)) return [min];
    let step = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = count * step / span;
    if (err <= 0.15) step *= 10; else if (err <= 0.35) step *= 5; else if (err <= 0.75) step *= 2;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) {
      ticks.push(Math.round(v * 1e6) / 1e6);
    }
    return ticks;
  }
  function num(x) { const n = typeof x === "number" ? x : parseFloat(x); return isFinite(n) ? n : null; }
  function fmt(x, d) { return x == null || !isFinite(x) ? "—" : Number(x).toFixed(d).replace(/^-(0\.?0*)$/, "$1"); }
  function yn(b) { return b ? "✔" : "✗"; }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();
