/* Does it smell? — odor prediction from molecular structure.
   Implements the transport-feature models of Mayhew et al., PNAS 2022.
   All chemistry runs client-side via RDKit.js (WebAssembly). */

(function () {
  "use strict";

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
  let byKeyTox = new Map();  // InChIKey -> toxicity record
  let byCanTox = new Map();  // canonical SMILES -> toxicity record
  let byKeyQ = new Map();    // InChIKey -> odor-quality record
  let byCanQ = new Map();    // canonical SMILES -> odor-quality record
  let qualPromise = null;    // lazy-load promise for quality.json (1.1 MB)
  let qSeq = 0;              // guards async quality renders against races
  let current = null;        // { mol data for manual recompute }

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
    fetch("data/molecules.json").then((r) => r.json()).then((d) => {
      DATA = d;
      for (const rec of d) {
        if (rec.ikey) byKey.set(rec.ikey, rec);
        if (rec.can) byCan.set(rec.can, rec);
      }
    }),
    // Toxicity reference set is optional — degrade gracefully if it fails.
    fetch("data/toxicity.json").then((r) => r.json()).then((d) => {
      for (const rec of d) {
        if (rec.ikey) byKeyTox.set(rec.ikey, rec);
        if (rec.can) byCanTox.set(rec.can, rec);
      }
    }).catch(() => {}),
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
    if (current) renderTransport(current, true);
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

  async function lookupName(name) {
    const base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/";
    const url = base + encodeURIComponent(name) + "/property/IsomericSMILES,Title/JSON";
    const r = await fetch(url);
    if (!r.ok) return null;
    const j = await r.json();
    const p = j && j.PropertyTable && j.PropertyTable.Properties && j.PropertyTable.Properties[0];
    if (!p) return null;
    // PubChem has changed this column name over time; accept any SMILES variant.
    const smi = p.IsomericSMILES || p.SMILES || p.ConnectivitySMILES || p.CanonicalSMILES;
    if (!smi) return null;
    return { smiles: smi, name: p.Title || name };
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
    let hit = (ikey && byKey.get(ikey)) || byCan.get(canon) || null;

    // Structure depiction
    try { $("structure-svg").innerHTML = mol.get_svg(230, 190); } catch (e) { $("structure-svg").innerHTML = ""; }
    $("mol-name").textContent = displayName ? displayName : canon;

    // Properties table
    const props = [
      ["Molecular weight", fmt(desc.mw, 2) + " Da"],
      ["logP (Crippen)", fmt(desc.logp, 2)],
      ["Heteroatoms", desc.nhet == null ? "—" : String(desc.nhet)],
      ["Heavy atoms", desc.heavy == null ? "—" : String(desc.heavy)],
      ["Canonical SMILES", `<code>${escapeHtml(canon)}</code>`],
    ];
    if (hit) {
      if (hit.vp != null) props.push(["Vapor pressure (dataset)", fmt(hit.vp, 4) + " mmHg"]);
      if (hit.logp != null) props.push(["logP (dataset, Moriguchi)", fmt(hit.logp, 2)]);
    }
    $("props").innerHTML = props.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

    // ---- Model verdicts ----
    const flags = structuralFlags(mol, canon);
    const rule = ruleOfThree(desc);
    const datasetProb = hit ? hit.p : null;

    // Store state for manual transport recompute
    current = { desc, hit, canon };

    // Primary verdict priority: dataset transport-ML > rule of three.
    let primary;
    if (datasetProb != null) {
      primary = {
        odorous: datasetProb >= 0.5,
        label: datasetProb >= 0.5 ? "Odorous" : "Odorless",
        conf: `${Math.round(Math.max(datasetProb, 1 - datasetProb) * 100)}% confidence`,
        source: "Transport-ML model (from the paper's dataset)",
        maybe: false,
      };
    } else if (flags.salt || flags.inorganic) {
      // The transport rule of three is validated only for organic molecules.
      primary = {
        odorous: false,
        label: "Uncertain",
        conf: flags.salt
          ? "Salt / multi-component species — transport rules don't apply cleanly"
          : "Inorganic / metal-containing — outside the model's chemical space",
        source: "Rule of three not applicable",
        maybe: true,
      };
    } else {
      primary = {
        odorous: rule.odorous,
        label: rule.odorous ? "Probably odorous" : (rule.tooHeavyOrPolar ? "Probably odorless" : "Uncertain"),
        conf: rule.odorous ? "Rule of three satisfied" : "Rule of three not satisfied",
        source: "Rule of three (structure-only estimate)",
        maybe: !rule.odorous && !rule.tooHeavyOrPolar,
      };
    }
    renderVerdict(primary, hit);
    renderBasis(rule, datasetProb, hit, flags);
    renderTransport(current, false);
    renderToxicity(ikey, canon);
    renderQuality(ikey, canon);
  }

  // ---- Odor quality (lookup only, lazy-loaded) ----
  // Published-atlas descriptors aggregated via Pyrfume. quality.json is large,
  // so it is fetched on first use rather than at boot.
  function ensureQuality() {
    if (!qualPromise) {
      qualPromise = fetch("data/quality.json").then((r) => r.json()).then((d) => {
        for (const rec of d) {
          if (rec.ikey) byKeyQ.set(rec.ikey, rec);
          if (rec.can) byCanQ.set(rec.can, rec);
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

  // ---- Toxicity reference (lookup only) ----
  // TTC hierarchy (Kroes/Munro; matches the R app's README, not its code):
  // a genotoxicity alert takes precedence over the Cramer class.
  function toxTTC(rec) {
    if (rec.gradient === "Mutagen") return 1.5;
    if (rec.mutagen === "NO") return 1.5;   // "NO" = Ames structural alert present
    return TTC_BY_CRAMER[rec.cramer] != null ? TTC_BY_CRAMER[rec.cramer] : null;
  }

  function sniffsToTTC(rec) {
    const ttc = toxTTC(rec);
    if (ttc == null || !(rec.vp > 0) || !(rec.mw > 0)) return null;
    // µg of saturated (neat) headspace inhaled per 0.5 L sniff.
    const massPerSniff = (rec.vp * SNIFF_L / (TOX_R * TOX_T)) * rec.mw * 1e6;
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

  function renderToxicity(ikey, canon) {
    const box = $("tox-box");
    if (!box) return;
    const rec = (ikey && byKeyTox.get(ikey)) || byCanTox.get(canon) || null;
    if (!rec) {
      box.innerHTML =
        `<p class="hint" style="margin:0">This molecule isn't in the toxicological reference set, ` +
        `so no TTC figure is shown. (Lookup only for now — a predictive version is planned.)</p>`;
      return;
    }
    const r = sniffsToTTC(rec);
    if (!r) {
      box.innerHTML = `<p class="hint" style="margin:0">Insufficient data to compute a TTC reference for this molecule.</p>`;
      return;
    }
    const alert = rec.mutagen === "NO";
    const basis = alert
      ? "a structural alert for mutagenicity (Ames)"
      : `Cramer ${CRAMER_LABEL[rec.cramer] || rec.cramer}`;
    const count = fmtSniffs(r.sniffs);
    const lead = r.sniffs < 1
      ? `A single sniff of the neat headspace already exceeds its TTC (equivalent to ~${count} sniffs)`
      : `Reaches its TTC after <strong>${count}</strong> sniff${r.sniffs === 1 ? "" : "s"} of neat headspace`;
    box.innerHTML =
      `<div class="model" style="border:none;padding-top:0">` +
      `<div class="name">${lead} <span class="tag dataset">dataset</span></div>` +
      `<div class="verdict-line">` +
      `TTC = ${r.ttc} µg/person/day, from ${basis}. ` +
      `Vapor pressure = ${fmtSci(rec.vp)} mmHg; a 0.5 L sniff of the saturated headspace carries ~${fmtSci(r.massPerSniff)} µg.` +
      `</div>` +
      `<div class="verdict-line" style="color:var(--muted)">A comparative reference point only — it counts how many sniffs of ` +
      `undiluted headspace would together equal the daily Threshold of Toxicological Concern. Not a safety determination, ` +
      `exposure limit, or recommendation.</div>` +
      `<div class="verdict-line" style="color:var(--muted)">The TTC is a deliberately <strong>conservative</strong> generic ` +
      `screening threshold used when no substance-specific data exist. Where a molecule has its own toxicological safety ` +
      `data, the actual tolerable exposure is often far higher — so this figure can substantially <em>understate</em> how ` +
      `much can be tolerated (vanillin, for example, has established safe-use levels well above what this count implies).</div>` +
      `</div>`;
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
    return { verdict: odorous ? "odor" : "odorless", low: b.low, high: b.high, logp, vp };
  }

  // ---- Rendering ----
  function renderVerdict(primary, hit) {
    const el = $("verdict");
    const cls = primary.maybe ? "v-maybe" : (primary.odorous ? "v-odor" : "v-odorless");
    el.parentElement.className = "card verdict-card";
    el.className = "verdict " + cls;
    const icon = primary.maybe ? "❓" : (primary.odorous ? "👃" : "🚫");
    // Ground-truth label follows the site's odor colour code: Odor = red, Odorless = blue.
    const gtColor = hit && hit.odor === "Odor" ? "var(--odor)" : "var(--odorless)";
    el.innerHTML =
      `<span class="pill">${escapeHtml(primary.source)}</span>` +
      `<p class="big">${icon} ${escapeHtml(primary.label)}</p>` +
      `<p class="sub">${escapeHtml(primary.conf)}` +
      (hit ? ` · <span style="color:${gtColor}">in study dataset (ground truth: ${escapeHtml(hit.odor)})</span>` : "") +
      `</p>`;
    $("result").classList.remove("hidden");
  }

  function renderBasis(rule, datasetProb, hit, flags) {
    const rows = [];

    // Rule of three
    {
      const na = flags && (flags.salt || flags.inorganic);
      const tag = na
        ? `<span class="tag na">Not applicable</span>`
        : rule.odorous
        ? `<span class="tag odor">Odorous</span>`
        : (rule.tooHeavyOrPolar ? `<span class="tag odorless">Odorless-leaning</span>` : `<span class="tag na">Inconclusive</span>`);
      const mw = rule.desc.mw == null ? "?" : fmt(rule.desc.mw, 1);
      const het = rule.desc.nhet == null ? "?" : rule.desc.nhet;
      const note = na
        ? `<div class="verdict-line" style="color:var(--muted)">The rule of three is validated only for organic molecules; ` +
          `it does not apply to ${flags.salt ? "salts / multi-component species" : "inorganic or metal-containing compounds"}.</div>`
        : "";
      rows.push(
        `<div class="model"><div class="name">Rule of three ${tag}</div>` +
        `<div class="verdict-line">MW ${mw} Da (need 30–300: ${yn(rule.mwOK)}), ` +
        `heteroatoms ${het} (need &lt;4: ${yn(rule.hetOK)}).</div>${note}</div>`
      );
    }

    // Transport-ML probability (dataset only)
    {
      let body;
      if (datasetProb != null) {
        const pct = Math.round(datasetProb * 100);
        const tag = datasetProb >= 0.5 ? `<span class="tag odor">Odorous</span>` : `<span class="tag odorless">Odorless</span>`;
        body =
          `<div class="name">Transport-ML probability ${tag} <span class="tag dataset">dataset</span></div>` +
          `<div class="prob-bar"><div style="width:${pct}%"></div></div>` +
          `<div class="verdict-line">p(odorous) = ${datasetProb.toFixed(3)} — the gradient-boosted model value reported in the paper.</div>`;
      } else {
        body =
          `<div class="name">Transport-ML probability <span class="tag na">not available</span></div>` +
          `<div class="verdict-line">Only provided for the 1,924 molecules in the study's curated dataset ` +
          `(it needs experimental transport features that can't be computed from structure alone).</div>`;
      }
      rows.push(`<div class="model">${body}</div>`);
    }

    $("basis").innerHTML = rows.join("");
  }

  function renderTransport(state, useManual) {
    const box = $("transport-box");
    const desc = state.desc, hit = state.hit;

    let vp = null, logp = null, vpSource = "", logpSource = "";
    if (useManual) {
      const mv = parseFloat($("man-vp").value);
      const ml = parseFloat($("man-logp").value);
      if (!isNaN(mv)) { vp = mv; vpSource = "your value"; }
      if (!isNaN(ml)) { logp = ml; logpSource = "your value"; }
    }
    if (vp == null && hit && hit.vp != null) { vp = hit.vp; vpSource = "dataset"; }
    if (logp == null && hit && hit.logp != null) { logp = hit.logp; logpSource = "dataset (Moriguchi)"; }
    if (logp == null && desc.logp != null) { logp = desc.logp; logpSource = "computed (Crippen)"; }

    const res = transportVerdict(vp, logp);
    if (res.verdict == null) {
      box.innerHTML =
        `<p class="hint" style="margin:0">No vapor pressure available for this molecule, so the boundary ` +
        `model can't be evaluated. Enter a measured vapor pressure below to run it.</p>`;
      // prefill logp for convenience
      if (logp != null && !$("man-logp").value) $("man-logp").value = fmt(logp, 3);
      return;
    }

    const tag = res.verdict === "odor" ? `<span class="tag odor">Odorous</span>` : `<span class="tag odorless">Odorless</span>`;
    box.innerHTML =
      `<div class="model" style="border:none;padding-top:0">` +
      `<div class="name">Boundary verdict ${tag}</div>` +
      `<div class="verdict-line">` +
      `Vapor pressure = ${fmt(vp, 4)} mmHg <small style="color:var(--muted)">(${vpSource})</small>, ` +
      `logP = ${fmt(logp, 2)} <small style="color:var(--muted)">(${logpSource})</small>.<br>` +
      `Odorous window for this volatility: ${fmt(res.low, 2)} &lt; logP &lt; ${fmt(res.high, 2)}. ` +
      `This molecule ${res.verdict === "odor" ? "falls inside" : "falls outside"} it.` +
      `</div>` +
      transportPlotSVG(vp, logp, res) +
      `</div>`;
    if (logp != null && !$("man-logp").value) $("man-logp").value = fmt(logp, 3);
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
  function fmt(x, d) { return x == null || !isFinite(x) ? "—" : Number(x).toFixed(d); }
  function yn(b) { return b ? "✔" : "✗"; }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();
