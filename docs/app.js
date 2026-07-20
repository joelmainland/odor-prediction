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

  let RDKit = null;
  let DATA = null;           // array of molecule records
  let byKey = new Map();     // InChIKey -> record
  let byCan = new Map();     // canonical SMILES -> record
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
    el.innerHTML =
      `<span class="pill">${escapeHtml(primary.source)}</span>` +
      `<p class="big">${icon} ${escapeHtml(primary.label)}</p>` +
      `<p class="sub">${escapeHtml(primary.conf)}` +
      (hit ? ` · <span style="color:var(--accent)">in study dataset (ground truth: ${escapeHtml(hit.odor)})</span>` : "") +
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
      `</div></div>`;
    if (logp != null && !$("man-logp").value) $("man-logp").value = fmt(logp, 3);
  }

  // ---- utils ----
  function num(x) { const n = typeof x === "number" ? x : parseFloat(x); return isFinite(n) ? n : null; }
  function fmt(x, d) { return x == null || !isFinite(x) ? "—" : Number(x).toFixed(d); }
  function yn(b) { return b ? "✔" : "✗"; }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();
