# Does it smell? 👃

A small web app that predicts whether a molecule is **odorous** or **odorless**
from its chemical structure. Enter a SMILES string or a common chemical name and
the page tells you whether the molecule is likely to have a smell — and why.

**Live site:** _enable GitHub Pages (see below), then it appears at_
`https://<user>.github.io/<repo>/`

## The science

This tool implements the odor-classification models from:

> Mayhew EJ, Arayata CJ, Gerkin RC, Lee BK, Magill JM, Snyder LL, Little KA,
> Yu CW, Mainland JD. **Transport features predict if a molecule is odorous.**
> *PNAS* 119(15):e2116576119 (2022). https://doi.org/10.1073/pnas.2116576119

The key insight of the paper: a molecule can only smell if it can physically
complete the journey to an olfactory receptor. That journey is governed by simple
transport properties — **volatility** (vapor pressure / boiling point) and
**hydrophobicity** (logP). Molecules that are too heavy or too polar never reach
the receptors and are therefore odorless.

Two of the paper's models are used here:

| Model | What it needs | Where it runs |
|-------|---------------|---------------|
| **Rule of three** — odorous if 30 ≤ MW ≤ 300 Da and fewer than 4 heteroatoms | structure only | your browser |
| **Transport boundary model** — logistic-regression boundaries in vapor‑pressure/logP space | a vapor-pressure value | your browser |

For any of the **1,924 molecules** in the study's curated dataset, the page also
shows the actual gradient-boosted **transport-ML probability** reported in the paper.

## How it works (no server required)

Everything runs client-side, so the site is a static page that GitHub Pages can host:

- **[RDKit.js](https://www.rdkit.org/)** (WebAssembly) parses SMILES, draws the 2‑D
  structure, and computes molecular weight, Crippen logP and heteroatom counts.
- **[PubChem PUG-REST](https://pubchem.ncbi.nlm.nih.gov/)** resolves chemical names
  to structures.
- `docs/data/molecules.json` is the paper's dataset (extracted from
  [Dataset S1](https://github.com/emayhew/OlfactorySpace)): SMILES, odor class,
  transport-ML probability, boiling point, vapor pressure, MW, heteroatoms and logP,
  keyed by InChIKey for exact lookup.

## Repository layout

```
docs/
  index.html          # the page
  style.css
  app.js              # prediction logic + RDKit.js + PubChem glue
  data/molecules.json # 1,924-molecule dataset from the paper
```

## Running locally

Because the page uses `fetch`, open it through a local web server (not `file://`):

```bash
cd docs
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Enabling GitHub Pages

In the repository: **Settings → Pages → Build and deployment →
Source: Deploy from a branch → Branch: `main` / folder: `/docs`**. The site goes
live at `https://<user>.github.io/<repo>/` within a minute or two.

## Attribution & licensing

- **Model, method and data:** © the authors of Mayhew *et al.* (2022). The paper
  states there are no restrictions on the availability of the dataset used here.
- **This web app code:** MIT License (see `LICENSE`).
- This is an **independent educational reimplementation** of the published model,
  not an official tool of the authors or their institutions. Predictions estimate
  odor *class*, not odor *quality* or *intensity*, and can be wrong — especially
  for molecules unlike those in the training set.
