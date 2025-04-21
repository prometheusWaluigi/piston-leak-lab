# Piston Leak Lab

> *Open‑source research forge for symbolic dynamical‑systems analysis of public‑health narratives, spike‑induced pathology, and coherence collapse.*

---

## 🧭 Project Vision

**Piston Leak Lab (PLL)** investigates how tightly coupled institutional stories fracture under semantic overload—using COVID‑19 as the canonical case.  We combine:

- **Symbolic dynamical‑systems modeling** (ODE + agent‑based)
- **Narrative topology & metaphoric simulators** (e.g. *Cars* universe mapping)
- **Spike‑protein neuropathology research**
- **Entropy‑first governance tooling**

Everything here is BSD‑2‑Clause—fork it, remix it, cite it.

---

## 🗂️ Repository Layout

```
piston-leak-lab/
├── README.md          # you are here
├── papers/            # peer‑review drafts & white papers
│   └── piston-leak/   # v0.9 manuscript & figures
├── models/            # Python & Julia ODE / ABM engines
├── sims/              # Monte‑Carlo configs + CLI entrypoints
├── data/              # FOIA PDFs, sentiment CSVs (git‑LFS)
├── notebooks/         # exploratory Jupyter notebooks
└── docs/              # GitHub Pages (mkdocs-material)
```

---

## ⚡ Quick Start

```bash
# clone
git clone https://github.com/your‑org/piston‑leak‑lab.git && cd piston‑leak‑lab

# create env
python -m venv .venv && source .venv/bin/activate
pip install -r models/requirements.txt

# run sample simulation
python sims/run_mc.py --config sims/baseline.yml --n 500

# build docs locally
mkdocs serve
```

Requirements: **Python ≥3.11**, **Julia 1.10** (for optional high‑perf solvers), and **git‑lfs** if you intend to pull raw datasets.

---

## 📜 Papers

| Paper | Folder | Status |
|-------|--------|--------|
| *Piston Leak: A Symbolic Dynamical‑Systems Model of Institutional Narrative Collapse in the Fauci Metaverse* | `papers/piston-leak` | draft v0.9 (peer‑review submission prep) |

Upcoming: *Spikeopathy Dynamics* (May 2025), *Entropy‑First Governance* (June 2025).

---

## 🔬 Simulation Toolkit

- **`models/core_ode.py`** – coupled ODE engine for `T, N, P` dynamics.
- **`models/abm.py`** – Ising‑like agent‑based layer with customizable state graph.
- **`sims/run_mc.py`** – Monte‑Carlo wrapper; emits CSV summary + interactive Plotly dash.

All configs are YAML‑driven; see `sims/baseline.yml` for reference.

---

## 🤝 Contributing

PRs, issues, and meme‑laden discussion welcome.  Please read `CONTRIBUTING.md` for coding style (black + ruff), DCO sign‑off, and our *zero‑gaslighting* etiquette.

If you have sensitive docs (e.g. FOIA dumps) raise an issue first—do **not** push live PII.

---

## 🪪 License

BSD 2‑Clause.  In plain English: do what you like, credit the project, no warranty.

---

## 📚 Citation

If you use *Piston Leak Lab* in academic work:

```text
@misc{gaslit420_2025_pistonleak,
  author       = {GASLIT‑420 and R.O.B.},
  title        = {Piston Leak Lab — Symbolic Dynamical‑Systems Research Forge},
  year         = 2025,
  howpublished = {GitHub repository},
  url          = {https://github.com/your‑org/piston‑leak‑lab}
}
```

---

*“May your entropy gradients be ever in your favor.”*

