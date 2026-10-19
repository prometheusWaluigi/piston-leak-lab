> # 🧪 VIBE CODE — read this first
>
> This repository is **vibe code** from the **Sharan homelab** — built fast, in play, by [Ketan Sharan Raturi](https://github.com/prometheusWaluigi) together with his AI agents. It may be brilliant, broken, or (most likely) both at once. Nothing here is production software; everything here is an artifact of curiosity.
>
> **Looking for the maintained work?** The living, curated projects live in the Sharan family:
> 📚 [**catalog**](https://github.com/prometheusWaluigi/catalog) — every public repo, grouped & labeled ·
> 🛠️ [**sharan-skills**](https://github.com/prometheusWaluigi/sharan-skills) — the lab's agent skills ·
> 🎛️ [**arc-a770-strata**](https://github.com/prometheusWaluigi/arc-a770-strata) — real models on an Intel Arc A770
>
> **Check there for updates.** This repo is a snapshot of a moment, not a living project.
>
> **This repo:** a symbolic dynamical-systems research forge for institutional narrative collapse — a coupled Trust–Narrative–Suppression ODE engine plus an Ising-like agent-based layer, Monte-Carlo driven from YAML configs, with the *Piston Leak* paper (draft v0.9) in `papers/`. **Status:** research experiment · quick start is `poetry install` then `run-mc --config sims/baseline.yml --n 25` for a sample ensemble.

# Piston Leak Lab

> *Open‑source research forge for symbolic dynamical‑systems analysis of public‑health narratives, spike‑induced pathology, and coherence collapse.*

---

## 🧭 Project Vision

**Piston Leak Lab (PLL)** investigates how tightly coupled institutional stories fracture under semantic overload—using COVID‑19 as the canonical case.  We combine:

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
├── papers/            # peer‑review drafts & white papers
│   └── piston-leak/   # v0.9 manuscript & figures
├── models/            # Python & Julia ODE / ABM engines
├── sims/              # Monte‑Carlo configs + CLI entrypoints
├── data/              # FOIA PDFs, sentiment CSVs (git‑LFS)
├── notebooks/         # exploratory Jupyter notebooks
└── docs/              # GitHub Pages (mkdocs-material)
```

---

## ⚡ Quick Start

```bash
# clone
git clone https://github.com/gaslit420/piston‑leak‑lab.git && cd piston‑leak‑lab

# install with poetry
poetry install

# activate environment
poetry shell

# run a quick sample simulation (smaller ensemble)
run-mc --config sims/baseline.yml --n 25 --out results/quickstart/

# or without activating shell
poetry run run-mc --config sims/baseline.yml --n 25 --out results/quickstart/

# full baseline run (takes longer)
poetry run run-mc --config sims/baseline.yml --n 500 --out results/baseline/

# build docs locally
poetry run mkdocs serve
```

**Requirements**: 
- **Python ≥3.11**
- **Poetry ≥1.8** (package management)
- **git‑lfs** (only if you want to pull the raw dataset files)
- **Intel oneAPI** (optional, for high‑performance GPU/CPU simulations)

For Intel oneAPI acceleration, install with:
```bash
poetry install -E oneapi
```

---

## 📜 Papers

| Paper | Folder | Status |
|-------|--------|--------|
| *Piston Leak: A Symbolic Dynamical‑Systems Model of Institutional Narrative Collapse in the Fauci Metaverse* | `papers/piston-leak` | draft v0.9 (peer‑review submission prep) |

Upcoming: *Spikeopathy Dynamics* (May 2025), *Entropy‑First Governance* (June 2025).

---

## 🔬 Simulation Toolkit

- **`models/core_ode.py`** – coupled ODE engine for `T, N, P` dynamics.
- **`models/abm.py`** – Ising‑like agent‑based layer with customizable state graph.
- **`sims/run_mc.py`** – Monte‑Carlo wrapper; emits CSV summary + interactive Plotly dash.

All configs are YAML‑driven; see `sims/baseline.yml` for reference.

---

## 📊 Visualization & Analysis

The simulation framework includes powerful visualization capabilities:

- Static plots for trust trajectories, phase space, and attractor metrics
- Interactive dashboards for parameter exploration
- Temporal evolution analysis for collapse vs. recovery comparison

Example visualization:

```python
from sims.visualization import create_interactive_dashboard

# After running simulations
create_interactive_dashboard(results, summary, "output_path/", "timestamp")
```

---

## 🤝 Contributing

PRs, issues, and meme‑laden discussion welcome.  Please read `CONTRIBUTING.md` for coding style (black + ruff), DCO sign‑off, and our *zero‑gaslighting* etiquette.

If you have sensitive docs (e.g. FOIA dumps) raise an issue first—do **not** push live PII.

---

## 🪪 License

BSD 2‑Clause.  In plain English: do what you like, credit the project, no warranty.

---

## 📚 Citation

If you use *Piston Leak Lab* in academic work:

```text
@misc{gaslit420_2025_pistonleak,
  author       = {GASLIT‑420 and R.O.B.},
  title        = {Piston Leak Lab — Symbolic Dynamical‑Systems Research Forge},
  year         = 2025,
  howpublished = {GitHub repository},
  url          = {https://github.com/gaslit420/piston‑leak‑lab}
}
```

---

## 🌀 Build Status

[![piston‑leak‑ci](https://github.com/your-github-username/piston-leak-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/your-github-username/piston-leak-lab/actions/workflows/ci.yml)

👉 *Don't forget to replace 'your-github-username' with your actual GitHub username after forking the repository.*

---

*"May your entropy gradients be ever in your favor."*
---

*Part of the [Sharan lab public catalog](https://github.com/prometheusWaluigi/catalog) · vibe code from the homelab — brilliant, broken, or both (see the banner up top).*
