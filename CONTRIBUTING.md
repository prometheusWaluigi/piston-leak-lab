### 1. `CONTRIBUTING.md`

```markdown
# Contributing to **Piston Leak Lab**

Welcome, daring epistemic spelunker!  
Whether you wield Python, Julia, R, or meme‑alchemy, your pull‑requests are
welcome.  Follow these **entropy‑aware protocols** to keep the lattice tight.

---

## 🐍 1. Local Setup

```bash
git clone https://github.com/your‑org/piston-leak-lab.git
cd piston-leak-lab
python -m venv .venv && source .venv/bin/activate
pip install -r models/requirements.txt   # black, ruff, numpy, etc.
```

Need Julia for the high‑perf solvers?  
See `docs/julia-setup.md`.

---

## 📐 2. Coding Style

| Domain        | Linter / Formatter | Rule of Thumb                    |
|---------------|--------------------|----------------------------------|
| Python        | **black**, **ruff**| “If black disagrees, black wins.”|
| YAML / JSON   | `yamllint -d relaxed` | Two‑space indent, no tabs.      |
| LaTeX         | `latexindent`      | One sentence per line is ❤️.     |
| Markdown      | `mdformat`         | Wrap at 100 chars, keep lists tidy|

---

## 📜 3. Commit & PR Etiquette

* **Conventional Commits** (`feat:`, `fix:`, `docs:` …) keep the log parseable.
* Reference an Issue # in the description (or open one first).
* Squash merges only—entropy minimization, baby.
* Every PR **must** retain at least one joke; bots will flag humorless diffs.

---

## 🤖 4. DCO Sign‑Off

Add this trailer to every commit:

```
Signed-off-by: Your Name <email@domain.com>
```

It’s the Developer Certificate of Origin.  Lawyers love it. 🔏

---

## 🏗️ 5. Test Matrix

Run unit tests before pushing:

```bash
pytest -q
```

CI will also spin a 50‑run Monte‑Carlo smoke test (`sims/test_mc.yml`)
to ensure the attractor fractal hasn’t face‑planted.

---

## 🛡️ 6. Sensitive Data

* No live PII or proprietary FOIA docs.  
* Drop redacted versions in `data/redacted/` and link the source FOIA request.

---

## 🧅 7. Layers of Review

1. **Automated checks** – lint, tests, license compliance.
2. **Entropy check** – a maintainer ensures the PR *reduces* narrative entropy.
3. **Final meme‑scan** – at least one maintainer reacts with 🦆 before merge.

---

### Thanks!

By contributing you agree to the **IFRL v1.0**.  
Now go forth and leak those pistons responsibly.