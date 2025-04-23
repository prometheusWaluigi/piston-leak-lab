#!/usr/bin/env python3
"""run_all_configs.py – batch‑runner for *Piston Leak* Monte‑Carlo configs.

*New* ⚡ **v0.2** – auto‑routes results so runs don’t clobber each other.
Each scenario’s outputs now land in `<result_root>/<slug>/`.

Usage
-----
```bash
python run_all_configs.py \
  --dir sims/generated \
  --n 200 \
  --exe "poetry run run-mc" \
  --result-root results \
  --extra --progress --save-summary
```

Arguments
~~~~~~~~~
--dir          Directory containing YAML scenario configs (default: `sims/generated`).
--n            Monte‑Carlo iterations per scenario (default: 100).
--exe          Simulator CLI command (default: `poetry run run-mc`).
--result-root  Where to place per‑scenario subfolders (default: `results`).
--extra        Everything after this flag passes verbatim to each run‑mc call.

The script infers the *slug* from filenames like `03_leak_storm.yml` → `leak_storm` and
creates `results/leak_storm/` (or your chosen root path) to keep artifacts separate.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # noqa: D401
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Batch‑run Monte‑Carlo configs without overwrite chaos.")
    parser.add_argument("--dir", default="sims/generated", help="Directory with YAML configs")
    parser.add_argument("--n", type=int, default=100, help="Runs per scenario (default: 100)")
    parser.add_argument("--exe", default="poetry run run-mc", help="Simulator CLI command")
    parser.add_argument("--result-root", default="results", help="Root folder for outputs (default: results)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Flags forwarded to run‑mc")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    args = parse_args(argv)

    cfg_dir = Path(args.dir).expanduser().resolve()
    if not cfg_dir.is_dir():
        sys.exit(f"[ERR] Directory not found: {cfg_dir}")

    cfg_paths = sorted(cfg_dir.glob("*.yml"))
    if not cfg_paths:
        sys.exit(f"[ERR] No .yml configs in {cfg_dir}")

    exe_parts = shlex.split(args.exe)
    extra_parts = args.extra or []
    result_root = Path(args.result_root).expanduser().resolve()

    for cfg in cfg_paths:
        # Derive slug: strip numeric prefix if present → `03_leak_storm.yml` → `leak_storm`
        stem = cfg.stem
        slug = stem.split("_", 1)[1] if "_" in stem else stem
        out_dir = result_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [*exe_parts, "--config", str(cfg), "--n", str(args.n), "--out", str(out_dir), *extra_parts]
        print("\n▶", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] run‑mc exited with status {exc.returncode} for {cfg}")

    print("\nAll Monte‑Carlo ensembles complete – results in", result_root)


if __name__ == "__main__":
    main()
