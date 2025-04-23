#!/usr/bin/env python3
"""
Generate a suite of Monte‑Carlo YAML configs for **Piston Leak Lab**
------------------------------------------------------------------
Create *ten* parameter‑tweaked simulation scenarios that explore
different sectors of the Trust–Entropy–Pressure landscape and inject
GASLIT‑AF perturbations.

Usage
-----
    python generate_montecarlo_configs.py \
        --baseline sims/baseline.yml \
        --out sims/generated

Arguments
~~~~~~~~~
--baseline   Path to an existing YAML config to inherit from (required).
--out        Directory where new configs are written (required).

The script will emit files like:
    sims/generated/
        01_high_transparency_ramp.yml
        02_authoritarian_clamp.yml
        ...
        10_believer_lockstep.yml

Install deps:
    pip install pyyaml

‣ Tip: add an *entry‑point* in `pyproject.toml` to let Poetry expose the
  command as `poetry run piston‑leak‑gen‑configs`.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Dict, Generator, Tuple

import yaml

Scenario = Tuple[str, str, Dict]

# ---------------------------------------------------------------------
# Scenario blueprints  (slug, title, overrides)
# ---------------------------------------------------------------------
SCENARIOS: list[Scenario] = [
    (
        "high_transparency_ramp",
        "Gradual Transparency Ramp‑Up (Recovery Test)",
        {
            "parameters": {
                "transparency_type": "adaptive",
                "base_transparency": 0.05,
                "transparency_adaptivity": 0.35,
                "leak_frequency": 0.02,
            }
        },
    ),
    (
        "authoritarian_clamp",
        "Hard Suppression / Low Transparency (Authoritarian Clamp)",
        {
            "parameters": {
                "k_policy": 1.2,
                "gamma1": 0.4,
                "beta1": 0.1,
                "transparency_value": 0.02,
                "noise_sigma": 0.05,
            }
        },
    ),
    (
        "leak_storm",
        "High‑Noise Leak Storm (Wikileaks Week)",
        {
            "parameters": {
                "noise_sigma": 0.25,
                "leak_frequency": 0.25,
                "alpha2": 0.6,
            }
        },
    ),
    (
        "policy_fatigue",
        "Policy Fatigue (Suppression Decay Stress)",
        {
            "parameters": {
                "delta": 0.35,
                "k_policy": 0.6,
            }
        },
    ),
    (
        "resilience_drive",
        "Transparency + Trust Resilience Drive",
        {
            "parameters": {
                "beta1": 0.5,
                "transparency_type": "reactive",
                "trust_threshold": 0.5,
                "low_trust_transparency": 0.45,
            },
            "ensemble": {"runs": 300, "perturb_parameters": True},
        },
    ),
    (
        "gaslit_meltdown",
        "GASLIT‑AF Meltdown (Entropy Surge + Genetic Fragility)",
        {
            "parameters": {
                "alpha1": 0.8,  # trust erodes faster
                "beta2": 0.15,  # entropy dampening weaker
                "entropy_critical": 0.9,
                "noise_sigma": 0.2,
                "leak_frequency": 0.15,
            }
        },
    ),
    (
        "legacy_collapse",
        "Legacy Narrative Collapse (Long Horizon)",
        {
            "time": {"t_max": 1460},  # 4‑year horizon
            "parameters": {"epsilon_overlap": 0.1},
        },
    ),
    (
        "network_heterogeneity",
        "Scale‑Free vs Small‑World Network Heterogeneity",
        {
            "parameters": {
                "network": {
                    "type": "barabasi_albert",
                    "n_agents": 5000,
                    "k": 4,
                }
            }
        },
    ),
    (
        "skeptic_surge",
        "Skeptic Influence Surge (Influence Tilt)",
        {
            "parameters": {
                "influence_field": {
                    "believer_weight": 0.9,
                    "skeptic_weight": 1.5,
                    "agnostic_weight": 0.7,
                },
                "init_believer_prob": 0.45,
                "init_skeptic_prob": 0.35,
                "init_agnostic_prob": 0.20,
            }
        },
    ),
    (
        "believer_lockstep",
        "Believer Lockstep (Echo‑Chamber Stress)",
        {
            "parameters": {
                "influence_field": {
                    "believer_weight": 1.4,
                    "skeptic_weight": 0.8,
                    "agnostic_weight": 0.6,
                },
                "network": {
                    "type": "erdos_renyi",
                    "n_agents": 2500,
                    "k": 8,
                },
            }
        },
    ),
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def deep_update(target: dict, changes: dict):
    """Recursively merge ``changes`` into ``target`` (in‑place)."""
    for key, val in changes.items():
        if isinstance(val, dict):
            target.setdefault(key, {})
            deep_update(target[key], val)
        else:
            target[key] = val


def build_configs(baseline_path: Path) -> Generator[tuple[int, str, dict], None, None]:
    base_cfg = yaml.safe_load(baseline_path.read_text())

    for idx, (slug, title, overrides) in enumerate(SCENARIOS, 1):
        cfg = deepcopy(base_cfg)
        deep_update(cfg, overrides)

        # Annotate meta section
        cfg.setdefault("meta", {})
        cfg["meta"].update(
            {
                "description": title,
                "parent_baseline": baseline_path.name,
                "version": f"0.1.{idx}",
                "date": str(date.today()),
            }
        )

        yield idx, slug, cfg

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="generate_montecarlo_configs.py",
        description="Generate a battery of Monte‑Carlo scenario configs.",
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline YAML config")
    parser.add_argument("--out", required=True, help="Output directory for generated configs")
    ns = parser.parse_args(argv)

    baseline_path = Path(ns.baseline).expanduser().resolve()
    out_dir = Path(ns.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, slug, cfg in build_configs(baseline_path):
        fname = f"{idx:02d}_{slug}.yml"
        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        # Print a neat relative path if possible
        try:
            display_path = out_path.relative_to(Path.cwd())
        except ValueError:
            display_path = out_path
        print(f"→  {display_path}")

    print("\nDone! 10 configs ready for simulation.")


if __name__ == "__main__":
    main()
