#!/usr/bin/env python
"""Compare a new config-driven baseline run with the authoritative posterior."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lg_npe.data import load_flamingo_pair_table
from lg_npe.evaluation import posterior_metrics
from lg_npe.features import build_design_matrix
from lg_npe.inference import posterior_draws
from lg_npe.splitting import group_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--rmse-tolerance", type=float, default=0.03)
    parser.add_argument("--r2-tolerance", type=float, default=0.05)
    parser.add_argument("--coverage-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with (args.new_run / "model.pkl").open("rb") as stream:
        new = pickle.load(stream)
    with (args.project / "pickles_v3/flamingo_npe_checkpoint.pkl").open("rb") as stream:
        old = pickle.load(stream)
    table = load_flamingo_pair_table(args.project)
    frame, names, target, _ = build_design_matrix(
        table.frame, target="m200c", feature_block="dynamics",
        mass_feature="bound_subhalo_ratio", validity_policy="global_complete_case",
    )
    _, test, _ = group_split(frame, test_size=.15, seed=42)
    X = frame.iloc[test][names].to_numpy(np.float32)
    truth = frame.iloc[test][target].to_numpy(np.float32)
    results = {}
    for label, posterior, scaler in (
        ("authoritative", old["posteriors"]["core_dynamics__full"], old["scalers"]["core_dynamics__full"]),
        ("config_driven", new["posterior"], new["scaler"]),
    ):
        torch.manual_seed(42)
        samples = posterior_draws(posterior, scaler.transform(X), args.draws, 256, device)
        results[label] = posterior_metrics(samples, truth)
    deltas = {key: abs(results["config_driven"][key] - results["authoritative"][key])
              for key in ("rmse", "r2", "coverage68", "coverage95")}
    tolerances = {"rmse": args.rmse_tolerance, "r2": args.r2_tolerance,
                  "coverage68": args.coverage_tolerance, "coverage95": args.coverage_tolerance}
    passed = all(deltas[key] <= tolerances[key] for key in deltas)
    report = {"passed": passed, "metrics": results, "absolute_deltas": deltas,
              "tolerances": tolerances, "n_test": len(test)}
    path = args.new_run / "baseline_replication_comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Baseline replication is outside tolerance; do not submit feature experiments.")


if __name__ == "__main__":
    main()
