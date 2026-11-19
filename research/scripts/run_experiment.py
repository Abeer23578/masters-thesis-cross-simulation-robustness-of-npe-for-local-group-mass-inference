#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lg_npe.config import load_experiment_config
from lg_npe.data import load_flamingo_pair_table, load_tng_pair_table
from lg_npe.evaluation import posterior_metrics
from lg_npe.features import build_design_matrix
from lg_npe.inference import posterior_draws, train_npe
from lg_npe.registry import environment_record, write_json
from lg_npe.splitting import group_split


def limited_indices(indices, limit, seed):
    indices = np.asarray(indices)
    if limit is None or len(indices) <= int(limit):
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, size=int(limit), replace=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config/split without training")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    source = load_flamingo_pair_table(args.project)
    transfer = load_tng_pair_table(args.project)
    source_df, names, target_col, source_counts = build_design_matrix(
        source.frame, target=config["target"], feature_block=config["feature_block"],
        mass_feature=config["mass_feature"], validity_policy=config["validity_policy"],
    )
    transfer_df, transfer_names, transfer_target, transfer_counts = build_design_matrix(
        transfer.frame, target=config["target"], feature_block=config["feature_block"],
        mass_feature=config["mass_feature"], validity_policy=config["validity_policy"],
    )
    if names != transfer_names or target_col != transfer_target:
        raise RuntimeError("Source and transfer feature definitions disagree")
    train_idx, test_idx, groups = group_split(
        source_df, test_size=float(config["split"]["test_size"]), seed=int(config["seed"]),
    )
    limits = config["limits"]
    train_idx = limited_indices(train_idx, limits["max_train_rows"], config["seed"])
    test_idx = limited_indices(test_idx, limits["max_source_test_rows"], config["seed"] + 1)
    transfer_idx = limited_indices(np.arange(len(transfer_df)), limits["max_transfer_rows"], config["seed"] + 2)
    summary = {
        "experiment_id": config["experiment_id"], "config": config,
        "simulation_train": config["simulation_train"],
        "simulation_test": config["simulation_test"],
        "target_name": config["target"],
        "target_column": target_col,
        "target_definition": {
            "m200c": (
                "log10(host M200c/Msun) for same-host pairs; "
                "log10((host1 M200c + host2 M200c)/Msun) for different-host pairs"
            ),
            "bound_subhalo_sum": "log10((Msub_bound_1 + Msub_bound_2)/Msun) for every pair",
        }[config["target"]],
        "feature_configuration": {
            "feature_block": config["feature_block"],
            "mass_feature": config["mass_feature"],
            "validity_policy": config["validity_policy"],
        },
        "feature_names": names,
        "stellar_mass_definition": {
            "flamingo": "SOAP BoundSubhalo/StellarMass; all bound stars; physical Msun",
            "tng": "SUBFIND SubhaloMassType[:,4]; all bound stellar mass; physical Msun after 1e10/h",
            "ordering": "primary=max(Mstar_a,Mstar_b), secondary=min; stable ID tie-break",
            "systematic": "HBT-HERONS versus SUBFIND structure-finder difference; no correction applied",
        },
        "same_diff_host_handling": (
            "pooled source training with connected-host-component split; pooled and same/different-host metrics"
        ),
        "sample_counts": {"source": source_counts, "transfer": transfer_counts,
                          "source_train": len(train_idx), "source_test": len(test_idx),
                          "transfer_evaluation": len(transfer_idx)},
        "same_diff_counts": {
            "source": source_df.same_host.value_counts().to_dict(),
            "transfer": transfer_df.same_host.value_counts().to_dict(),
        },
        "provenance": {"source": source.provenance, "transfer": transfer.provenance,
                       "technical_note": "Saved Optuna num_layers=2 is deliberately unused."},
        "environment": environment_record(),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, default=str))
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.get("require_cuda", True) and device != "cuda":
        raise RuntimeError("This configuration requires CUDA")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(source_df.iloc[train_idx][names].to_numpy(np.float32))
    theta_train = source_df.iloc[train_idx][target_col].to_numpy(np.float32)
    posterior, inference, density = train_npe(X_train, theta_train, config, device)

    evaluations = {}
    for label, frame, idx in (("source_heldout", source_df, test_idx), ("transfer_tng", transfer_df, transfer_idx)):
        X = scaler.transform(frame.iloc[idx][names].to_numpy(np.float32))
        truth = frame.iloc[idx][target_col].to_numpy(np.float32)
        samples = posterior_draws(posterior, X, int(config["posterior_draws"]),
                                  int(config["evaluation_batch_size"]), device)
        same_host = frame.iloc[idx]["same_host"].to_numpy(bool)
        evaluations[label] = {"pooled": posterior_metrics(samples, truth)}
        if same_host.any():
            evaluations[label]["same_host"] = posterior_metrics(samples[:, same_host], truth[same_host])
        if (~same_host).any():
            evaluations[label]["different_host"] = posterior_metrics(samples[:, ~same_host], truth[~same_host])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.project / config["output_root"] / config["experiment_id"] / timestamp
    output.mkdir(parents=True, exist_ok=False)
    with (output / "model.pkl").open("wb") as stream:
        pickle.dump({"posterior": posterior, "density_estimator": density, "scaler": scaler,
                     "feature_names": names, "target": target_col, "config": config}, stream)
    np.savez_compressed(output / "split_indices.npz", train=train_idx, test=test_idx,
                        transfer=transfer_idx, source_groups=groups)
    summary["evaluation"] = evaluations
    summary["training_summary"] = {key: value[-1] if isinstance(value, list) and value else value
                                   for key, value in inference.summary.items()}
    summary["output_path"] = str(output)
    write_json(output / "run_manifest.json", summary)
    print(json.dumps({"output": str(output), "evaluation": evaluations}, indent=2))


if __name__ == "__main__":
    main()
