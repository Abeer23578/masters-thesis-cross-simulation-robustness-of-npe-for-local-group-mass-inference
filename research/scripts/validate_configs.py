#!/usr/bin/env python
"""Validate every committed experiment against the authoritative real data once."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lg_npe.config import load_experiment_config
from lg_npe.data import load_flamingo_pair_table, load_tng_pair_table
from lg_npe.features import build_design_matrix
from lg_npe.splitting import group_split


def main():
    source = load_flamingo_pair_table(PROJECT)
    transfer = load_tng_pair_table(PROJECT)
    rows = []
    primary_source_indices = None
    primary_transfer_indices = None
    for path in sorted((PROJECT / "configs/npe").rglob("*.json")):
        config = load_experiment_config(path)
        src, features, target, src_counts = build_design_matrix(
            source.frame, target=config["target"], feature_block=config["feature_block"],
            mass_feature=config["mass_feature"], validity_policy=config["validity_policy"],
        )
        tng, tng_features, tng_target, tng_counts = build_design_matrix(
            transfer.frame, target=config["target"], feature_block=config["feature_block"],
            mass_feature=config["mass_feature"], validity_policy=config["validity_policy"],
        )
        if features != tng_features or target != tng_target:
            raise RuntimeError(f"Source/transfer schema mismatch in {path}")
        train, test, groups = group_split(src, config["split"]["test_size"], config["seed"])
        if primary_source_indices is None:
            primary_source_indices = src["pair_index"].to_numpy()
            primary_transfer_indices = tng["pair_index"].to_numpy()
        elif config["validity_policy"] == "global_complete_case":
            # The approved primary comparison must keep one population across mass representations.
            if not (src["pair_index"].to_numpy() == primary_source_indices).all():
                raise RuntimeError(f"Primary FLAMINGO population changed in {path}")
            if not (tng["pair_index"].to_numpy() == primary_transfer_indices).all():
                raise RuntimeError(f"Primary TNG population changed in {path}")
        rows.append({
            "config": str(path.relative_to(PROJECT)), "experiment_id": config["experiment_id"],
            "target": config["target"], "features": features,
            "source_pairs": len(src), "source_train": len(train), "source_test": len(test),
            "transfer_pairs": len(tng), "source_counts": src_counts, "transfer_counts": tng_counts,
        })
    output = PROJECT / "out/mass_comparison/config_validation.json"
    output.write_text(json.dumps({"status": "passed", "configs": rows}, indent=2), encoding="utf-8")
    print(f"Validated {len(rows)} configs against real data -> {output}")


if __name__ == "__main__":
    main()
