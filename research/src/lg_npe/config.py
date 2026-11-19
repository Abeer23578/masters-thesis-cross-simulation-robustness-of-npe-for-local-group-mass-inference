from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .constants import HISTORICAL_NPE_DEFAULTS
from .validation import validate_experiment_config


DEFAULT_CONFIG = {
    "simulation_train": "flamingo",
    "simulation_test": "tng",
    "target": "m200c",
    "feature_block": "dynamics",
    "mass_feature": "bound_subhalo_ratio",
    "validity_policy": "global_complete_case",
    "seed": 42,
    "split": {"method": "connected_host_components", "test_size": 0.15},
    "scaler": {"type": "standard", "fit_on": "source_train"},
    "npe": HISTORICAL_NPE_DEFAULTS,
    "posterior_draws": 200,
    "evaluation_batch_size": 256,
    "limits": {"max_train_rows": None, "max_source_test_rows": None, "max_transfer_rows": None},
    "output_root": "out/mass_comparison",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path)
    user = json.loads(path.read_text(encoding="utf-8"))
    config = _deep_merge(DEFAULT_CONFIG, user)
    config["config_path"] = str(path.resolve())
    validate_experiment_config(config)
    return config
