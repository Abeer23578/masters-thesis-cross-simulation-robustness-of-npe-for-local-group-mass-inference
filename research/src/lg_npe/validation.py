from __future__ import annotations

from .constants import (
    BLOCK_ADDITIONS,
    BOUND_MASS_DERIVED_FEATURES,
    MASS_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


def validate_experiment_config(config: dict) -> None:
    target = config.get("target")
    mass_feature = config.get("mass_feature")
    block = config.get("feature_block")
    policy = config.get("validity_policy")

    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target {target!r}; expected one of {sorted(TARGET_COLUMNS)}")
    if mass_feature not in MASS_FEATURE_COLUMNS:
        raise ValueError(
            f"Unknown mass_feature {mass_feature!r}; expected one of {sorted(MASS_FEATURE_COLUMNS)}"
        )
    if block not in BLOCK_ADDITIONS:
        raise ValueError(f"Unknown feature_block {block!r}; expected one of {sorted(BLOCK_ADDITIONS)}")
    if policy not in {"global_complete_case", "per_configuration"}:
        raise ValueError("validity_policy must be global_complete_case or per_configuration")
    if config.get("simulation_train") != "flamingo" or config.get("simulation_test") != "tng":
        raise ValueError("The approved initial design is FLAMINGO training and TNG transfer evaluation.")
    if config.get("split", {}).get("method") != "connected_host_components":
        raise ValueError("Only the historical connected_host_components split is currently valid.")
    if config.get("scaler", {}).get("fit_on") != "source_train":
        raise ValueError("The transfer experiment must use the source-training scaler.")
    if "num_layers" in config.get("npe", {}):
        raise ValueError(
            "num_layers is intentionally rejected: it was saved by Optuna but unused by the historical model."
        )
    if target == "bound_subhalo_sum" and mass_feature == "bound_subhalo_ratio":
        raise ValueError("Leakage: bound_subhalo_ratio cannot be used for bound_subhalo_sum.")


def validate_no_bound_mass_leakage(target: str, feature_names: list[str]) -> None:
    if target != "bound_subhalo_sum":
        return
    direct = sorted(set(feature_names) & BOUND_MASS_DERIVED_FEATURES)
    suspicious = sorted(
        name for name in feature_names
        if ("bound" in name.lower() or "msub" in name.lower()) and name not in direct
    )
    offenders = direct + suspicious
    if offenders:
        raise ValueError(
            "Leakage: bound_subhalo_sum target cannot use bound-mass-derived inputs: "
            + ", ".join(offenders)
        )
