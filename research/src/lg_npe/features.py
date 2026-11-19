from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    BASELINE_COMPLETE_CASE_FEATURES,
    BASE_DYNAMICS,
    BLOCK_ADDITIONS,
    MASS_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from .validation import validate_no_bound_mass_leakage


def ordered_stellar_masses(
    mass_a: np.ndarray,
    mass_b: np.ndarray,
    id_a: np.ndarray | None = None,
    id_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return primary/secondary stellar masses and whether endpoint a is primary.

    Primary is the larger stellar mass. Exact ties use the smaller stable identifier.
    """
    mass_a = np.asarray(mass_a)
    mass_b = np.asarray(mass_b)
    if id_a is None:
        id_a = np.arange(len(mass_a)) * 2
    if id_b is None:
        id_b = np.arange(len(mass_b)) * 2 + 1
    id_a, id_b = np.asarray(id_a), np.asarray(id_b)
    a_primary = (mass_a > mass_b) | ((mass_a == mass_b) & (id_a <= id_b))
    primary = np.where(a_primary, mass_a, mass_b)
    secondary = np.where(a_primary, mass_b, mass_a)
    return primary, secondary, a_primary


def add_stellar_representations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    primary, secondary, a_primary = ordered_stellar_masses(
        frame["mstar_a"].to_numpy(), frame["mstar_b"].to_numpy(),
        frame["id_a"].to_numpy(), frame["id_b"].to_numpy(),
    )
    valid = (primary > 0) & (secondary > 0)
    frame["stellar_primary_is_a"] = a_primary
    frame["log_mstar_primary"] = np.where(valid, np.log10(primary), np.nan)
    frame["log_mstar_secondary"] = np.where(valid, np.log10(secondary), np.nan)
    frame["log_stellar_ratio"] = np.where(valid, np.log10(secondary / primary), np.nan)
    frame["log_stellar_sum"] = np.where(valid, np.log10(primary + secondary), np.nan)
    return frame


def feature_names(feature_block: str, mass_feature: str) -> list[str]:
    return BASE_DYNAMICS + MASS_FEATURE_COLUMNS[mass_feature] + BLOCK_ADDITIONS[feature_block]


def build_design_matrix(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_block: str,
    mass_feature: str,
    validity_policy: str,
) -> tuple[pd.DataFrame, list[str], str, dict[str, int]]:
    frame = add_stellar_representations(frame)
    names = feature_names(feature_block, mass_feature)
    target_column = TARGET_COLUMNS[target]
    validate_no_bound_mass_leakage(target, names)

    counts = {"input_pairs": len(frame)}
    if validity_policy == "global_complete_case":
        required = BASELINE_COMPLETE_CASE_FEATURES + [target_column]
    elif validity_policy == "per_configuration":
        required = names + [target_column]
    else:
        raise ValueError(f"Unknown validity policy: {validity_policy}")
    mask = np.isfinite(frame[required].to_numpy(dtype=np.float64)).all(axis=1)
    counts["finite_required_columns"] = int(mask.sum())
    counts["global_complete_case"] = int(mask.sum()) if validity_policy == "global_complete_case" else None
    selected = frame.loc[mask].reset_index(drop=True)
    finite_config = np.isfinite(selected[names + [target_column]].to_numpy(dtype=np.float64)).all(axis=1)
    counts["finite_configuration_features_and_target"] = int(finite_config.sum())
    if not finite_config.all():
        if validity_policy == "global_complete_case":
            raise ValueError(
                "The approved global complete-case population contains invalid values for this configuration; "
                "do not silently change its population."
            )
        selected = selected.loc[finite_config].reset_index(drop=True)
    counts["final_pairs"] = len(selected)
    return selected, names, target_column, counts
