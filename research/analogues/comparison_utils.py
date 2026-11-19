"""Read-only input checks and descriptive utilities for the v3 comparison notebooks."""
from pathlib import Path
import json

import numpy as np
from scipy.stats import wasserstein_distance
from review_fixes import configure_run


def validate_selection(data, manifest):
    """Check sample-index alignment before any CDF or comparison is computed."""
    pi, pj = data["pairs_i"], data["pairs_j"]
    n = len(pi)
    n_gal = len(data["keep_idx"])
    if n != manifest["pair_count"] or n_gal != manifest["sample_size"]:
        raise ValueError("TNG counts disagree with the corrected selection manifest.")
    for name in ("pairs_i", "pairs_j"):
        idx = data[name]
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"{name} must contain one-dimensional sample indices.")
        if len(idx) != n or np.any(idx < 0) or np.any(idx >= n_gal):
            raise ValueError(f"{name} is not aligned with keep_idx.")
    for name in ("separation", "vr", "vt", "log_mass_ratio", "log_m200c", "have_same_host"):
        if data[name].shape != (n,) or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid or misaligned TNG pair feature: {name}")
    for name in ("grnr", "g_r", "stellar_mass", "metallicity", "vel_disp"):
        if data[name].shape != (n_gal,):
            raise ValueError(f"Invalid TNG sample-level array: {name}")
    if not np.isin(data["have_same_host"], [0, 1]).all():
        raise ValueError("TNG host membership must contain only booleans or zero/one.")
    same = data["have_same_host"].astype(bool)
    if not np.array_equal(same, data["grnr"][pi] == data["grnr"][pj]):
        raise ValueError("TNG host flags disagree with the pair-to-galaxy mapping.")
    if int(same.sum()) != manifest["same_host"] or int((~same).sum()) != manifest["different_host"]:
        raise ValueError("TNG host populations disagree with the selection manifest.")
    if np.any(data["log_mass_ratio"] > 0):
        raise ValueError("log_mass_ratio must be log10(smaller/larger total bound mass).")
    data["have_same_host"] = same
    return data


def load_comparison_inputs(project, notebook_name):
    project = Path(project)
    results = configure_run(project)
    tng_dir = project / "analogues/tng_for_comparison_v3"
    manifest = json.loads((tng_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    data = validate_selection({p.stem: np.load(p) for p in tng_dir.glob("*.npy")}, manifest)
    plots = results / "figures" / notebook_name
    plots.mkdir(parents=True, exist_ok=True)
    print(f"Corrected TNG: {len(data['pairs_i']):,} pairs; "
          f"{manifest['same_host']:,} same-host, {manifest['different_host']:,} different-host.")
    print("TNG pair features: kpc, peculiar km/s, log10(Msun), log10(smaller/larger bound mass).")
    return results, plots, tng_dir, data, manifest


def finite_lookup(cache, ids, feature):
    values = np.array([cache.get(int(i), np.nan) for i in ids], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing/nonfinite {feature} for {int((~np.isfinite(values)).sum())} current pair members.")
    return values


def normalized_wasserstein(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    scale = np.sqrt((len(a) * a.var() + len(b) * b.var()) / (len(a) + len(b)))
    distance = wasserstein_distance(a, b)
    return distance / scale if scale > 0 else (0.0 if distance == 0 else np.inf)


def group_blocks(groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Provide one connected-host group per nonempty row.")
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(groups[order][1:] != groups[order][:-1]) + 1
    return np.split(order, boundaries)


def resample_blocks(blocks, rng):
    """Sample connected-host groups, retaining every row in each sampled group."""
    return np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
