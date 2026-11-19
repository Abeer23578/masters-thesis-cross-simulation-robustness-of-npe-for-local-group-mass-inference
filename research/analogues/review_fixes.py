"""Shared, versioned conventions for the corrected z=0 thesis analysis.

The affine scalar MAF is conditional Gaussian in log mass. This module changes
observation conditioning and diagnostics, not the FLAMINGO training features.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

RUN_VERSION = "v3-isolation-peculiar-calibration"
H0_BY_SIM = {"flamingo": 68.1, "tng": 67.74}
NON_NEGATIVE_FEATURES = {
    "separation", "vt", "sigma_mean", "sigma_diff", "kappa_mean",
    "kappa_diff", "gr_diff", "met_diff",
}
MODEL_DESCRIPTION = "conditional Gaussian in log10 mass, restricted by posterior support"


def configure_run(project):
    """Reject stale data/results rather than silently resuming another sample."""
    project = Path(project)
    manifest_path = project / "analogues/tng_for_comparison_v3/selection_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Build the corrected TNG sample first: python TNG300/rebuild_comparison.py"
        )
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    if selection.get("run_version") != RUN_VERSION:
        raise ValueError("The TNG selection manifest is not the corrected v3 version.")
    digest = hashlib.sha256()
    for path in sorted(manifest_path.parent.glob("*.npy")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != selection["dataset_sha256"]:
        raise ValueError("TNG arrays changed after reconstruction; rebuild and review their manifest.")
    feature_digest = hashlib.sha256()
    for name in ("tng_sigma_star.pkl", "tng_kappa_corot.pkl", "metallicity_log_zscore.pkl"):
        feature_digest.update(name.encode())
        feature_digest.update((feature_cache_dir(project) / name).read_bytes())
    result_dir = project / "pickles_v3"
    result_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "run_version": RUN_VERSION,
        "tng_dataset_sha256": selection["dataset_sha256"],
        "custom_features_sha256": feature_digest.hexdigest(),
        "training_vr": "peculiar_relative",
        "lg_input_vr": "physical_relative_converted_before_scaling",
        "H0_km_s_Mpc": H0_BY_SIM,
        "model_family": MODEL_DESCRIPTION,
    }
    target = result_dir / "run_manifest.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != expected:
            raise ValueError("pickles_v3 belongs to another dataset/convention; use a new result directory.")
    else:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2)
    return result_dir


def feature_cache_dir(project):
    return Path(project) / "analogues/cache/tng_features_v3"


def require_legacy_environment(project):
    """Prevent incompatible pickle loads and altered legacy holdout membership."""
    from importlib.metadata import version
    pins = {}
    for line in (Path(project) / "env_roihu.txt").read_text().splitlines():
        if "==" in line:
            name, pinned = line.split("==", 1)
            pins[name] = pinned
    names = ("torch", "sbi", "scikit-learn", "numpy", "scipy")
    mismatches = {name: (version(name), pins[name]) for name in names
                  if version(name).split("+")[0] != pins[name].split("+")[0]}
    if mismatches:
        raise RuntimeError(
            "Use the original Roihu environment for legacy model reuse and FLAMINGO "
            f"holdouts. Installed/required versions: {mismatches}. "
            "Catalogue reconstruction and regression tests can run locally."
        )


def lg_observation(scenario="vdM2012"):
    """Physical observations. None marks an unavailable justified vt likelihood.

    Salomon's positive Gaussian and the ratio's upper-truncated Gaussian are
    explicitly approximations; their location/scale are not truncated moments.
    """
    if scenario not in {"vdM2012", "Salomon2021"}:
        raise ValueError(f"Unknown observation scenario: {scenario}")
    obs = {"separation": 765.0, "vr": -109.3, "vt": 17.0, "log_mass_ratio": -0.14}
    unc = {"separation": 15.0, "vr": 4.4, "vt": None, "log_mass_ratio": 0.20}
    if scenario == "Salomon2021":
        obs["vt"], unc["vt"] = 82.4, 31.2
    return obs, unc


def lg_to_peculiar(x_physical, feats, h0):
    """Convert raw observations, never scaled features or simulation features."""
    x = np.array(x_physical, dtype=np.float64, copy=True)
    if x.ndim not in (1, 2) or x.shape[-1] != len(feats):
        raise ValueError("Expected a feature vector or a batch with the stated feature order.")
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError("H0 must be positive in km/s/Mpc.")
    if "vr" in feats:
        if "separation" not in feats:
            raise ValueError("Radial conversion requires separation in kpc.")
        x[..., feats.index("vr")] -= h0 * x[..., feats.index("separation")] / 1000.0
    return x.astype(np.float32)


def draw_noisy_feature(feat, mean, std, size, rng):
    if std is None:
        raise ValueError(f"No justified uncertainty model for {feat}; use a point scenario.")
    if not np.isfinite(std) or std < 0:
        raise ValueError("The underlying normal scale must be finite and non-negative.")
    if std == 0:
        if (feat == "log_mass_ratio" and mean > 0) or (feat in NON_NEGATIVE_FEATURES and mean < 0):
            raise ValueError("A fixed observation is outside the feature support.")
        return np.full(size, mean, dtype=float)
    if feat == "log_mass_ratio":
        return truncnorm.rvs(-np.inf, -mean / std, loc=mean, scale=std,
                             size=size, random_state=rng)
    if feat in NON_NEGATIVE_FEATURES:
        return truncnorm.rvs(-mean / std, np.inf, loc=mean, scale=std,
                             size=size, random_state=rng)
    return rng.normal(mean, std, size=size)


def draw_lg_features(obs, unc, feats, h0, n_draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.column_stack([
        draw_noisy_feature(f, obs[f], unc[f], n_draws, rng) for f in feats
    ])
    # Use the distance from the same draw, preserving the induced correlation.
    return lg_to_peculiar(raw, feats, h0)


def sample_lg_point(posterior, scaler, feats, obs, *, h0, device, n=10_000):
    import torch
    raw = np.array([[obs[f] for f in feats]], dtype=np.float32)
    x = torch.tensor(scaler.transform(lg_to_peculiar(raw, feats, h0)), device=device)
    with torch.no_grad():
        return posterior.sample((n,), x=x).cpu().numpy().reshape(-1)


def sample_lg_marginalized(posterior, scaler, feats, obs, unc, *, h0, device,
                           n_draws=1000, n_per=10, seed=0):
    import torch
    features = draw_lg_features(obs, unc, feats, h0, n_draws, seed)
    x = torch.tensor(scaler.transform(features), device=device)
    with torch.no_grad():
        return posterior.sample_batched((n_per,), x=x).cpu().numpy().reshape(-1)


def calibration_summary(samples, theta_true):
    """Descriptive finite-rank and central-interval checks; no IID KS p-values.

    Rows of samples are posterior draws; columns are evaluation pairs. Coverage
    has Monte Carlo error from finite draws. Pair dependence requires grouped
    resampling for confidence intervals, not independent Bernoulli error bars.
    """
    samples = np.asarray(samples)
    theta_true = np.asarray(theta_true)
    if samples.ndim != 2 or samples.shape[1] != theta_true.size or samples.shape[0] < 2:
        raise ValueError("Expected (posterior draws, evaluation pairs) and one truth per pair.")
    if theta_true.size == 0 or not np.isfinite(samples).all() or not np.isfinite(theta_true).all():
        raise ValueError("Calibration requires nonempty finite samples and truths.")
    K, n = samples.shape
    ranks = (samples < theta_true[None, :]).sum(axis=0)
    levels = np.array([0.50, 0.68, 0.80, 0.90, 0.95])
    lo = np.quantile(samples, (1 - levels) / 2, axis=0)
    hi = np.quantile(samples, (1 + levels) / 2, axis=0)
    covered = (theta_true[None, :] >= lo) & (theta_true[None, :] <= hi)
    counts = np.bincount(ranks, minlength=K + 1)
    reference_cdf = np.arange(1, K + 2) / (K + 1)
    residual = np.median(samples, axis=0) - theta_true
    std = samples.std(axis=0)
    beta = np.divide(np.abs(residual), std, out=np.full(n, np.inf), where=std > 0)
    return {
        "ranks": ranks, "beta": beta, "median_beta": float(np.median(beta)),
        "n_test": n, "n_posterior_samples": K,
        "rank_cdf_distance": float(np.max(np.abs(counts.cumsum() / n - reference_cdf))),
        "coverage_levels": levels, "coverage": covered.mean(axis=1),
        "covered": covered, "coverage68": float(covered[1].mean()),
        "coverage95": float(covered[-1].mean()),
        "mean_interval_width": (hi - lo).mean(axis=1),
        "median_residual": residual, "bias": float(residual.mean()),
        "run_version": RUN_VERSION,
    }


def tng_volume_mpc3(header):
    if not np.isclose(float(header.get("Redshift", 0.0)), 0.0):
        raise ValueError("This thesis volume convention is implemented for z=0 only.")
    return (float(header["BoxSize"]) / float(header["HubbleParam"]) / 1000.0) ** 3
