"""Reconstruct the corrected sample without replacing any frozen v1/v2 products.

Run from the project root: python -B TNG300/rebuild_comparison.py
TNG_BASE and LG_PROJECT override the catalogue and project locations.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import illustris_python as il
import numpy as np

# Use the TNG package even when imported by a notebook that has FLAMINGO's
# unrelated `analogues` package on its path.
PROJECT = Path(os.environ.get("LG_PROJECT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT))
from TNG300.analogues import AnalogueSample, SelectionConfig, FilterConfig
sys.path.insert(0, str(PROJECT / "analogues"))
from review_fixes import RUN_VERSION, tng_volume_mpc3


def save_new_array(path, value):
    if path.exists():
        np.testing.assert_array_equal(np.load(path), value,
                                      err_msg=f"Refusing to replace different data: {path}")
    else:
        with path.open("xb") as stream:
            np.save(stream, value)


def export_comparison(analogue_sample, base_path, project=PROJECT):
    project = Path(project)
    old = project / "analogues/tng_for_comparison"
    out = project / "analogues/tng_for_comparison_v3"
    cache = project / "analogues/cache/tng_features_v3"
    old_cache = project / "analogues/cache/run_001_mstar2e10"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    sample, pairs = analogue_sample._sample, analogue_sample.pairs
    header = il.groupcat.loadHeader(str(base_path), 99)
    h = float(header["HubbleParam"])
    volume = tng_volume_mpc3(header)
    np.testing.assert_array_equal(sample.keep_idx, np.load(old / "keep_idx.npy"),
                                  err_msg="Sample ordering changed: custom features require ID remapping.")
    # Preserve the existing matched-aperture feature definitions and ordering.
    for filename in ("metallicity.npy", "vel_disp.npy", "tng_metallicity.pkl"):
        if not (out / filename).exists():
            shutil.copy2(old / filename, out / filename)
    for filename in ("tng_sigma_star.pkl", "tng_kappa_corot.pkl", "metallicity_log_zscore.pkl"):
        if not (cache / filename).exists():
            shutil.copy2(old_cache / filename, cache / filename)
    # Refresh only the TNG descriptive z-score for the corrected pair population.
    # Neural-network features use the dex array and their fitted StandardScaler.
    with (old_cache / "metallicity_log_zscore.pkl").open("rb") as stream:
        met_transform = pickle.load(stream)
    dex = met_transform["tng_met_dex"]
    population = dex[np.r_[pairs.i, pairs.j]]
    finite_population = population[np.isfinite(population)]
    met_transform["mu_tng"] = finite_population.mean()
    met_transform["std_tng"] = finite_population.std(ddof=1)
    met_transform["tng_met_scaled"] = (dex - met_transform["mu_tng"]) / met_transform["std_tng"]
    with (cache / "metallicity_log_zscore.pkl").open("wb") as stream:
        pickle.dump(met_transform, stream)
    values = {
        "keep_idx": sample.keep_idx, "stellar_mass": sample.m_stellar,
        "grnr": sample.grnr, "g_r": sample.g_band - sample.r_band,
        "pairs_i": pairs.i, "pairs_j": pairs.j,
        "separation": pairs.separation, "vr": pairs.vr, "vt": pairs.vt,
        # find_pairs retains its historical extra conversion; undo it exactly once.
        "log_m200c": pairs.log_tot_virial_mass - 10 - np.log10(1 / h),
        "log_mass_ratio": pairs.log_mass_ratio,
        "have_same_host": pairs.have_same_host,
        "is_blue_blue": pairs.is_blue_blue, "is_red_red": pairs.is_red_red,
    }
    expected_mass = np.where(pairs.have_same_host, sample.m200c[pairs.i],
                             sample.m200c[pairs.i] + sample.m200c[pairs.j])
    np.testing.assert_allclose(values["log_m200c"], np.log10(expected_mass), atol=3e-6)
    assert np.all(values["log_mass_ratio"] <= 0)
    for name, value in values.items():
        save_new_array(out / f"{name}.npy", value)
    old_pairs = set(zip(np.load(old / "pairs_i.npy").tolist(), np.load(old / "pairs_j.npy").tolist()))
    new_pairs = set(zip(pairs.i.tolist(), pairs.j.tolist()))
    global_ids = sample.keep_idx[np.unique(np.r_[pairs.i, pairs.j])]
    missing = {}
    for filename in ("tng_sigma_star.pkl", "tng_kappa_corot.pkl"):
        with (cache / filename).open("rb") as stream:
            lookup = pickle.load(stream)
        missing[filename] = [int(i) for i in global_ids if not np.isfinite(lookup.get(int(i), np.nan))]
    met = np.load(out / "metallicity.npy")
    missing["metallicity.npy"] = global_ids[~np.isfinite(met[np.unique(np.r_[pairs.i, pairs.j])])].tolist()
    digest = hashlib.sha256()
    for path in sorted(out.glob("*.npy")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    manifest = {
        "run_version": RUN_VERSION, "dataset_sha256": digest.hexdigest(),
        "catalogue": str(Path(base_path).resolve()), "snapshot": 99,
        "h": h, "BoxSize_ckpc_h": float(header["BoxSize"]), "volume_mpc3": volume,
        "selection": asdict(analogue_sample._selection_config),
        "filters": asdict(analogue_sample._filter_config),
        "intruder_mass": "total bound SubhaloMass", "vr": "peculiar km/s",
        "sample_size": len(sample.keep_idx), "old_pair_count": len(old_pairs),
        "pair_count": len(new_pairs), "retained_pairs": len(old_pairs & new_pairs),
        "removed_pairs": len(old_pairs - new_pairs), "added_pairs": len(new_pairs - old_pairs),
        "same_host": int(pairs.have_same_host.sum()),
        "different_host": int((~pairs.have_same_host.astype(bool)).sum()),
        "pair_density_mpc3": len(new_pairs) / volume,
        "missing_custom_features": missing,
    }
    target = out / "selection_manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError("Existing v3 manifest differs; preserve it and choose a new version.")
    else:
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "missing_custom_features"}, indent=2))
    print("Missing custom features:", {k: len(v) for k, v in missing.items()})
    if any(missing.values()):
        raise RuntimeError("Refresh the missing v3 custom features before any model fit.")
    return out


def main():
    base = os.environ.get("TNG_BASE", r"<TNG_BASE>")
    analogue_sample = AnalogueSample(
        base_path=base, snap=99, verbose=True,
        selection_config=SelectionConfig(2e10, 5e11, 500., 1000., .68),
        filter_config=FilterConfig(vr_min=-400, vr_max=0, vt_min=0, vt_max=300,
            v_tot_min=0, v_tot_max=500, third_massive_factor=1.5,
            density_radius=2000, intruder_factor=.5),
    )
    export_comparison(analogue_sample, base)


if __name__ == "__main__":
    main()
