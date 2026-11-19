from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PairTable:
    simulation: str
    frame: pd.DataFrame
    stage_counts: dict[str, int]
    provenance: dict


def _finite_lookup(mapping: dict, keys: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray([mapping.get(int(key), np.nan) for key in keys], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing values for selected pair members")
    return values


def _load_bound_sidecar(project: Path, simulation: str) -> tuple[dict[str, np.ndarray], dict]:
    root = project / "data/derived/bound_subhalo_masses"
    npz_path = root / f"{simulation}_bound_subhalo_masses_v1.npz"
    manifest_path = root / f"{simulation}_bound_subhalo_masses_v1.json"
    if not npz_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {simulation} bound-mass sidecar. Run scripts/export_bound_subhalo_masses.py first."
        )
    with np.load(npz_path) as data:
        arrays = {name: data[name] for name in data.files}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return arrays, manifest


def load_flamingo_pair_table(project: str | Path) -> PairTable:
    project = Path(project)
    cache = project / "analogues/cache/run_001_mstar2e10"
    with (cache / "flamingo_sample.pkl").open("rb") as stream:
        sample = pickle.load(stream)
    with (cache / "flamingo_pairs.pkl").open("rb") as stream:
        pairs = pickle.load(stream)
    with (project / "analogues/cache/tng_features_v3/metallicity_log_zscore.pkl").open("rb") as stream:
        met = pickle.load(stream)
    bound, bound_manifest = _load_bound_sidecar(project, "flamingo")

    i, j = pairs["i"], pairs["j"]
    met_i, met_j = met["flamingo_met_dex"][i], met["flamingo_met_dex"][j]
    gr_i, gr_j = sample["g_r"][i], sample["g_r"][j]
    sigma_i, sigma_j = sample["vel_disp"][i], sample["vel_disp"][j]
    kappa_i, kappa_j = sample["kappa_rot"][i], sample["kappa_rot"][j]
    frame = pd.DataFrame({
        "pair_index": np.arange(len(i), dtype=np.int64),
        "id_a": sample["row_index"][i], "id_b": sample["row_index"][j],
        "host_i": sample["host_top_index"][i], "host_j": sample["host_top_index"][j],
        "same_host": pairs["have_same_host"].astype(bool),
        "mstar_a": sample["stellar_mass"][i], "mstar_b": sample["stellar_mass"][j],
        "Msub_bound_1": bound["Msub_bound_1"], "Msub_bound_2": bound["Msub_bound_2"],
        "Msub_bound_sum": bound["Msub_bound_sum"],
        "separation": pairs["separation_kpc"], "vr": pairs["vr"], "vt": pairs["vt"],
        "log_mass_ratio": pairs["log_mass_ratio"],
        "log_m200c": pairs["log_tot_virial_mass"],
        "log_bound_subhalo_sum": np.log10(bound["Msub_bound_sum"]),
        "gr_mean": (gr_i + gr_j) / 2, "gr_diff": np.abs(gr_i - gr_j),
        "met_diff": np.abs(met_i - met_j),
        "sigma_mean": (sigma_i + sigma_j) / 2, "sigma_diff": np.abs(sigma_i - sigma_j),
        "kappa_mean": (kappa_i + kappa_j) / 2, "kappa_diff": np.abs(kappa_i - kappa_j),
    })
    return PairTable(
        "flamingo", frame, {"authoritative_pairs": len(frame)},
        {"sample": str(cache / "flamingo_sample.pkl"), "pairs": str(cache / "flamingo_pairs.pkl"),
         "bound_mass_sidecar": bound_manifest},
    )


def load_tng_pair_table(project: str | Path) -> PairTable:
    project = Path(project)
    data_dir = project / "analogues/tng_for_comparison_v3"
    feature_dir = project / "analogues/cache/tng_features_v3"
    manifest = json.loads((data_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    arrays = {name: np.load(data_dir / f"{name}.npy") for name in (
        "keep_idx", "pairs_i", "pairs_j", "grnr", "g_r", "stellar_mass",
        "separation", "vr", "vt", "log_mass_ratio", "log_m200c", "have_same_host",
    )}
    with (feature_dir / "tng_sigma_star.pkl").open("rb") as stream:
        sigma_lookup = pickle.load(stream)
    with (feature_dir / "tng_kappa_corot.pkl").open("rb") as stream:
        kappa_lookup = pickle.load(stream)
    with (feature_dir / "metallicity_log_zscore.pkl").open("rb") as stream:
        met = pickle.load(stream)
    bound, bound_manifest = _load_bound_sidecar(project, "tng")

    i, j = arrays["pairs_i"], arrays["pairs_j"]
    global_i, global_j = arrays["keep_idx"][i], arrays["keep_idx"][j]
    sigma_i = _finite_lookup(sigma_lookup, global_i, "tng_sigma_i")
    sigma_j = _finite_lookup(sigma_lookup, global_j, "tng_sigma_j")
    kappa_i = _finite_lookup(kappa_lookup, global_i, "tng_kappa_i")
    kappa_j = _finite_lookup(kappa_lookup, global_j, "tng_kappa_j")
    gr_i, gr_j = arrays["g_r"][i], arrays["g_r"][j]
    met_i, met_j = met["tng_met_dex"][i], met["tng_met_dex"][j]
    frame = pd.DataFrame({
        "pair_index": np.arange(len(i), dtype=np.int64),
        "id_a": global_i, "id_b": global_j,
        "host_i": arrays["grnr"][i], "host_j": arrays["grnr"][j],
        "same_host": arrays["have_same_host"].astype(bool),
        "mstar_a": arrays["stellar_mass"][i], "mstar_b": arrays["stellar_mass"][j],
        "Msub_bound_1": bound["Msub_bound_1"], "Msub_bound_2": bound["Msub_bound_2"],
        "Msub_bound_sum": bound["Msub_bound_sum"],
        "separation": arrays["separation"], "vr": arrays["vr"], "vt": arrays["vt"],
        "log_mass_ratio": arrays["log_mass_ratio"], "log_m200c": arrays["log_m200c"],
        "log_bound_subhalo_sum": np.log10(bound["Msub_bound_sum"]),
        "gr_mean": (gr_i + gr_j) / 2, "gr_diff": np.abs(gr_i - gr_j),
        "met_diff": np.abs(met_i - met_j),
        "sigma_mean": (sigma_i + sigma_j) / 2, "sigma_diff": np.abs(sigma_i - sigma_j),
        "kappa_mean": (kappa_i + kappa_j) / 2, "kappa_diff": np.abs(kappa_i - kappa_j),
    })
    if len(frame) != manifest["pair_count"]:
        raise ValueError("TNG pair count disagrees with the authoritative selection manifest")
    return PairTable(
        "tng", frame, {"authoritative_pairs": len(frame)},
        {"selection_manifest": manifest, "bound_mass_sidecar": bound_manifest},
    )
