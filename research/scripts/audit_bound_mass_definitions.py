#!/usr/bin/env python
"""Audit bound-subhalo mass definitions and component contributions.

The audit uses the fixed held-out FLAMINGO population, the corrected TNG
population and the existing out-of-fold common-support scores. It verifies the
catalogue totals against their mass components and compares conditional
component-to-stellar mappings. No posterior estimator is fitted or updated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DOMAIN_RESULT_ID = "20260925T103101Z"
RANDOM_SEED = 260929
BOOTSTRAP_REPEATS = 500
OVERLAP_HALF_WIDTHS = (0.10, 0.20, 0.50)
PRIMARY_HALF_WIDTH = 0.10
MIN_ROWS_PER_SIMULATION = 30
SOAP_PATH = "FLAMINGO/L1_m8/L1_m8/SOAP-HBT/halo_properties_0078.hdf5"
SOAP_MASS_FACTOR = 1e10

PAIR_OUTCOMES = OrderedDict([
    ("catalogue_total_per_stellar", "log_catalogue_total_per_stellar"),
    ("reconstructed_total_per_stellar", "log_reconstructed_total_per_stellar"),
    ("dark_matter_per_stellar", "log_dark_matter_per_stellar"),
    ("dark_matter_plus_stellar_per_stellar", "log_dm_plus_stellar_per_stellar"),
    ("gas_plus_stellar_per_stellar", "log_gas_plus_stellar_per_stellar"),
    ("bound_baryons_per_stellar", "log_bound_baryons_per_stellar"),
])
MEMBER_OUTCOMES = OrderedDict([
    ("catalogue_total_per_stellar", "log_catalogue_total_per_stellar"),
    ("dark_matter_plus_stellar_per_stellar", "log_dm_plus_stellar_per_stellar"),
    ("bound_baryons_per_stellar", "log_bound_baryons_per_stellar"),
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numbered_group_files(base: Path, snapshot: int) -> list[Path]:
    root = base / f"groups_{snapshot:03d}"
    files = list(root.glob(f"fof_subhalo_tab_{snapshot:03d}.*.hdf5"))
    files.sort(key=lambda path: int(path.name.split(".")[-2]))
    if not files:
        raise FileNotFoundError(f"No group catalogue files found under {root}")
    return files


def read_groupcat_rows(
    files: list[Path],
    group: str,
    dataset: str,
    global_indices: np.ndarray,
) -> np.ndarray:
    """Read selected global group-catalogue rows without loading all objects."""
    indices = np.asarray(global_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("At least one group-catalogue index is required.")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Group-catalogue indices must be sorted and unique.")

    count_attribute = {
        "Subhalo": "Nsubgroups_ThisFile",
        "Group": "Ngroups_ThisFile",
    }[group]
    output = None
    offset = 0
    filled = np.zeros(len(indices), dtype=bool)

    for path in files:
        with h5py.File(path, "r") as handle:
            count = int(handle["Header"].attrs[count_attribute])
            select = (indices >= offset) & (indices < offset + count)
            if select.any():
                local_indices = indices[select] - offset
                values = np.asarray(handle[group][dataset][local_indices])
                if output is None:
                    output = np.empty(
                        (len(indices), *values.shape[1:]), dtype=values.dtype
                    )
                output[select] = values
                filled[select] = True
            offset += count

    if output is None or not filled.all():
        missing = indices[~filled]
        raise IndexError(
            f"Could not read {len(missing)} {group}/{dataset} rows; "
            f"first missing index is {missing[0] if len(missing) else 'none'}."
        )
    return output


def lookup(sorted_ids: np.ndarray, values: np.ndarray, requested: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_ids, np.asarray(requested, dtype=np.int64))
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], np.asarray(requested, dtype=np.int64)
    ):
        raise KeyError("An endpoint identifier is missing from the component lookup.")
    return values[positions]


def load_flamingo_components(
    global_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    import hdfstream

    soap = hdfstream.open("cosma", "/")[SOAP_PATH]
    fields = OrderedDict([
        ("total", "TotalMass"),
        ("dark_matter", "DarkMatterMass"),
        ("gas", "GasMass"),
        ("stellar", "StellarMass"),
        ("black_hole_dynamical", "BlackHolesDynamicalMass"),
        ("black_hole_subgrid", "BlackHolesSubgridMass"),
    ])
    arrays = {}
    field_metadata = {}
    for label, field in fields.items():
        dataset = soap[f"BoundSubhalo/{field}"]
        arrays[label] = np.asarray(dataset[global_ids], dtype=np.float64) * SOAP_MASS_FACTOR
        attributes = {}
        for key, value in dataset.attrs.items():
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            attributes[str(key)] = value
        field_metadata[label] = {
            "path": f"BoundSubhalo/{field}",
            "description": attributes.get("Description"),
            "value_stored_as_physical": attributes.get("Value stored as physical"),
            "h_scale_exponent": attributes.get("h-scale exponent"),
            "mass_conversion_to_msun": SOAP_MASS_FACTOR,
        }
    return arrays, field_metadata


def load_tng_components(
    base: Path,
    global_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict, np.ndarray]:
    files = numbered_group_files(base, 99)
    with h5py.File(files[0], "r") as handle:
        h = float(handle["Header"].attrs["HubbleParam"])
        redshift = float(handle["Header"].attrs["Redshift"])
        total_subhaloes = int(handle["Header"].attrs["Nsubgroups_Total"])
    conversion = 1e10 / h
    mass_type = read_groupcat_rows(
        files, "Subhalo", "SubhaloMassType", global_ids
    ).astype(np.float64) * conversion
    total = read_groupcat_rows(
        files, "Subhalo", "SubhaloMass", global_ids
    ).astype(np.float64) * conversion
    group_number = read_groupcat_rows(
        files, "Subhalo", "SubhaloGrNr", global_ids
    ).astype(np.int64)
    unique_groups = np.unique(group_number)
    first_subhalo = read_groupcat_rows(
        files, "Group", "GroupFirstSub", unique_groups
    ).astype(np.int64)
    central_for_group = lookup(unique_groups, first_subhalo, group_number)
    is_central = global_ids == central_for_group
    arrays = {
        "total": total,
        "gas": mass_type[:, 0],
        "dark_matter": mass_type[:, 1],
        "type2": mass_type[:, 2],
        "tracer": mass_type[:, 3],
        "stellar": mass_type[:, 4],
        "black_hole_dynamical": mass_type[:, 5],
        "black_hole_subgrid": np.full(len(global_ids), np.nan),
    }
    metadata = {
        "path": str(base.resolve()),
        "snapshot": 99,
        "redshift": redshift,
        "hubble_parameter": h,
        "stored_mass_units": "1e10 Msun/h",
        "mass_conversion_to_msun": conversion,
        "source_fields": ["SubhaloMass", "SubhaloMassType", "SubhaloGrNr", "GroupFirstSub"],
        "subhalo_mass_type_order": ["gas", "dark_matter", "type2", "tracers", "stars_and_wind", "black_holes"],
        "group_catalogue_files": len(files),
        "total_subhaloes": total_subhaloes,
    }
    return arrays, metadata, is_central


def equal_reference_location_scale(
    source: pd.DataFrame,
    transfer: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source_values = source[columns].to_numpy(np.float64)
    transfer_values = transfer[columns].to_numpy(np.float64)
    centre = 0.5 * (source_values.mean(axis=0) + transfer_values.mean(axis=0))
    variance = 0.5 * (
        np.mean((source_values - centre) ** 2, axis=0)
        + np.mean((transfer_values - centre) ** 2, axis=0)
    )
    keep = np.isfinite(variance) & (variance > 1e-12)
    return centre[keep], np.sqrt(variance[keep]), [
        column for column, retained in zip(columns, keep) if retained
    ]


def make_design(
    frame: pd.DataFrame,
    columns: list[str],
    centre: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (frame[columns].to_numpy(np.float64) - centre) / scale
    return np.column_stack([np.ones(len(frame)), standardized])


def weighted_least_squares(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if weights is None:
        return np.linalg.pinv(design.T @ design) @ (design.T @ targets)
    weighted_design = design * weights[:, None]
    return np.linalg.pinv(design.T @ weighted_design) @ (
        design.T @ (targets * weights[:, None])
    )


def bootstrap_row_weights(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_components = int(inverse.max()) + 1
    sampled = rng.integers(0, n_components, size=n_components)
    multiplicity = np.bincount(sampled, minlength=n_components).astype(np.float64)
    return multiplicity[inverse]


def prepare_mapping(
    source: pd.DataFrame,
    transfer: pd.DataFrame,
    predictors: list[str],
    outcome_columns: list[str],
) -> dict:
    centre, scale, columns = equal_reference_location_scale(
        source, transfer, predictors
    )
    source_design = make_design(source, columns, centre, scale)
    transfer_design = make_design(transfer, columns, centre, scale)
    source_targets = source[outcome_columns].to_numpy(np.float64)
    transfer_targets = transfer[outcome_columns].to_numpy(np.float64)
    return {
        "source": source,
        "transfer": transfer,
        "columns": columns,
        "centre": centre,
        "scale": scale,
        "source_design": source_design,
        "transfer_design": transfer_design,
        "source_targets": source_targets,
        "transfer_targets": transfer_targets,
        "source_beta": weighted_least_squares(source_design, source_targets),
        "transfer_beta": weighted_least_squares(transfer_design, transfer_targets),
    }


def bootstrap_mapping(
    prepared: dict,
    outcome_names: list[str],
    source_group_column: str,
    transfer_group_column: str,
    analysis: str,
    subset: str,
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    source_groups = prepared["source"][source_group_column].to_numpy()
    transfer_groups = prepared["transfer"][transfer_group_column].to_numpy()
    for repeat in range(BOOTSTRAP_REPEATS):
        source_weights = bootstrap_row_weights(source_groups, rng)
        transfer_weights = bootstrap_row_weights(transfer_groups, rng)
        source_beta = weighted_least_squares(
            prepared["source_design"], prepared["source_targets"], source_weights
        )
        transfer_beta = weighted_least_squares(
            prepared["transfer_design"], prepared["transfer_targets"], transfer_weights
        )
        difference = transfer_beta[0] - source_beta[0]
        for outcome_index, outcome in enumerate(outcome_names):
            rows.append({
                "analysis": analysis,
                "subset": subset,
                "outcome": outcome,
                "repeat": repeat,
                "value": difference[outcome_index],
            })
    return rows


def summarise_draws(draws: pd.DataFrame, point_estimates: pd.DataFrame) -> pd.DataFrame:
    summary = (
        draws.groupby(["analysis", "subset", "outcome"], sort=False)["value"]
        .agg(
            bootstrap_median="median",
            lower_95=lambda values: np.quantile(values, 0.025),
            upper_95=lambda values: np.quantile(values, 0.975),
            bootstrap_mean="mean",
            bootstrap_std="std",
            fraction_above_zero=lambda values: np.mean(values > 0),
            repeats="size",
        )
        .reset_index()
    )
    return summary.merge(
        point_estimates,
        on=["analysis", "subset", "outcome"],
        how="left",
        validate="one_to_one",
    )


def attach_pair_components(
    frame: pd.DataFrame,
    ids: np.ndarray,
    components: dict[str, np.ndarray],
    central: np.ndarray,
) -> pd.DataFrame:
    frame = frame.copy()
    for endpoint, id_column in (("a", "id_a"), ("b", "id_b")):
        requested = frame[id_column].to_numpy(np.int64)
        for name, values in components.items():
            frame[f"{name}_{endpoint}"] = lookup(ids, values, requested)
        frame[f"is_central_{endpoint}"] = lookup(ids, central, requested).astype(bool)

    for name in ("total", "dark_matter", "gas", "stellar", "black_hole_dynamical"):
        frame[f"{name}_sum"] = frame[f"{name}_a"] + frame[f"{name}_b"]
    frame["reconstructed_total_sum"] = (
        frame["dark_matter_sum"]
        + frame["gas_sum"]
        + frame["stellar_sum"]
        + frame["black_hole_dynamical_sum"]
    )
    frame["bound_baryon_sum"] = (
        frame["gas_sum"] + frame["stellar_sum"] + frame["black_hole_dynamical_sum"]
    )
    frame["log_stellar_sum"] = np.log10(frame["stellar_sum"])
    frame["log_catalogue_total_per_stellar"] = np.log10(
        frame["total_sum"] / frame["stellar_sum"]
    )
    frame["log_reconstructed_total_per_stellar"] = np.log10(
        frame["reconstructed_total_sum"] / frame["stellar_sum"]
    )
    frame["log_dark_matter_per_stellar"] = np.log10(
        frame["dark_matter_sum"] / frame["stellar_sum"]
    )
    frame["log_dm_plus_stellar_per_stellar"] = np.log10(
        (frame["dark_matter_sum"] + frame["stellar_sum"]) / frame["stellar_sum"]
    )
    frame["log_gas_plus_stellar_per_stellar"] = np.log10(
        (frame["gas_sum"] + frame["stellar_sum"]) / frame["stellar_sum"]
    )
    frame["log_bound_baryons_per_stellar"] = np.log10(
        frame["bound_baryon_sum"] / frame["stellar_sum"]
    )
    return frame


def make_member_frame(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for endpoint in ("a", "b"):
        stellar = frame[f"stellar_{endpoint}"].to_numpy(np.float64)
        total = frame[f"total_{endpoint}"].to_numpy(np.float64)
        dark_matter = frame[f"dark_matter_{endpoint}"].to_numpy(np.float64)
        gas = frame[f"gas_{endpoint}"].to_numpy(np.float64)
        black_hole = frame[f"black_hole_dynamical_{endpoint}"].to_numpy(np.float64)
        part = frame[[
            "pair_index", "simulation", "same_host", "separation", "vr", "vt",
            "domain_score_stellar_individual", "connected_component_stellar_individual",
        ]].copy()
        part["member_id"] = frame[f"id_{endpoint}"]
        part["endpoint"] = endpoint
        part["is_central"] = frame[f"is_central_{endpoint}"].to_numpy(bool)
        if endpoint == "a":
            part["is_stellar_primary"] = frame["stellar_primary_is_a"].to_numpy(bool)
        else:
            part["is_stellar_primary"] = ~frame["stellar_primary_is_a"].to_numpy(bool)
        part["log_mstar_member"] = np.log10(stellar)
        part["log_catalogue_total_per_stellar"] = np.log10(total / stellar)
        part["log_dm_plus_stellar_per_stellar"] = np.log10(
            (dark_matter + stellar) / stellar
        )
        part["log_bound_baryons_per_stellar"] = np.log10(
            (gas + stellar + black_hole) / stellar
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--tng-base",
        type=Path,
        default=Path(os.environ.get("TNG_BASE", r"<TNG_BASE>")),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    from lg_npe.data import load_flamingo_pair_table, load_tng_pair_table
    from lg_npe.features import add_stellar_representations

    domain_matches = list(
        (project / "out").glob(f"*/domain_common_support/{DOMAIN_RESULT_ID}")
    )
    if len(domain_matches) != 1:
        raise FileNotFoundError(
            f"Expected one domain result {DOMAIN_RESULT_ID}; found {len(domain_matches)}."
        )
    domain_result = domain_matches[0]
    domain_scores = pd.read_csv(domain_result / "domain_scores_oof.csv.gz")
    posterior_join = pd.read_csv(
        domain_result / "posterior_domain_join.csv.gz",
        usecols=["population", "pair_index"],
    )
    source_evaluation_pairs = np.sort(
        posterior_join.loc[
            posterior_join["population"] == "source_heldout", "pair_index"
        ].unique()
    )
    if len(source_evaluation_pairs) != 21_409:
        raise RuntimeError("The fixed held-out source population is not 21,409 pairs.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else project / "out" / "domain_shift" / "bound_mass_definition_audit"
    )
    output = output_root / timestamp
    output.mkdir(parents=True, exist_ok=False)

    source_table = load_flamingo_pair_table(project)
    transfer_table = load_tng_pair_table(project)
    source = add_stellar_representations(source_table.frame)
    source = source.loc[source["pair_index"].isin(source_evaluation_pairs)].copy()
    transfer = add_stellar_representations(transfer_table.frame)
    source["simulation"] = "flamingo"
    transfer["simulation"] = "tng"

    for feature_space in ("stellar_sum", "stellar_individual"):
        columns = ["pair_index", "domain_score", "connected_component"]
        for simulation, frame in (("flamingo", source), ("tng", transfer)):
            scores = domain_scores.loc[
                (domain_scores["simulation"] == simulation)
                & (domain_scores["feature_space"] == feature_space),
                columns,
            ].rename(columns={
                "domain_score": f"domain_score_{feature_space}",
                "connected_component": f"connected_component_{feature_space}",
            })
            merged = frame.merge(scores, on="pair_index", how="left", validate="one_to_one")
            if merged[f"domain_score_{feature_space}"].isna().any():
                raise RuntimeError(f"Domain score join failed for {simulation}/{feature_space}.")
            if simulation == "flamingo":
                source = merged
            else:
                transfer = merged

    source_ids = np.unique(np.concatenate([
        source["id_a"].to_numpy(np.int64), source["id_b"].to_numpy(np.int64)
    ]))
    transfer_ids = np.unique(np.concatenate([
        transfer["id_a"].to_numpy(np.int64), transfer["id_b"].to_numpy(np.int64)
    ]))

    print(f"Reading {len(source_ids):,} FLAMINGO member rows from SOAP...")
    source_components, soap_metadata = load_flamingo_components(source_ids)
    sample_path = project / "analogues/cache/run_001_mstar2e10/flamingo_sample.pkl"
    with sample_path.open("rb") as stream:
        source_sample = pickle.load(stream)
    sample_ids = np.asarray(source_sample["row_index"], dtype=np.int64)
    if np.any(np.diff(sample_ids) <= 0):
        raise ValueError("FLAMINGO sample row identifiers are not sorted and unique.")
    source_central = lookup(
        sample_ids,
        np.asarray(source_sample["is_central"], dtype=bool),
        source_ids,
    )

    print(f"Reading {len(transfer_ids):,} TNG member rows from local group catalogues...")
    transfer_components, tng_metadata, transfer_central = load_tng_components(
        args.tng_base.resolve(), transfer_ids
    )

    source = attach_pair_components(
        source, source_ids, source_components, source_central
    )
    transfer = attach_pair_components(
        transfer, transfer_ids, transfer_components, transfer_central
    )

    reconstruction_rows = []
    for simulation, frame, component_arrays in (
        ("flamingo", source, source_components),
        ("tng", transfer, transfer_components),
    ):
        reconstructed_members = (
            component_arrays["dark_matter"]
            + component_arrays["gas"]
            + component_arrays["stellar"]
            + component_arrays["black_hole_dynamical"]
        )
        for level, total, reconstructed in (
            ("unique_members", component_arrays["total"], reconstructed_members),
            ("pair_sums", frame["total_sum"].to_numpy(), frame["reconstructed_total_sum"].to_numpy()),
        ):
            relative = (reconstructed - total) / total
            log_residual = np.log10(reconstructed) - np.log10(total)
            reconstruction_rows.append({
                "simulation": simulation,
                "level": level,
                "n": len(total),
                "median_relative_residual": np.median(relative),
                "p05_relative_residual": np.quantile(relative, 0.05),
                "p95_relative_residual": np.quantile(relative, 0.95),
                "maximum_absolute_relative_residual": np.max(np.abs(relative)),
                "median_log10_residual_dex": np.median(log_residual),
                "maximum_absolute_log10_residual_dex": np.max(np.abs(log_residual)),
            })
    reconstruction_audit = pd.DataFrame(reconstruction_rows)
    reconstruction_audit.to_csv(output / "mass_reconstruction_audit.csv", index=False)

    source_sidecar_error = np.max(np.abs(
        source["total_sum"].to_numpy() / source["Msub_bound_sum"].to_numpy() - 1
    ))
    transfer_sidecar_error = np.max(np.abs(
        transfer["total_sum"].to_numpy() / transfer["Msub_bound_sum"].to_numpy() - 1
    ))
    source_stellar_error = np.max(np.abs(
        source["stellar_sum"].to_numpy()
        / (source["mstar_a"] + source["mstar_b"]).to_numpy() - 1
    ))
    transfer_stellar_error = np.max(np.abs(
        transfer["stellar_sum"].to_numpy()
        / (transfer["mstar_a"] + transfer["mstar_b"]).to_numpy() - 1
    ))
    if max(source_sidecar_error, transfer_sidecar_error, source_stellar_error, transfer_stellar_error) > 1e-6:
        raise RuntimeError("A raw catalogue mass does not reproduce its cached pair value.")

    subsets = OrderedDict([
        ("pooled", lambda frame: np.ones(len(frame), dtype=bool)),
        ("same_host", lambda frame: frame["same_host"].to_numpy(bool)),
        ("different_host", lambda frame: ~frame["same_host"].to_numpy(bool)),
    ])
    predictors_base = ["separation", "vr", "vt", "log_stellar_sum"]
    pair_summary_rows = []
    pair_point_rows = []
    pair_bootstrap_rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for subset, selector in subsets.items():
        for half_width in OVERLAP_HALF_WIDTHS:
            selected = {}
            for simulation, frame in (("flamingo", source), ("tng", transfer)):
                overlap = (
                    np.abs(frame["domain_score_stellar_sum"].to_numpy() - 0.5)
                    <= half_width + 1e-12
                )
                selected[simulation] = frame.loc[selector(frame) & overlap].copy()
            if min(len(selected["flamingo"]), len(selected["tng"])) < MIN_ROWS_PER_SIMULATION:
                continue
            predictors = predictors_base + (["same_host"] if subset == "pooled" else [])
            outcome_names = list(PAIR_OUTCOMES)
            outcome_columns = list(PAIR_OUTCOMES.values())
            prepared = prepare_mapping(
                selected["flamingo"], selected["tng"], predictors, outcome_columns
            )
            difference = prepared["transfer_beta"][0] - prepared["source_beta"][0]
            for outcome_index, outcome in enumerate(outcome_names):
                row = {
                    "analysis": "pair_component_mapping",
                    "subset": subset,
                    "half_width": half_width,
                    "score_interval": (
                        "all" if half_width >= 0.5
                        else f"{0.5-half_width:.2f}-{0.5+half_width:.2f}"
                    ),
                    "outcome": outcome,
                    "n_flamingo": len(prepared["source"]),
                    "n_tng": len(prepared["transfer"]),
                    "components_flamingo": prepared["source"]["connected_component_stellar_sum"].nunique(),
                    "components_tng": prepared["transfer"]["connected_component_stellar_sum"].nunique(),
                    "adjusted_gap_tng_minus_flamingo": difference[outcome_index],
                }
                pair_summary_rows.append(row)
                if half_width == PRIMARY_HALF_WIDTH:
                    pair_point_rows.append({
                        "analysis": row["analysis"],
                        "subset": subset,
                        "outcome": outcome,
                        "estimate": difference[outcome_index],
                    })
            if half_width == PRIMARY_HALF_WIDTH:
                pair_bootstrap_rows.extend(bootstrap_mapping(
                    prepared,
                    outcome_names,
                    "connected_component_stellar_sum",
                    "connected_component_stellar_sum",
                    "pair_component_mapping",
                    subset,
                    rng,
                ))

    pair_mapping_summary = pd.DataFrame(pair_summary_rows)
    pair_mapping_summary.to_csv(output / "pair_component_mapping_summary.csv", index=False)
    pair_bootstrap_draws = pd.DataFrame(pair_bootstrap_rows)
    pair_bootstrap_draws.to_csv(output / "pair_component_bootstrap_draws.csv.gz", index=False)
    pair_bootstrap_summary = summarise_draws(
        pair_bootstrap_draws, pd.DataFrame(pair_point_rows)
    )
    pair_bootstrap_summary.to_csv(output / "pair_component_bootstrap_summary.csv", index=False)

    fraction_rows = []
    for subset, selector in subsets.items():
        for simulation, frame in (("flamingo", source), ("tng", transfer)):
            overlap = (
                np.abs(frame["domain_score_stellar_sum"].to_numpy() - 0.5)
                <= PRIMARY_HALF_WIDTH + 1e-12
            )
            current = frame.loc[selector(frame) & overlap]
            for component in ("dark_matter", "gas", "stellar", "black_hole_dynamical"):
                fraction = current[f"{component}_sum"] / current["total_sum"]
                fraction_rows.append({
                    "subset": subset,
                    "simulation": simulation,
                    "component": component,
                    "n": len(current),
                    "median_fraction": fraction.median(),
                    "mean_fraction": fraction.mean(),
                    "p05_fraction": fraction.quantile(0.05),
                    "p95_fraction": fraction.quantile(0.95),
                })
    component_fraction_summary = pd.DataFrame(fraction_rows)
    component_fraction_summary.to_csv(output / "component_fraction_summary.csv", index=False)

    source_members = make_member_frame(source)
    transfer_members = make_member_frame(transfer)
    member_summary_rows = []
    member_point_rows = []
    member_bootstrap_rows = []
    member_predictors = [
        "separation", "vr", "vt", "log_mstar_member",
        "same_host", "is_stellar_primary",
    ]
    for centrality, central_value in (("central", True), ("satellite", False)):
        selected = {}
        for simulation, frame in (
            ("flamingo", source_members), ("tng", transfer_members)
        ):
            overlap = (
                np.abs(frame["domain_score_stellar_individual"].to_numpy() - 0.5)
                <= PRIMARY_HALF_WIDTH + 1e-12
            )
            selected[simulation] = frame.loc[
                overlap & frame["is_central"].eq(central_value)
            ].copy()
        if min(len(selected["flamingo"]), len(selected["tng"])) < MIN_ROWS_PER_SIMULATION:
            continue
        outcome_names = list(MEMBER_OUTCOMES)
        outcome_columns = list(MEMBER_OUTCOMES.values())
        prepared = prepare_mapping(
            selected["flamingo"], selected["tng"], member_predictors, outcome_columns
        )
        difference = prepared["transfer_beta"][0] - prepared["source_beta"][0]
        for outcome_index, outcome in enumerate(outcome_names):
            row = {
                "analysis": "member_centrality_mapping",
                "subset": centrality,
                "outcome": outcome,
                "n_rows_flamingo": len(prepared["source"]),
                "n_rows_tng": len(prepared["transfer"]),
                "unique_members_flamingo": prepared["source"]["member_id"].nunique(),
                "unique_members_tng": prepared["transfer"]["member_id"].nunique(),
                "components_flamingo": prepared["source"]["connected_component_stellar_individual"].nunique(),
                "components_tng": prepared["transfer"]["connected_component_stellar_individual"].nunique(),
                "adjusted_gap_tng_minus_flamingo": difference[outcome_index],
            }
            member_summary_rows.append(row)
            member_point_rows.append({
                "analysis": row["analysis"],
                "subset": centrality,
                "outcome": outcome,
                "estimate": difference[outcome_index],
            })
        member_bootstrap_rows.extend(bootstrap_mapping(
            prepared,
            outcome_names,
            "connected_component_stellar_individual",
            "connected_component_stellar_individual",
            "member_centrality_mapping",
            centrality,
            rng,
        ))

    member_mapping_summary = pd.DataFrame(member_summary_rows)
    member_mapping_summary.to_csv(output / "member_centrality_mapping_summary.csv", index=False)
    member_bootstrap_draws = pd.DataFrame(member_bootstrap_rows)
    member_bootstrap_draws.to_csv(
        output / "member_centrality_bootstrap_draws.csv.gz", index=False
    )
    member_bootstrap_summary = summarise_draws(
        member_bootstrap_draws, pd.DataFrame(member_point_rows)
    )
    member_bootstrap_summary.to_csv(
        output / "member_centrality_bootstrap_summary.csv", index=False
    )

    mass_mapping_matches = sorted(
        (project / "out/domain_shift/mass_mapping_decomposition").glob("*/decomposition_linear_summary.csv")
    )
    prior_total_gap = None
    prior_total_gap_difference = None
    if mass_mapping_matches:
        previous = pd.read_csv(mass_mapping_matches[-1])
        previous_row = previous.loc[
            (previous["analysis"] == "pair_sum_decomposition")
            & (previous["outcome"] == "stellar_to_bound")
            & (previous["subset"] == "pooled")
            & np.isclose(previous["half_width"], PRIMARY_HALF_WIDTH)
        ]
        if len(previous_row) == 1:
            prior_total_gap = float(
                previous_row.iloc[0]["adjusted_gap_tng_minus_flamingo"]
            )
            current_total_gap = float(pair_mapping_summary.loc[
                (pair_mapping_summary["subset"] == "pooled")
                & (pair_mapping_summary["outcome"] == "catalogue_total_per_stellar")
                & np.isclose(pair_mapping_summary["half_width"], PRIMARY_HALF_WIDTH),
                "adjusted_gap_tng_minus_flamingo",
            ].iloc[0])
            prior_total_gap_difference = current_total_gap - prior_total_gap
            # Raw SOAP values and the cached sidecar are both float32-derived.
            # Their fitted gaps can differ at the final few decimal places even
            # when the underlying pair masses agree to the stated tolerance.
            if abs(prior_total_gap_difference) > 1e-7:
                raise RuntimeError("The audit does not reproduce the earlier stellar-to-bound gap.")

    try:
        git_revision = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_revision = None

    manifest = {
        "experiment_id": f"bound_mass_definition_audit_{timestamp}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(project),
        "git_revision": git_revision,
        "output": str(output),
        "purpose": (
            "Audit raw bound-mass definitions, component reconstruction and conditional "
            "component-to-stellar mappings; no NPE training"
        ),
        "domain_result_id": DOMAIN_RESULT_ID,
        "domain_manifest_sha256": sha256(domain_result / "domain_overlap_manifest.json"),
        "settings": {
            "random_seed": RANDOM_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "overlap_half_widths": list(OVERLAP_HALF_WIDTHS),
            "primary_half_width": PRIMARY_HALF_WIDTH,
            "bootstrap_unit": "connected host component within simulation",
            "linear_reference": "equal simulation weight",
        },
        "source_population": "fixed 21,409-pair held-out FLAMINGO population",
        "transfer_population": "all 1,530 corrected TNG pairs",
        "flamingo": {
            "catalogue": "SOAP over HBT-HERONS",
            "soap_path": SOAP_PATH,
            "field_metadata": soap_metadata,
            "sidecar_manifest_sha256": sha256(
                project / "data/derived/bound_subhalo_masses/flamingo_bound_subhalo_masses_v1.json"
            ),
            "unique_members_read": len(source_ids),
        },
        "tng": {
            "catalogue": "SUBFIND group catalogue",
            **tng_metadata,
            "sidecar_manifest_sha256": sha256(
                project / "data/derived/bound_subhalo_masses/tng_bound_subhalo_masses_v1.json"
            ),
            "unique_members_read": len(transfer_ids),
        },
        "raw_to_cached_maximum_relative_errors": {
            "flamingo_total_pair_sum": float(source_sidecar_error),
            "tng_total_pair_sum": float(transfer_sidecar_error),
            "flamingo_stellar_pair_sum": float(source_stellar_error),
            "tng_stellar_pair_sum": float(transfer_stellar_error),
        },
        "prior_stellar_to_bound_gap": prior_total_gap,
        "prior_gap_reproduction_difference": prior_total_gap_difference,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
        "reporting_notes": [
            "Both catalogue totals are reconstructed from gravitationally bound mass components.",
            "The finders remain different: HBT-HERONS for FLAMINGO and SUBFIND for TNG.",
            "Component gaps identify which mass component carries the discrepancy but do not remove finder dependence.",
            "Central and satellite comparisons use the fixed stellar-individual score interval 0.4--0.6.",
            "No posterior estimator was fitted, updated or selected.",
        ],
        "outputs": sorted(
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
        ),
    }
    (output / "bound_mass_definition_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nPrimary pair-component gaps:")
    print(pair_bootstrap_summary.round(5).to_string(index=False))
    print("\nCentral/satellite member gaps:")
    print(member_bootstrap_summary.round(5).to_string(index=False))
    print(f"\nCompleted bound-mass definition audit: {output}")


if __name__ == "__main__":
    main()
