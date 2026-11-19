"""FLAMINGO Local Group analogue pipeline helpers.

This module adapts a pair pipeline to the FLAMINGO
L1_m8 SOAP catalogue. 
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# Global constants that are related to physics and used in simulation
H = 0.681
BOX_MPC = 1000.0
BOX_KPC = BOX_MPC * 1e3
# SOAP metadata:
# Value stored as physical = 1
# h-scale exponent = 0
# Therefore masses are stored in units of 1e10 Msun
# and should NOT be divided by h.
MASS_UNIT_MSUN = 1e10   # verified that this is needed from metadata
MPC_TO_KPC = 1e3

# Path to filter values and SOAP path
DEFAULT_SOAP_PATH = "FLAMINGO/L1_m8/L1_m8/SOAP-HBT/halo_properties_0078.hdf5"
DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("flamingo_selection_config.json"))

# Filteration values. Consistent with TNG300-1
DEFAULT_SELECTION_CONFIG: dict[str, Any] = {
    "sample": {
        "mstar_min_msun": 2e10,
        "mstar_max_msun": 5e11,
        "blue_threshold_gr": 0.68,
        "disc_threshold_kappa": 0.35
    },
    "pairs": {
        "r_min_kpc": 500.0,
        "r_max_kpc": 1000.0
    },
    "filters": {
        "vr_min_kms": -400.0,
        "vr_max_kms": 0.0,
        "vt_min_kms": 0.0,
        "vt_max_kms": 300.0,
        "v_total_max_kms": 500.0,
        "same_host": None,
        "color_pair": "any",
        "disc_pair": "any",
        "third_massive_factor": 1.5,    # Equivalent to TNG300 values
        "density_radius_kpc": 2000.0,
        "intruder_factor": 0.5,
    }
}

# All the relevant fields from FLAMINGO SOAP path for now might add more later
FIELD_PATHS = {
    "stellar_mass": "BoundSubhalo/StellarMass",
    "pos_mpc": "BoundSubhalo/CentreOfMass",
    "vel": "BoundSubhalo/CentreOfMassVelocity",
    "bound_total_mass": "BoundSubhalo/TotalMass",
    "gama_luminosity": "BoundSubhalo/StellarLuminosity",
    "kappa_rot": "BoundSubhalo/KappaCorotStars",
    "vel_disp_matrix": "BoundSubhalo/StellarVelocityDispersionMatrix",
    "metallicity": "BoundSubhalo/StellarMassFractionInMetals",
    "n_stars": "BoundSubhalo/NumberOfStarParticles",
    "m200c": "SO/200_crit/TotalMass",
    "r200c_mpc": "SO/200_crit/SORadius",
    "concentration": "SO/200_crit/Concentration",
    "host_halo_index": "SOAP/HostHaloIndex",
    "is_central": "InputHalos/IsCentral",
    "subhalo_rank_by_bound_mass": "SOAP/SubhaloRankByBoundMass",
}

# Merge user config file and default config file
def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged

# Loads JSON config and merges with defaults
def load_selection_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the JSON selection config and merge it with defaults."""
    with open(path, "r", encoding="utf-8") as config_file:
        user_config = json.load(config_file)
    return _deep_update(DEFAULT_SELECTION_CONFIG, user_config)

# To open Flamingo SOAP catalogue
def open_default_soap():
    """Open the default FLAMINGO L1_m8 z=0 SOAP catalogue."""
    import hdfstream

    root = hdfstream.open("cosma", "/")
    return root[DEFAULT_SOAP_PATH]

# Used to load only part of catalogue
def _read(dataset, selection: slice | np.ndarray | None = None):
    if selection is None:
        return dataset[...]
    return dataset[selection]

# Gives shape
def _shape_of(soap_file, path: str) -> tuple[int, ...]:
    return tuple(soap_file[path].shape)

# To Verify SOAP schema
def validate_required_fields(soap_file) -> list[dict[str, Any]]:
    """Return field metadata for all datasets required by the pipeline."""
    rows: list[dict[str, Any]] = []
    for name, path in FIELD_PATHS.items():
        dataset = soap_file[path]
        attrs = dict(dataset.attrs)
        rows.append(
            {
                "name": name,
                "path": path,
                "shape": tuple(dataset.shape),
                "dtype": str(dataset.dtype),
                "description": attrs.get("Description", ""),
                "masked": bool(attrs.get("Masked", 0)),
                "mask_datasets": attrs.get("Mask Datasets", None),
                "mask_threshold": attrs.get("Mask Threshold", None),
                "value_stored_as_physical": attrs.get("Value stored as physical", None),
                "a_scale_exponent": attrs.get("a-scale exponent", None),
                "h_scale_exponent": attrs.get("h-scale exponent", None),
            }
        )
    return rows

# For reading purposes for validate_required_fields()
def print_schema_report(soap_file) -> None:
    """Print a compact schema report for the corrected FLAMINGO paths."""
    print("Top-level groups:", list(soap_file))
    print("\nRequired fields:")
    for row in validate_required_fields(soap_file):
        masked = "masked" if row["masked"] else "unmasked"
        print(f"  {row['name']:28s} {row['path']:45s} {row['shape']} {row['dtype']} {masked}")
        if row["mask_datasets"] is not None:
            print(f"    mask: {row['mask_datasets']} >= {row['mask_threshold']}")
        if row["description"]:
            print(f"    {row['description']}")

# Load all the fields and apply some unit conversions
def load_flamingo_fields(soap_file, selection: slice | np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Load required FLAMINGO SOAP fields in physical units."""
    n_total = _shape_of(soap_file, FIELD_PATHS["stellar_mass"])[0]
    if selection is None:
        row_index = np.arange(n_total, dtype=np.int64)
    elif isinstance(selection, slice):
        row_index = np.arange(n_total, dtype=np.int64)[selection]
    else:
        row_index = np.asarray(selection, dtype=np.int64)

    stellar_mass = _read(soap_file[FIELD_PATHS["stellar_mass"]], selection) * MASS_UNIT_MSUN
    pos_kpc = _read(soap_file[FIELD_PATHS["pos_mpc"]], selection) * MPC_TO_KPC
    vel = _read(soap_file[FIELD_PATHS["vel"]], selection)
    bound_total_mass = _read(soap_file[FIELD_PATHS["bound_total_mass"]], selection) * MASS_UNIT_MSUN
    n_stars = _read(soap_file[FIELD_PATHS["n_stars"]], selection)
    host_halo_index = _read(soap_file[FIELD_PATHS["host_halo_index"]], selection)
    is_central = _read(soap_file[FIELD_PATHS["is_central"]], selection).astype(bool)
    rank = _read(soap_file[FIELD_PATHS["subhalo_rank_by_bound_mass"]], selection)

    m200c = _read(soap_file[FIELD_PATHS["m200c"]], selection) * MASS_UNIT_MSUN
    r200c_kpc = _read(soap_file[FIELD_PATHS["r200c_mpc"]], selection) * MPC_TO_KPC
    concentration = _read(soap_file[FIELD_PATHS["concentration"]], selection)

    # Concentration is only defined for centrals so we zero out satellites
    concentration = np.where(is_central, concentration, np.nan)

    photometry_available = n_stars >= 100    # Something equivalent to TNG300 not available so this best substitute as it removes noisy colors, morophology and kinematics
    # Lower resolution objects removed with this and mass filter

    lum = _read(soap_file[FIELD_PATHS["gama_luminosity"]], selection)
    lum_gr = lum[:, 1:3]     #  GAMA bands order CONFIRMED: (u, g, r, i, z) — cols 1,2 are g,r
    valid_phot = photometry_available & (lum_gr[:, 0] > 0) & (lum_gr[:, 1] > 0)
    g_r = np.full(len(stellar_mass), np.nan, dtype=np.float32)
    g_r[valid_phot] = -2.5 * np.log10(lum_gr[valid_phot, 0] / lum_gr[valid_phot, 1])

    kappa_rot = _read(soap_file[FIELD_PATHS["kappa_rot"]], selection)
    kappa_rot = np.where(photometry_available, kappa_rot, np.nan)
    vel_disp_matrix = _read(soap_file[FIELD_PATHS["vel_disp_matrix"]], selection)
    assert vel_disp_matrix.shape[1] in (3, 6), \
        f"Unexpected vel_disp_matrix shape {vel_disp_matrix.shape}"
    if vel_disp_matrix.shape[1] == 6:
       # The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.
        vel_disp = np.sqrt(
            np.maximum(0.0, (vel_disp_matrix[:, 0] + vel_disp_matrix[:, 1] + vel_disp_matrix[:, 2]) / 3.0)
        )
    else:
        vel_disp = np.sqrt(
            np.maximum(0.0, (vel_disp_matrix[:, 0] + vel_disp_matrix[:, 1] + vel_disp_matrix[:, 2]) / 3.0)
        )
    vel_disp = np.where(photometry_available, vel_disp, np.nan)
    metallicity = _read(soap_file[FIELD_PATHS["metallicity"]], selection)

    valid_hosts = host_halo_index[host_halo_index >= 0]
    if len(valid_hosts) > 0:
        assert valid_hosts.max() < n_total, \
            "host_halo_index contains out-of-bounds global indices — catalogue integrity issue."
    host_top_index = np.where(host_halo_index >= 0, host_halo_index, row_index)

    return {
        "row_index": row_index,
        "stellar_mass": stellar_mass,
        "pos_kpc": pos_kpc,
        "vel": vel,
        "rank": rank,
        "bound_total_mass": bound_total_mass,
        "m200c": m200c,
        "r200c_kpc": r200c_kpc,
        "g_r": g_r,
        "kappa_rot": kappa_rot,
        "vel_disp": vel_disp,
        "metallicity": metallicity,
        "concentration": concentration,
        "host_halo_index": host_halo_index,
        "host_top_index": host_top_index,
        "is_central": is_central,
        "quality_flag": n_stars >= 100,
        "n_stars": n_stars,
    }

# Constructs a sample using the mass based and the no of star filters which was defined in load_flamingo_fields()
def generate_flamingo_sample(
    fields: dict[str, np.ndarray],
    mstar_min: float = 2e10,
    mstar_max: float = 5e11,
    blue_threshold_gr: float = 0.68,  # Equivalent to TNG300 value
    disc_threshold_kappa: float = 0.35, # Still experimental might change later
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Select central galaxies plus the largest stellar mass satellite per host."""
    stellar_mass = fields["stellar_mass"]
    is_central_full = fields["is_central"]
    host_top = fields["host_top_index"]
    quality = fields["quality_flag"]

    in_window = (stellar_mass >= mstar_min) & (stellar_mass <= mstar_max)
    good = quality & in_window

    # host eligibility is gated on the central passing the mass window.
    # Only hosts whose central passes become eligible. I did this to mirror TNG generate_sample exactly.
    is_central_full = fields["is_central"]
    central_good = is_central_full & good
    central_idx = np.flatnonzero(central_good)
    eligible_hosts = np.unique(host_top[central_idx])  # host_top_index of passing centrals

    # Among satellites in eligible hosts pick the one with the highest stellar mass (not bound-mass rank==1) that also passes the window.
    is_satellite = ~is_central_full   # NOTE: Did not choose rank cause it uses bounded mass which is not equivalent to TNG300 pipeline.
    sat_mask = good & is_satellite & np.isin(host_top, eligible_hosts)
    sat_idx = np.flatnonzero(sat_mask)

    if sat_idx.size == 0:
        best_sat_idx = np.array([], dtype=np.int64)
    else:
        sat_host = host_top[sat_idx]
        sat_mstar = stellar_mass[sat_idx]
        # sort by host asc, stellar mass desc to first occurrence per host = largest
        order = np.lexsort((-sat_mstar, sat_host))
        sat_idx_s = sat_idx[order]
        sat_host_s = sat_host[order]
        new_group = np.r_[True, sat_host_s[1:] != sat_host_s[:-1]]
        best_sat_idx = sat_idx_s[new_group]

    keep_idx = np.unique(np.concatenate([central_idx, best_sat_idx]))
    if verbose:
        print("\n[DIAGNOSTIC] Sample Selection")
        print(f"  Input rows: {len(stellar_mass):,}")
        print(f"  Quality n_stars>100: {quality.sum():,}")
        print(f"  In stellar mass window: {good.sum():,}")
        print(f"  Eligible hosts (central passes): {len(eligible_hosts):,}")
        print(f"  Centrals kept: {len(central_idx):,}")
        print(f"  Largest stellar-mass satellites kept: {len(best_sat_idx):,}")
        print(f"  Final sample: {len(keep_idx):,}")

    gr = fields["g_r"][keep_idx]
    kappa = fields["kappa_rot"][keep_idx]

    is_blue = np.where(np.isfinite(gr), gr < blue_threshold_gr, False)
    is_disc = np.where(np.isfinite(kappa), kappa > disc_threshold_kappa, False)

    return {
        "keep_idx": keep_idx,
        "row_index": fields["row_index"][keep_idx],
        "host_top_index": host_top[keep_idx],
        "is_central": is_central_full[keep_idx],
        "is_largest_satellite": ~is_central_full[keep_idx],
        "valid_color": np.isfinite(fields["g_r"][keep_idx]),
        "valid_morph": np.isfinite(kappa),
        "is_blue": is_blue,
        "is_red": np.isfinite(gr) & ~is_blue,
        "is_disc": is_disc,
        "pos_kpc": fields["pos_kpc"][keep_idx],
        "vel": fields["vel"][keep_idx],
        "stellar_mass": fields["stellar_mass"][keep_idx],
        "m200c": fields["m200c"][keep_idx],
        "r200c_kpc": fields["r200c_kpc"][keep_idx],
        "bound_total_mass": fields["bound_total_mass"][keep_idx],
        "g_r": gr,
        "kappa_rot": kappa,
        "vel_disp": fields["vel_disp"][keep_idx],
        "metallicity": fields["metallicity"][keep_idx],
        "concentration": fields["concentration"][keep_idx],
    }

# pair finding using kdtrees
def find_flamingo_pairs(
    sample: dict[str, np.ndarray],
    r_min_kpc: float = 400.0,
    r_max_kpc: float = 1000.0,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Find pairs in physical kpc using FLAMINGO's z=0 physical box."""
    pos = sample["pos_kpc"] % BOX_KPC
    vel = sample["vel"]

    if len(pos) < 2:
        empty = np.array([], dtype=np.int64)
        return {"i": empty, "j": empty}

    tree = cKDTree(pos, boxsize=BOX_KPC)
    pairs_raw = tree.query_pairs(r_max_kpc, output_type="ndarray").astype(np.int64)
    if len(pairs_raw) == 0:
        empty = np.array([], dtype=np.int64)
        return {"i": empty, "j": empty}

    i = pairs_raw[:, 0]
    j = pairs_raw[:, 1]
    dr = pos[j] - pos[i]
    dr -= BOX_KPC * np.round(dr / BOX_KPC)
    dist = np.linalg.norm(dr, axis=1)
    keep = dist >= r_min_kpc
    i, j, dr, dist = i[keep], j[keep], dr[keep], dist[keep]

    dv = vel[j] - vel[i]
    rhat = dr / dist[:, None]
    vr = np.einsum("ij,ij->i", dv, rhat)
    v2 = np.einsum("ij,ij->i", dv, dv)
    vt = np.sqrt(np.maximum(0.0, v2 - vr**2))

    same_host = sample["host_top_index"][i] == sample["host_top_index"][j]
    # Build host -> M200c lookup from centrals only.
    # For a central, host_top_index points to itself, so this maps
    # each host identifier to its central's M200c.
    central_mask = sample["is_central"]
    assert np.all(
        sample["host_top_index"][central_mask] == sample["row_index"][central_mask]
    ), "Centrals' host_top_index should point to their own row — check HostHaloIndex semantics."
    host_to_m200c = dict(zip(
        sample["host_top_index"][central_mask],
        sample["m200c"][central_mask]
    ))

    # For each pair member, look up the M200c of their host.
    m200c_host_i = np.array([host_to_m200c.get(int(h), np.nan)
                              for h in sample["host_top_index"][i]])
    m200c_host_j = np.array([host_to_m200c.get(int(h), np.nan)
                              for h in sample["host_top_index"][j]])

    # Same-host pairs share one halo, so use that halo's M200c directly.
    # Different-host pairs sum the two hosts' M200c values.
    total_m200c = np.where(same_host, m200c_host_i, m200c_host_i + m200c_host_j)
    positive_m200c = total_m200c > 0

    # NOTE: despite the name, this is M200c (SO/200_crit/TotalMass), NOT M_vir.
    # Consumers map it to the column "log_m200c", which is the accurate name.
    log_tot_virial_mass = np.full(len(i), np.nan, dtype=np.float64)
    log_tot_virial_mass[positive_m200c] = np.log10(total_m200c[positive_m200c])

    valid_color_pair = sample["valid_color"][i] & sample["valid_color"][j]
    valid_morph_pair = sample["valid_morph"][i] & sample["valid_morph"][j]
    m_i = sample["bound_total_mass"][i]
    m_j = sample["bound_total_mass"][j]
    positive_mass = (m_i > 0) & (m_j > 0)

    log_mass_ratio = np.full(len(i), np.nan, dtype=np.float32)
    log_mass_ratio[positive_mass] = np.log10(
        np.minimum(m_i[positive_mass], m_j[positive_mass])
        / np.maximum(m_i[positive_mass], m_j[positive_mass])
    )

    total_subhalo_mass = m_i + m_j
    log_total_bound_mass = np.full(len(i), np.nan, dtype=np.float32)
    log_total_bound_mass[positive_mass] = np.log10(total_subhalo_mass[positive_mass])

    if verbose:
        print("\n[DIAGNOSTIC] Pair Finding Results")
        print(f"  Total pairs found: {len(i):,}")
        print(f"  Same host: {same_host.sum():,}")
        if len(i):
            print(f"  Separation range: {dist.min():.1f} - {dist.max():.1f} kpc")
            print(f"  vr range: {vr.min():.1f} - {vr.max():.1f} km/s")
            print(f"  vt range: {vt.min():.1f} - {vt.max():.1f} km/s")

    return {
        "i": i,
        "j": j,
        "have_same_host": same_host,
        "separation_kpc": dist,
        "vr": vr,
        "vt": vt,
        "log_tot_virial_mass": log_tot_virial_mass,
        "log_total_bound_mass": log_total_bound_mass,
        "log_mass_ratio": log_mass_ratio,
        "is_blue_blue": sample["is_blue"][i] & sample["is_blue"][j] & valid_color_pair,
        "is_red_red":   sample["is_red"][i]  & sample["is_red"][j]  & valid_color_pair,
        "is_blue_red":  ((sample["is_blue"][i] & sample["is_red"][j]) | (sample["is_red"][i] & sample["is_blue"][j])) & valid_color_pair,
        "is_disc_disc": sample["is_disc"][i] & sample["is_disc"][j] & valid_morph_pair,
        "is_sph_sph":  (~sample["is_disc"][i]) & (~sample["is_disc"][j]) & valid_morph_pair,
    }
# Applies isolation criteria: Rejects systems where a third object is too massive. For LG like.
def flamingo_isolation_filter_no_third_factor(
    fields: dict[str, np.ndarray],
    pairs: dict[str, np.ndarray],
    sample: dict[str, np.ndarray],
    third_massive_factor: float = 1.5,
) -> np.ndarray:
    """Isolation filter: reject pairs with a very massive third subhalo in the
    same host halo.
    """
    pair_i = pairs["i"]
    pair_j = pairs["j"]
    x = third_massive_factor

    # Stellar masses and host indices for all subhalos in the full catalogue
    mstar_all = fields["stellar_mass"]          # M_sun, shape (N_total,)
    host_all  = fields["host_top_index"]        # shape (N_total,)
    quality   = fields["quality_flag"]          # n_stars >= 100

    # Only consider quality subhalos with positive stellar mass
    good_all = quality & (mstar_all > 0)

    # Host indices of the sample subhalos that form pairs
    sample_host = sample["host_top_index"]      # shape (N_sample,)
    hosts_in_pairs = np.unique(
        np.concatenate([sample_host[pair_i], sample_host[pair_j]])
    )

    # Restrict the all-subhalo search to only the hosts relevant to our pairs
    cand = good_all & np.isin(host_all, hosts_in_pairs)
    cand_idx  = np.flatnonzero(cand)            # global indices into fields
    cand_host = host_all[cand_idx]
    cand_m    = mstar_all[cand_idx]

    # Sort by host ascending, stellar mass descending — top 3 per host
    order    = np.lexsort((-cand_m, cand_host))
    cand_idx  = cand_idx[order]
    cand_host = cand_host[order]
    cand_m    = cand_m[order]

    starts = np.flatnonzero(np.r_[True, cand_host[1:] != cand_host[:-1]])
    ends   = np.r_[starts[1:], cand_host.size]

    top_idx: dict[int, np.ndarray] = {}
    top_m:   dict[int, np.ndarray] = {}
    for s, e in zip(starts, ends):
        g    = int(cand_host[s])
        take = min(3, e - s)
        top_idx[g] = cand_idx[s : s + take]
        top_m[g]   = cand_m[s : s + take]

    # Stellar masses of the sample members (global indices for exclusion check)
    sample_global_idx = sample["row_index"]     # global index into fields
    sample_mstar      = sample["stellar_mass"]  # already in M_sun

    keep_iso = np.ones(len(pair_i), dtype=bool)

    for k in range(len(pair_i)):
        a = int(pair_i[k])
        b = int(pair_j[k])

        ha = int(sample_host[a])
        hb = int(sample_host[b])

        scale = max(float(sample_mstar[a]), float(sample_mstar[b]))

        a_global = int(sample_global_idx[a])
        b_global = int(sample_global_idx[b])

        def violates(h: int, _a=a_global, _b=b_global, _scale=scale) -> bool:
            if h not in top_idx:
                return False
            for idx_val, m_val in zip(top_idx[h], top_m[h]):
                if int(idx_val) != _a and int(idx_val) != _b:
                    return float(m_val) >= x * _scale
            return False

        if ha == hb:
            if violates(ha):
                keep_iso[k] = False
        else:
            if violates(ha) or violates(hb):
                keep_iso[k] = False

    return keep_iso

# Reject pairs if third subhalo has mass greater than 0.5(M_i+M_j)
def flamingo_isolation_filter_no_intruder(
    fields: dict[str, np.ndarray],
    pairs: dict[str, np.ndarray],
    sample: dict[str, np.ndarray],
    density_radius_kpc: float = 2000.0,
    intruder_factor: float = 0.5,
) -> np.ndarray:
    """Isolation filter: reject pairs if any third subhalo within
    density_radius_kpc of the pair's centre of mass has total mass greater
    than intruder_factor * (M_i + M_j).
    """
    pair_i = pairs["i"]
    pair_j = pairs["j"]

    # Use total bound mass for both the pair members and the intruder search,
    # matching TNG's use of SubhaloMass (total) rather than stellar mass
    m_all   = fields["bound_total_mass"]        # M_sun, shape (N_total,)
    pos_all = fields["pos_kpc"] % BOX_KPC       # physical kpc, shape (N_total, 3)
    quality = fields["quality_flag"]

    valid = quality & (m_all > 0)
    tree          = cKDTree(pos_all[valid], boxsize=BOX_KPC)
    valid_indices = np.flatnonzero(valid)
    m_valid       = m_all[valid]

    # Pair member global indices and total masses
    sample_global_idx = sample["row_index"]
    idx_i = sample_global_idx[pair_i]
    idx_j = sample_global_idx[pair_j]

    m_i = sample["bound_total_mass"][pair_i]
    m_j = sample["bound_total_mass"][pair_j]
    m_combined = intruder_factor * (m_i + m_j)

    # Centre of mass in physical kpc with periodic shortest-path correction
    pos_i = sample["pos_kpc"][pair_i] % BOX_KPC
    pos_j = sample["pos_kpc"][pair_j] % BOX_KPC
    dr    = pos_j - pos_i
    dr   -= BOX_KPC * np.round(dr / BOX_KPC)   # shortest periodic path
    # weight by m_j / m_combined. Note that m_combined already has intruder_factor
    # so we re-derive the weight from raw masses
    weight = (m_j / (m_i + m_j))[:, None]
    com   = (pos_i + dr * weight) % BOX_KPC

    keep_iso = np.ones(len(pair_i), dtype=bool)

    for k in range(len(pair_i)):
        neighbour_local_idx = tree.query_ball_point(com[k], density_radius_kpc)
        if not neighbour_local_idx:
            continue

        neighbour_globals = valid_indices[neighbour_local_idx]
        neighbour_masses  = m_valid[neighbour_local_idx]

        for m_neigh, g_neigh in zip(neighbour_masses, neighbour_globals):
            if g_neigh != idx_i[k] and g_neigh != idx_j[k]:
                if m_neigh > m_combined[k]:
                    keep_iso[k] = False
                    break

    return keep_iso
# Final cuts using velocitites can add more if needed
def filter_flamingo_pairs(
    pairs: dict[str, np.ndarray],
    vr_min_kms: float | None = -400.0,
    vr_max_kms: float | None = 0.0,
    vt_min_kms: float | None = 0.0,
    vt_max_kms: float | None = 300.0,
    v_total_max_kms: float | None = 500.0,
    same_host: bool | None = None,
    color_pair: str = "any",
    disc_pair: str = "any",
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Filter pair dictionary by velocity, host membership, colour, and morphology.

    `color_pair` can be: `any`, `blue_blue`, `red_red`, `blue_red`.
    `disc_pair` can be: `any`, `disc_disc`, `sph_sph`.
    """
    n_pairs = len(pairs.get("i", []))
    mask = np.ones(n_pairs, dtype=bool)

    if n_pairs == 0:
        return pairs, mask

    vr = pairs["vr"]
    vt = pairs["vt"]
    v_total = np.sqrt(vr**2 + vt**2)

    if vr_min_kms is not None:
        mask &= vr >= vr_min_kms
    if vr_max_kms is not None:
        mask &= vr <= vr_max_kms
    if vt_min_kms is not None:
        mask &= vt >= vt_min_kms
    if vt_max_kms is not None:
        mask &= vt <= vt_max_kms
    if v_total_max_kms is not None:
        mask &= v_total <= v_total_max_kms
    if same_host is not None:
        mask &= pairs["have_same_host"] == same_host

    color_masks = {
        "any": np.ones(n_pairs, dtype=bool),
        "blue_blue": pairs["is_blue_blue"],
        "red_red": pairs["is_red_red"],
        "blue_red": pairs["is_blue_red"],
    }
    if color_pair not in color_masks:
        raise ValueError(f"Unknown color_pair={color_pair!r}; expected one of {sorted(color_masks)}")
    mask &= color_masks[color_pair]

    disc_masks = {
        "any": np.ones(n_pairs, dtype=bool),
        "disc_disc": pairs["is_disc_disc"],
        "sph_sph": pairs["is_sph_sph"],
    }
    if disc_pair not in disc_masks:
        raise ValueError(f"Unknown disc_pair={disc_pair!r}; expected one of {sorted(disc_masks)}")
    mask &= disc_masks[disc_pair]

    filtered = {
        key: value[mask] if isinstance(value, np.ndarray) and len(value) == n_pairs else value
        for key, value in pairs.items()
    }
    filtered["v_total"] = v_total[mask]

    if verbose:
        print("\n[DIAGNOSTIC] Pair Filters")
        print(f"  Input pairs: {n_pairs:,}")
        print(f"  Kept pairs: {mask.sum():,} ({mask.mean() * 100:.1f}%)")
        print(f"  color_pair: {color_pair}")
        print(f"  disc_pair: {disc_pair}")

    return filtered, mask

# Puts everything together
def run_flamingo_pipeline_from_config(
    config: dict[str, Any] | None = None,
    soap_file=None,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run load, sample, KDTree pair finding, and configured pair filtering."""
    cfg = _deep_update(DEFAULT_SELECTION_CONFIG, config or {})
    if soap_file is None:
        soap_file = open_default_soap()

    selection = None

    fields = load_flamingo_fields(soap_file, selection=selection)
    sample = generate_flamingo_sample(
        fields,
        mstar_min=cfg["sample"]["mstar_min_msun"],
        mstar_max=cfg["sample"]["mstar_max_msun"],
        blue_threshold_gr=cfg["sample"]["blue_threshold_gr"],
        disc_threshold_kappa=cfg["sample"]["disc_threshold_kappa"],
        verbose=verbose,
    )
    pairs = find_flamingo_pairs(
        sample,
        r_min_kpc=cfg["pairs"]["r_min_kpc"],
        r_max_kpc=cfg["pairs"]["r_max_kpc"],
        verbose=verbose,
    )
    # Isolation filters
    iso_mask_1 = flamingo_isolation_filter_no_third_factor(
        fields, pairs, sample,
        third_massive_factor=cfg["filters"].get("third_massive_factor", 1.5)
    )
    n_before = len(pairs["i"])
    pairs = {
        k: v[iso_mask_1] if isinstance(v, np.ndarray) and len(v) == n_before else v
        for k, v in pairs.items()
    }

    iso_mask_2 = flamingo_isolation_filter_no_intruder(
        fields, pairs, sample,
        density_radius_kpc=cfg["filters"].get("density_radius_kpc", 2000.0),
        intruder_factor=cfg["filters"].get("intruder_factor", 0.5)
    )
    n_before = len(pairs["i"])
    pairs = {
        k: v[iso_mask_2] if isinstance(v, np.ndarray) and len(v) == n_before else v
        for k, v in pairs.items()
    }
    pairs_filtered, _ = filter_flamingo_pairs(
        pairs,
        vr_min_kms=cfg["filters"]["vr_min_kms"],
        vr_max_kms=cfg["filters"]["vr_max_kms"],
        vt_min_kms=cfg["filters"]["vt_min_kms"],
        vt_max_kms=cfg["filters"]["vt_max_kms"],
        v_total_max_kms=cfg["filters"]["v_total_max_kms"],
        same_host=cfg["filters"]["same_host"],
        color_pair=cfg["filters"]["color_pair"],
        disc_pair=cfg["filters"]["disc_pair"],
        verbose=verbose,
    )
    return fields, sample, pairs_filtered

# Debugging function
def run_smoke_test(max_rows: int = 100_000) -> None:
    """Validate schema and run a small-row smoke test."""
    soap_file = open_default_soap()
    print_schema_report(soap_file)
    print(f"\nLoading first {max_rows:,} rows for smoke test...")
    fields = load_flamingo_fields(soap_file, selection=slice(0, max_rows))
    sample = generate_flamingo_sample(fields)
    pairs = find_flamingo_pairs(sample)
    print(f"\nSmoke-test pairs: {len(pairs['i']):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and smoke-test the FLAMINGO LG pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="JSON selection/filter config path.")
    parser.add_argument("--schema-only", action="store_true", help="Only print field schema and exit.")
    parser.add_argument("--use-config", action="store_true", help="Run with the JSON selection/filter config.")
    parser.add_argument("--max-rows", type=int, default=100_000, help="Rows to load for the smoke test.")
    args = parser.parse_args()

    soap_file = open_default_soap()
    print_schema_report(soap_file)
    if args.schema_only:
        return

    if args.use_config:
        cfg = load_selection_config(args.config)
        _, _, pairs = run_flamingo_pipeline_from_config(cfg, soap_file=soap_file)
        print(f"\nConfigured final pairs: {len(pairs['i']):,}")
    else:
        print(f"\nLoading first {args.max_rows:,} rows for smoke test...")
        fields = load_flamingo_fields(soap_file, selection=slice(0, args.max_rows))
        sample = generate_flamingo_sample(fields)
        pairs = find_flamingo_pairs(sample)
        print(f"\nSmoke-test pairs: {len(pairs['i']):,}")


if __name__ == "__main__":
    main()
