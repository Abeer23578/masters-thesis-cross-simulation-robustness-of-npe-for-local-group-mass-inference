"""Diagnostic report helpers for the FLAMINGO LG pipeline.

Reproduces the TNG300-style `tabulate` diagnostic tables (simulation
header constants, sample-selection cutflow, pair-finding results, and the
pipeline sample-reduction cutflow) for FLAMINGO, using the `fields` /
`sample` objects already loaded in consistency_check.ipynb from the
cached pickles. No SOAP file access is required — pair finding and the
isolation filters are cheap and are simply re-run on the cached `sample`.

Usage (from the notebook, after cell 1 has loaded `fields` and `sample`):

    from flamingo_diagnostics import print_full_report
    pair_diag = print_full_report(fields, sample)

Place this file in the same directory you added to `sys.path` in cell 0
of the notebook (alongside flamingo_lg_pipeline.py) so the import resolves.

CAVEAT — 'Tidally Dominant' and 'Force Ratio < 0.5':
flamingo_lg_pipeline.py does not compute a tidal-force-ratio diagnostic
anywhere (find_flamingo_pairs only tracks host membership, colour, and
morphology), so these two rows from the TNG300 printout cannot be
reproduced from the current pipeline output. Rather than guess at a
definition and print numbers that only look right, `pair_finding_tables`
omits them and `print_full_report` prints an explicit note instead. If
you share the TNG300 force-ratio definition (e.g. F_pair vs F_third at
some radius), I can wire an equivalent flamingo_force_ratio_filter()
into flamingo_lg_pipeline.py and this report will pick it up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tabulate import tabulate

from flamingo_lg_pipeline import (
    H,
    BOX_KPC,
    DEFAULT_SELECTION_CONFIG,
    find_flamingo_pairs,
    flamingo_isolation_filter_no_third_factor,
    flamingo_isolation_filter_no_intruder,
    filter_flamingo_pairs,
)


# ---------------------------------------------------------------------------
# Simulation header constants
# ---------------------------------------------------------------------------
def header_table(h: float = H, box_kpc: float = BOX_KPC) -> str:
    rows = [
        ["Hubble (h)", f"{h:g}", "dimensionless"],
        ["Box Side", f"{box_kpc:.2f}", "kpc"],
    ]
    return tabulate(rows, headers=["Parameter", "Value", "Units"], tablefmt="fancy_grid")


# ---------------------------------------------------------------------------
# Sample selection cutflow
# ---------------------------------------------------------------------------
def sample_selection_report(fields: dict[str, np.ndarray], cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """Recompute sample-selection cutflow counts directly from `fields`.

    Mirrors generate_flamingo_sample's selection logic exactly (same
    quality flag, same mass window, same eligible-host/eligible-satellite
    definitions) so the counts match what the pipeline actually produced,
    but also surfaces intermediate counts that generate_flamingo_sample
    doesn't return: the raw eligible-satellite count before max-per-host
    dedup, and the per-group breakdown.
    """
    cfg = cfg or DEFAULT_SELECTION_CONFIG
    mstar_min = cfg["sample"]["mstar_min_msun"]
    mstar_max = cfg["sample"]["mstar_max_msun"]

    stellar_mass = fields["stellar_mass"]
    is_central = fields["is_central"]
    host_top = fields["host_top_index"]
    quality = fields["quality_flag"]

    n_total = len(stellar_mass)
    n_quality = int(quality.sum())

    in_window = (stellar_mass >= mstar_min) & (stellar_mass <= mstar_max)
    good = quality & in_window
    n_window = int(good.sum())

    central_good = is_central & good
    central_idx = np.flatnonzero(central_good)
    eligible_hosts = np.unique(host_top[central_idx])
    n_centrals = len(central_idx)

    is_satellite = ~is_central
    sat_mask = good & is_satellite & np.isin(host_top, eligible_hosts)
    sat_idx = np.flatnonzero(sat_mask)
    n_sat_eligible = len(sat_idx)

    if sat_idx.size == 0:
        best_sat_idx = np.array([], dtype=np.int64)
    else:
        sat_host = host_top[sat_idx]
        sat_mstar = stellar_mass[sat_idx]
        order = np.lexsort((-sat_mstar, sat_host))
        sat_idx_s = sat_idx[order]
        sat_host_s = sat_host[order]
        new_group = np.r_[True, sat_host_s[1:] != sat_host_s[:-1]]
        best_sat_idx = sat_idx_s[new_group]

    keep_idx = np.unique(np.concatenate([central_idx, best_sat_idx]))
    n_final = len(keep_idx)

    n_groups_total = len(eligible_hosts)
    n_groups_with_sat = len(np.unique(host_top[best_sat_idx])) if best_sat_idx.size else 0
    n_groups_central_only = n_groups_total - n_groups_with_sat

    return {
        "n_total": n_total,
        "n_quality": n_quality,
        "n_window": n_window,
        "n_centrals": n_centrals,
        "n_sat_eligible": n_sat_eligible,
        "n_final": n_final,
        "n_groups_total": n_groups_total,
        "n_groups_with_sat": n_groups_with_sat,
        "n_groups_central_only": n_groups_central_only,
    }


def sample_selection_tables(fields: dict[str, np.ndarray], cfg: dict[str, Any] | None = None) -> tuple[str, str]:
    r = sample_selection_report(fields, cfg)

    rows = [
        ["Total Subhalos in Catalog", r["n_total"], "100.0%"],
        ["Passed Quality Flags (SubhaloFlag==1)", r["n_quality"],
         f"{100 * r['n_quality'] / r['n_total']:.1f}%"],
        ["Inside Stellar Mass Window", r["n_window"],
         f"{100 * r['n_window'] / r['n_quality']:.1f}% of good"],
        ["Eligible Centrals (Selected Groups)", r["n_centrals"],
         f"{100 * r['n_centrals'] / r['n_window']:.1f}% of window"],
        ["Eligible Satellites in those Groups", r["n_sat_eligible"],
         f"{100 * r['n_sat_eligible'] / r['n_window']:.1f}% of window"],
        ["Final Sample (Centrals + Max Satellite)", r["n_final"],
         f"{100 * r['n_final'] / r['n_total']:.2f}% of total"],
    ]
    t1 = tabulate(rows, headers=["Selection Step", "Count", "Yield"], tablefmt="fancy_grid")

    rows2 = [
        ["Total Unique Groups", r["n_groups_total"]],
        ["Groups with Satellites", r["n_groups_with_sat"]],
        ["Groups with Central Only", r["n_groups_central_only"]],
    ]
    t2 = tabulate(rows2, tablefmt="simple")
    return t1, t2


# ---------------------------------------------------------------------------
# Pair finding + cutflow
# ---------------------------------------------------------------------------
def pair_finding_report(fields: dict[str, np.ndarray], sample: dict[str, np.ndarray],
                         cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-run pair finding and the isolation/velocity filters on the cached
    `sample` (cheap — no SOAP access) purely to recover diagnostic counts
    at every stage, matching the TNG300 cutflow layout."""
    cfg = cfg or DEFAULT_SELECTION_CONFIG

    raw_pairs = find_flamingo_pairs(
        sample,
        r_min_kpc=cfg["pairs"]["r_min_kpc"],
        r_max_kpc=cfg["pairs"]["r_max_kpc"],
        verbose=False,
    )
    n_raw = len(raw_pairs["i"])

    same_host = int(raw_pairs["have_same_host"].sum()) if n_raw else 0
    diff_host = n_raw - same_host
    blue_blue = int(raw_pairs["is_blue_blue"].sum()) if n_raw else 0
    red_red = int(raw_pairs["is_red_red"].sum()) if n_raw else 0
    blue_red = int(raw_pairs["is_blue_red"].sum()) if n_raw else 0

    if n_raw:
        mean_sep = float(np.mean(raw_pairs["separation_kpc"]))
        v_total = np.sqrt(raw_pairs["vr"] ** 2 + raw_pairs["vt"] ** 2)
        mean_vtot = float(np.mean(v_total))
        mass_ratio_lin = 10.0 ** raw_pairs["log_mass_ratio"]
        mean_mu = float(np.nanmean(mass_ratio_lin))
    else:
        mean_sep = mean_vtot = mean_mu = float("nan")

    # --- cutflow: same steps run_flamingo_pipeline_from_config applies ---
    stage_counts: list[tuple[str, int]] = [("initial", n_raw)]
    pairs = raw_pairs

    x = cfg["filters"].get("third_massive_factor", 1.5)
    m1 = flamingo_isolation_filter_no_third_factor(fields, pairs, sample, third_massive_factor=x)
    pairs = {k: (v[m1] if isinstance(v, np.ndarray) and len(v) == n_raw else v) for k, v in pairs.items()}
    stage_counts.append((f"No third subhalo more massive than factor x={x} in same FoF group", len(pairs["i"])))

    n_before = len(pairs["i"])
    r_dens = cfg["filters"].get("density_radius_kpc", 2000.0)
    x2 = cfg["filters"].get("intruder_factor", 0.5)
    m2 = flamingo_isolation_filter_no_intruder(fields, pairs, sample, density_radius_kpc=r_dens, intruder_factor=x2)
    pairs = {k: (v[m2] if isinstance(v, np.ndarray) and len(v) == n_before else v) for k, v in pairs.items()}
    stage_counts.append((f"No subhalos more massive than factor x={x2} within r={r_dens:.0f} kpc from COM", len(pairs["i"])))

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
        verbose=False,
    )
    vr_lo, vr_hi = cfg["filters"]["vr_min_kms"], cfg["filters"]["vr_max_kms"]
    vt_lo, vt_hi = cfg["filters"]["vt_min_kms"], cfg["filters"]["vt_max_kms"]
    vtot_hi = cfg["filters"]["v_total_max_kms"]
    stage_counts.append(
        (f"vt in [{vt_lo:.0f}, {vt_hi:.0f}], vr in [{vr_lo:.0f}, {vr_hi:.0f}], v_tot in [0, {vtot_hi:.0f}]",
         len(pairs_filtered["i"]))
    )

    return {
        "n_raw": n_raw,
        "same_host": same_host,
        "diff_host": diff_host,
        "blue_blue": blue_blue,
        "red_red": red_red,
        "blue_red": blue_red,
        "mean_sep": mean_sep,
        "mean_vtot": mean_vtot,
        "mean_mu": mean_mu,
        "stage_counts": stage_counts,
        "pairs_final": pairs_filtered,
    }


def pair_finding_tables(fields: dict[str, np.ndarray], sample: dict[str, np.ndarray],
                         cfg: dict[str, Any] | None = None) -> tuple[str, str, dict[str, Any]]:
    r = pair_finding_report(fields, sample, cfg)
    n = r["n_raw"]

    def pct(x: int) -> str:
        return f"{100 * x / n:.1f}%" if n else "n/a"

    rows = [
        ["Total Pairs Found", n, "100.0%"],
        ["Same-host (FoF)", r["same_host"], pct(r["same_host"])],
        ["Different-host", r["diff_host"], pct(r["diff_host"])],
        ["Blue-Blue", r["blue_blue"], pct(r["blue_blue"])],
        ["Red-Red", r["red_red"], pct(r["red_red"])],
        ["Mixed Blue-Red", r["blue_red"], pct(r["blue_red"])],
    ]
    t1 = tabulate(rows, headers=["Category", "Count", "Share"], tablefmt="fancy_grid")

    rows2 = [
        ["Mean Separation", f"{r['mean_sep']:.2f} kpc"],
        ["Mean Relative Velocity", f"{r['mean_vtot']:.2f} km/s"],
        ["Mean Mass Ratio (\u03bc)", f"{r['mean_mu']:.3f}"],
    ]
    t2 = tabulate(rows2, tablefmt="simple")
    return t1, t2, r


def cutflow_table(stage_counts: list[tuple[str, int]]) -> str:
    n0 = stage_counts[0][1]
    prev = n0
    rows = []
    for name, n in stage_counts:
        step_yield = "100.0%" if name == "initial" else (f"{100 * n / prev:.1f}%" if prev else "n/a")
        total_survival = f"{100 * n / n0:.1f}%" if n0 else "n/a"
        rows.append([name, f"{n:,}", step_yield, total_survival])
        prev = n
    return tabulate(rows, headers=["Filter Step", "Pairs Remaining", "Step Yield", "Total Survival"],
                     tablefmt="fancy_grid")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def print_full_report(fields: dict[str, np.ndarray], sample: dict[str, np.ndarray],
                       cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Print the full FLAMINGO diagnostic report and return the pair-finding
    diagnostics dict (includes the final filtered pairs, for a sanity check
    against the `pairs` object already loaded from cache)."""
    print("[DIAGNOSTIC] Simulation Header Constants")
    print(header_table())
    print()

    t1, t2 = sample_selection_tables(fields, cfg)
    print("[DIAGNOSTIC] Sample Selection")
    print(t1)
    print(t2)
    print()

    pt1, pt2, pair_r = pair_finding_tables(fields, sample, cfg)
    print("[DIAGNOSTIC] Pair Finding Results")
    print(pt1)
    print(pt2)
    print(
        "\nNote: 'Tidally Dominant' and 'Force Ratio < 0.5' from the TNG300 report "
        "aren't computed anywhere in flamingo_lg_pipeline.py, so they're left out "
        "here rather than guessed at. Share the TNG300 definition and I can add a "
        "matching filter/diagnostic to the pipeline."
    )
    print()

    print("[PIPELINE] Sample Reduction Cutflow")
    print(cutflow_table(pair_r["stage_counts"]))

    return pair_r
