#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lg_npe.data import load_flamingo_pair_table, load_tng_pair_table


def binned_relation(x, y, edges, simulation):
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (x >= low) & (x < high)
        values = y[mask]
        rows.append({
            "simulation": simulation, "x_low": low, "x_high": high,
            "x_mid": (low + high) / 2, "count": int(mask.sum()),
            "median_log_mstar": float(np.median(values)) if len(values) else np.nan,
            "q16_log_mstar": float(np.quantile(values, .16)) if len(values) else np.nan,
            "q84_log_mstar": float(np.quantile(values, .84)) if len(values) else np.nan,
        })
    return rows


def stellar_vs_subhalo(tables, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    points = {}
    for sim, table in tables.items():
        f = table.frame
        mstar = np.concatenate([f["mstar_a"], f["mstar_b"]]).astype(float)
        msub = np.concatenate([f["Msub_bound_1"], f["Msub_bound_2"]]).astype(float)
        valid = np.isfinite(mstar) & np.isfinite(msub) & (mstar > 0) & (msub > 0)
        points[sim] = (np.log10(msub[valid]), np.log10(mstar[valid]))
    all_x = np.concatenate([value[0] for value in points.values()])
    all_y = np.concatenate([value[1] for value in points.values()])
    xlim = (float(np.min(all_x)), float(np.max(all_x)))
    ylim = (float(np.min(all_y)), float(np.max(all_y)))
    edges = np.linspace(xlim[0], xlim[1], 21)
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, (sim, (x, y)) in zip(axes, points.items()):
        if len(x) > 20_000:
            artist = ax.hexbin(x, y, gridsize=70, bins="log", mincnt=1, cmap="viridis")
            fig.colorbar(artist, ax=ax, label="log10 count")
        else:
            ax.scatter(x, y, s=7, alpha=.3, edgecolors="none")
        rows = binned_relation(x, y, edges, sim)
        summaries.extend(rows)
        summary = pd.DataFrame(rows)
        use = summary["count"] > 0
        ax.plot(summary.loc[use, "x_mid"], summary.loc[use, "median_log_mstar"], color="crimson")
        ax.fill_between(
            summary.loc[use, "x_mid"], summary.loc[use, "q16_log_mstar"],
            summary.loc[use, "q84_log_mstar"], color="crimson", alpha=.2,
        )
        ax.set(title=f"{sim.upper()} (N={len(x):,} pair members)", xlabel="log10 Msub,bound [Msun]",
               xlim=xlim, ylim=ylim)
    axes[0].set_ylabel("log10 Mstar,bound [Msun]")
    fig.tight_layout()
    fig.savefig(output / "stellar_vs_bound_subhalo_mass.png", dpi=180)
    fig.savefig(output / "stellar_vs_bound_subhalo_mass.pdf")
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(output / "binned_relation.csv", index=False)
    marginal = {
        sim: {
            "pair_member_count": len(x),
            "log_msub": {"median": float(np.median(x)), "q16": float(np.quantile(x,.16)), "q84": float(np.quantile(x,.84))},
            "log_mstar": {"median": float(np.median(y)), "q16": float(np.quantile(y,.16)), "q84": float(np.quantile(y,.84))},
        } for sim, (x, y) in points.items()
    }
    (output / "marginal_summary.json").write_text(json.dumps(marginal, indent=2), encoding="utf-8")


def wilson_interval(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre - half, centre + half


def host_fraction(tables, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(500.0, 1000.0, 11)
    rows = []
    for sim, table in tables.items():
        frame = table.frame
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (frame["separation"] >= low) & (frame["separation"] < high)
            n = int(mask.sum())
            diff = int((~frame.loc[mask, "same_host"]).sum())
            lo, hi = wilson_interval(diff, n)
            rows.append({"simulation": sim, "r_low_kpc": low, "r_high_kpc": high,
                         "r_mid_kpc": (low+high)/2, "n_total": n,
                         "n_same_host": n-diff, "n_different_host": diff,
                         "f_different_host": diff/n if n else np.nan,
                         "ci95_low": lo, "ci95_high": hi})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "different_host_fraction_by_separation.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sim, marker in (("flamingo", "o"), ("tng", "s")):
        sub = summary[summary.simulation == sim]
        y = sub.f_different_host.to_numpy()
        err = np.vstack([y-sub.ci95_low.to_numpy(), sub.ci95_high.to_numpy()-y])
        ax.errorbar(sub.r_mid_kpc, y, yerr=err, marker=marker, capsize=3, label=sim.upper())
        if sim == "tng":
            for _, row in sub.iterrows():
                ax.annotate(f"n={int(row.n_total)}", (row.r_mid_kpc, row.f_different_host),
                            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    ax.set(xlabel="3D pair separation [physical kpc]", ylabel="Different-host fraction",
           xlim=(495,1005), ylim=(0,1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "different_host_fraction_vs_separation.png", dpi=180)
    fig.savefig(output / "different_host_fraction_vs_separation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--diagnostic", choices=["all", "stellar", "host_fraction"], default="all")
    args = parser.parse_args()
    tables = {"flamingo": load_flamingo_pair_table(args.project), "tng": load_tng_pair_table(args.project)}
    if args.diagnostic in {"all", "stellar"}:
        stellar_vs_subhalo(tables, args.project / "out/mass_comparison/stellar_vs_subhalo")
    if args.diagnostic in {"all", "host_fraction"}:
        host_fraction(tables, args.project / "out/mass_comparison/host_fraction")


if __name__ == "__main__":
    main()
