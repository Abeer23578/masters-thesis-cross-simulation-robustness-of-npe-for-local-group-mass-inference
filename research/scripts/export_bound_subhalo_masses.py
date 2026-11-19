#!/usr/bin/env python
"""Create non-destructive pair-level total-bound-subhalo-mass sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_non_destructive(output: Path, manifest_path: Path, arrays: dict, manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not manifest_path.exists():
            raise FileNotFoundError(f"Sidecar exists without its manifest: {output}")
        with np.load(output) as existing:
            if set(existing.files) != set(arrays):
                raise FileExistsError(f"Refusing to replace sidecar with different fields: {output}")
            for name, value in arrays.items():
                np.testing.assert_array_equal(
                    existing[name], value,
                    err_msg=f"Refusing to replace different sidecar array {name}: {output}",
                )
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_old = {k: v for k, v in old_manifest.items() if k not in {"created_utc", "npz_sha256"}}
        comparable_new = {k: v for k, v in manifest.items() if k != "created_utc"}
        if comparable_old != comparable_new or old_manifest.get("npz_sha256") != sha256(output):
            raise ValueError(f"Existing sidecar manifest differs: {manifest_path}")
        print(f"Verified existing sidecar: {output}")
        return
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    new_hash = sha256(tmp)
    manifest = {**manifest, "npz_sha256": new_hash}
    tmp.replace(output)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


def export_flamingo(project: Path) -> None:
    cache = project / "analogues/cache/run_001_mstar2e10"
    with (cache / "flamingo_sample.pkl").open("rb") as stream:
        sample = pickle.load(stream)
    with (cache / "flamingo_pairs.pkl").open("rb") as stream:
        pairs = pickle.load(stream)
    m1 = np.asarray(sample["bound_total_mass"][pairs["i"]], dtype=np.float64)
    m2 = np.asarray(sample["bound_total_mass"][pairs["j"]], dtype=np.float64)
    arrays = {
        "pair_i": np.asarray(pairs["i"], dtype=np.int64),
        "pair_j": np.asarray(pairs["j"], dtype=np.int64),
        "Msub_bound_1": m1,
        "Msub_bound_2": m2,
        "Msub_bound_sum": m1 + m2,
    }
    np.testing.assert_allclose(np.log10(m1 + m2), pairs["log_total_bound_mass"], atol=3e-6)
    root = project / "data/derived/bound_subhalo_masses"
    manifest = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simulation": "FLAMINGO L1_m8 z=0",
        "source_field": "BoundSubhalo/TotalMass",
        "units": "physical Msun",
        "conversion": "SOAP value multiplied by 1e10; no h correction",
        "pair_endpoint_order": "authoritative pair i/j; ordering is irrelevant to the sum target",
        "pair_count": len(m1),
        "source_files": [str(cache / "flamingo_sample.pkl"), str(cache / "flamingo_pairs.pkl")],
    }
    write_non_destructive(
        root / "flamingo_bound_subhalo_masses_v1.npz",
        root / "flamingo_bound_subhalo_masses_v1.json", arrays, manifest,
    )


def export_tng(project: Path, tng_base: Path) -> None:
    import illustris_python as il

    data_dir = project / "analogues/tng_for_comparison_v3"
    keep_idx = np.load(data_dir / "keep_idx.npy")
    pair_i = np.load(data_dir / "pairs_i.npy")
    pair_j = np.load(data_dir / "pairs_j.npy")
    header = il.groupcat.loadHeader(str(tng_base), 99)
    h = float(header["HubbleParam"])
    mass_raw = il.groupcat.loadSubhalos(str(tng_base), 99, fields=["SubhaloMass"])
    sample_mass = np.asarray(mass_raw[keep_idx], dtype=np.float64) * 1e10 / h
    m1, m2 = sample_mass[pair_i], sample_mass[pair_j]
    arrays = {
        "pair_i": np.asarray(pair_i, dtype=np.int64),
        "pair_j": np.asarray(pair_j, dtype=np.int64),
        "global_subhalo_id_1": np.asarray(keep_idx[pair_i], dtype=np.int64),
        "global_subhalo_id_2": np.asarray(keep_idx[pair_j], dtype=np.int64),
        "Msub_bound_1": m1,
        "Msub_bound_2": m2,
        "Msub_bound_sum": m1 + m2,
    }
    ratio = np.log10(np.minimum(m1, m2) / np.maximum(m1, m2))
    np.testing.assert_allclose(ratio, np.load(data_dir / "log_mass_ratio.npy"), atol=3e-6)
    root = project / "data/derived/bound_subhalo_masses"
    manifest = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simulation": "TNG300-1 snapshot 99 z=0",
        "source_field": "SubhaloMass",
        "units": "physical Msun",
        "stored_units": "1e10 Msun/h",
        "h": h,
        "conversion": "SubhaloMass multiplied by 1e10/h",
        "definition": "total mass of all gravitationally bound member particles/cells",
        "pair_endpoint_order": "authoritative pair i/j; ordering is irrelevant to the sum target",
        "pair_count": len(m1),
        "source_catalogue": str(tng_base.resolve()),
        "authoritative_selection_manifest": str(data_dir / "selection_manifest.json"),
    }
    write_non_destructive(
        root / "tng_bound_subhalo_masses_v1.npz",
        root / "tng_bound_subhalo_masses_v1.json", arrays, manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(os.environ.get("LG_PROJECT", PROJECT)))
    parser.add_argument("--tng-base", type=Path, default=os.environ.get("TNG_BASE"))
    parser.add_argument("--simulation", choices=["flamingo", "tng", "both"], default="both")
    args = parser.parse_args()
    if args.simulation in {"flamingo", "both"}:
        export_flamingo(args.project)
    if args.simulation in {"tng", "both"}:
        if args.tng_base is None:
            parser.error("--tng-base or TNG_BASE is required for the TNG export")
        export_tng(args.project, args.tng_base)


if __name__ == "__main__":
    main()
