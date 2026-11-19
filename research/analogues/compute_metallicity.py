"""compute_metallicity.py
TNG300 stellar metallicity - FLAMINGO-comparable definition.

Computes a mass-weighted stellar metal fraction designed to be comparable
to FLAMINGO's BoundSubhalo/StellarMassFractionInMetals:

    Z_star = sum(m_i * Z_i) / sum(m_i)

where m_i is the stellar particle mass and Z_i is the stellar metal mass
fraction.

- use star particles only (PartType4)
- use GFM_Metallicity as the stellar metal mass fraction
- exclude particles with GFM_StellarFormationTime <= 0 as a
  TNG-specific particle-quality cut
- require a minimum of 100 valid star particles as a TNG resolution cut

The FLAMINGO SOAP implementation calculates the corresponding property
from the stellar particles selected for the bound subhalo. The 100-particle
threshold and formation-time cut are additional TNG selection criteria and
are not part of the SOAP definition itself.
Output:
  - tng_metallicity.pkl: {global_subhalo_id -> Z_star}
  - metallicity.npy: sample-aligned array ordered by keep_idx
Resumes from checkpoint if the pickle already exists.
"""

import io
import os
import pickle
import random
import time
from pathlib import Path

import h5py
import numpy as np
import requests


# --- Config ---------------------------------------------------------------

TNG_API_KEY = os.environ.get("TNG_API_KEY")
if not TNG_API_KEY:
    raise RuntimeError("Set TNG_API_KEY in the environment; credentials are never stored in source.")
HEADERS = {"api-key": TNG_API_KEY}

SIM = "TNG300-1"
SNAP = 99
BASE_URL = f"https://www.tng-project.org/api/{SIM}/snapshots/{SNAP}/subhalos"

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT_DIR = BASE_DIR / "tng_for_comparison_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "tng_metallicity.pkl"

MIN_PARTICLES = 100
CHECKPOINT = 50
SLEEP_MIN = 0.5
SLEEP_MAX = 1.5
SOLAR_METALLICITY = 0.0127


# --- Load sample indices --------------------------------------------------

keep_idx = np.load(OUT_DIR / "keep_idx.npy")
pairs_i = np.load(OUT_DIR / "pairs_i.npy")
pairs_j = np.load(OUT_DIR / "pairs_j.npy")

pairs_i_global = keep_idx[pairs_i]
pairs_j_global = keep_idx[pairs_j]
unique_targets = np.unique(np.concatenate([pairs_i_global, pairs_j_global]))
print(f"Unique subhalos to compute: {len(unique_targets):,}")


# --- Resume from checkpoint ----------------------------------------------

if OUT_PATH.exists():
    with open(OUT_PATH, "rb") as f:
        metallicity_cache = pickle.load(f)
    print(f"Resuming from checkpoint: {len(metallicity_cache):,} already cached")
else:
    metallicity_cache = {}

remaining = [int(x) for x in unique_targets if int(x) not in metallicity_cache]
print(f"Remaining: {len(remaining):,}")


# --- Fetch function -------------------------------------------------------

def fetch_metallicity(sub_id: int, max_retries: int = 5) -> float:
    """Fetch star particles for one subhalo and compute stellar metallicity.

    Returns np.nan if:
      - subhalo not found (404)
      - fewer than MIN_PARTICLES real star particles
      - any fetch/parse error
    """
    url = f"{BASE_URL}/{sub_id}/cutout.hdf5"
    params = {"stars": "Masses,GFM_Metallicity,GFM_StellarFormationTime"}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)
            if r.status_code == 404:
                return np.nan
            r.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                return np.nan
            time.sleep((2 ** attempt) + random.uniform(0, 1))

    try:
        with h5py.File(io.BytesIO(r.content), "r") as f:
            if "PartType4" not in f:
                return np.nan

            mass = f["PartType4"]["Masses"][:]
            metal = f["PartType4"]["GFM_Metallicity"][:]
            form_time = f["PartType4"]["GFM_StellarFormationTime"][:]
    except Exception:
        return np.nan

    # Exclude wind particles: only real stars contribute.
    real = form_time > 0
    mass = mass[real]
    metal = metal[real]

    valid = np.isfinite(mass) & np.isfinite(metal) & (mass > 0)
    mass = mass[valid]
    metal = metal[valid]

    if mass.size < MIN_PARTICLES:
        return np.nan

    return float(np.average(metal, weights=mass))


# --- Main loop ------------------------------------------------------------

for i, sid in enumerate(remaining, start=1):
    metallicity_cache[sid] = fetch_metallicity(sid)
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    if i % CHECKPOINT == 0 or i == len(remaining):
        with open(OUT_PATH, "wb") as f:
            pickle.dump(metallicity_cache, f)
        print(f"cached: {len(metallicity_cache):,}/{len(unique_targets):,}")

print(f"\nDONE -> {OUT_PATH}")


# --- Save notebook-friendly array ----------------------------------------

sample_metallicity = np.array(
    [metallicity_cache.get(int(x), np.nan) for x in keep_idx],
    dtype=np.float64,
)
np.save(OUT_DIR / "metallicity.npy", sample_metallicity)


# --- Summary --------------------------------------------------------------

met_i = sample_metallicity[pairs_i]
met_j = sample_metallicity[pairs_j]

met_pairs = np.concatenate([met_i, met_j])
met_pairs = met_pairs[np.isfinite(met_pairs) & (met_pairs > 0)]

print("\nStellar metallicity summary (FLAMINGO-equivalent):")
print(f"  N:      {len(met_pairs):,}")
if len(met_pairs) > 0:
    print(f"  Median: {np.median(met_pairs):.6f}")
    print(f"  Median: {np.median(met_pairs) / SOLAR_METALLICITY:.3f} Z_sun")
    print(f"  Range:  {met_pairs.min():.6f} - {met_pairs.max():.6f}")
else:
    print("  No valid metallicities found.")

print(f"\nSaved sample-aligned array -> {OUT_DIR / 'metallicity.npy'}")
