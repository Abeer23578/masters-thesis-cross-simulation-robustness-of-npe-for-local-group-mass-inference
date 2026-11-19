"""compute_sigma_star.py
TNG300-1 stellar velocity dispersion — FLAMINGO-equivalent definition.

Matches BoundSubhalo/StellarVelocityDispersionMatrix from FLAMINGO SOAP:
  - All bound stellar particles, no aperture cut
  - Mass-weighted CoM velocity subtracted
  - Wind particles excluded (GFM_StellarFormationTime <= 0)
  - Minimum 100 star particles (FLAMINGO metadata: Mask Threshold = 100)
  - sigma = sqrt((sigma_xx^2 + sigma_yy^2 + sigma_zz^2) / 3)

Output: dict {global_subhalo_id (int) -> sigma_star (float, km/s)}
Saved to: cache_dir / "tng_sigma_star.pkl"
Resumes from checkpoint if pkl already exists.
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


# ── Config ────────────────────────────────────────────────────────────────────

TNG_API_KEY = os.environ.get("TNG_API_KEY")
if not TNG_API_KEY:
    raise RuntimeError("Set TNG_API_KEY in the environment; credentials are never stored in source.")
HEADERS = {"api-key": TNG_API_KEY}

SIM       = "TNG300-1"
SNAP      = 99
BASE_URL  = f"https://www.tng-project.org/api/{SIM}/snapshots/{SNAP}/subhalos"

PROJECT = Path(os.environ.get("LG_PROJECT", str(Path(__file__).resolve().parents[1])))
TNG_DIR = PROJECT / "analogues/tng_for_comparison_v3"
CACHE_DIR = PROJECT / "analogues/cache/tng_features_v3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH  = CACHE_DIR / "tng_sigma_star.pkl"

MIN_PARTICLES = 100    # matches FLAMINGO Mask Threshold
CHECKPOINT    = 50     # save every N subhalos
SLEEP_MIN     = 0.5    # API rate-limit buffer
SLEEP_MAX     = 1.5


# ── Load pair indices ─────────────────────────────────────────────────────────

keep_idx    = np.load(TNG_DIR / "keep_idx.npy")
tng_pairs_i = np.load(TNG_DIR / "pairs_i.npy")
tng_pairs_j = np.load(TNG_DIR / "pairs_j.npy")

# Translate sample-level pair indices to global subhalo IDs
pairs_i_global = keep_idx[tng_pairs_i]
pairs_j_global = keep_idx[tng_pairs_j]

unique_subhalos = np.unique(np.concatenate([pairs_i_global, pairs_j_global]))
print(f"Unique subhalos to compute: {len(unique_subhalos):,}")


# ── Resume from checkpoint ────────────────────────────────────────────────────

if OUT_PATH.exists():
    with open(OUT_PATH, "rb") as f:
        sigma_cache = pickle.load(f)
    print(f"Resuming from checkpoint: {len(sigma_cache):,} already cached")
else:
    sigma_cache = {}

remaining = [int(x) for x in unique_subhalos if int(x) not in sigma_cache]
print(f"Remaining: {len(remaining):,}")


# ── Fetch function ────────────────────────────────────────────────────────────

def fetch_sigma_star(sub_id: int, max_retries: int = 5) -> float:
    """Fetch star particles for one subhalo and compute stellar sigma.

    Returns np.nan if:
      - subhalo not found (404)
      - fewer than MIN_PARTICLES real star particles
      - any fetch/parse error
    """
    url    = f"{BASE_URL}/{sub_id}/cutout.hdf5"
    params = {"stars": "Velocities,Masses,GFM_StellarFormationTime"}

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
            vel       = f["PartType4"]["Velocities"][:]
            mass      = f["PartType4"]["Masses"][:]
            form_time = f["PartType4"]["GFM_StellarFormationTime"][:]
    except Exception:
        return np.nan

    # exclude wind particles (formation time <= 0 marks winds, not real stars)
    real = form_time > 0
    vel  = vel[real]
    mass = mass[real]

    if len(vel) < MIN_PARTICLES:
        return np.nan

    # mass-weighted CoM velocity — matches FLAMINGO "relative to stellar CoM velocity"
    total_mass = mass.sum()
    vel_com    = np.sum(mass[:, None] * vel, axis=0) / total_mass
    vel        = vel - vel_com

    # mass-weighted diagonal velocity dispersion tensor
    sigma2_x = np.sum(mass * vel[:, 0] ** 2) / total_mass
    sigma2_y = np.sum(mass * vel[:, 1] ** 2) / total_mass
    sigma2_z = np.sum(mass * vel[:, 2] ** 2) / total_mass

    # RMS of diagonal — matches FLAMINGO sqrt((sigma_xx + sigma_yy + sigma_zz) / 3)
    return float(np.sqrt((sigma2_x + sigma2_y + sigma2_z) / 3))


# ── Main loop ─────────────────────────────────────────────────────────────────

for i, sid in enumerate(remaining, start=1):
    sigma_cache[sid] = fetch_sigma_star(sid)
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    if i % CHECKPOINT == 0 or i == len(remaining):
        with open(OUT_PATH, "wb") as f:
            pickle.dump(sigma_cache, f)
        print(f"cached: {len(sigma_cache):,}/{len(unique_subhalos):,}")

print(f"\nDONE → {OUT_PATH}")


# ── Summary ───────────────────────────────────────────────────────────────────

sig_i = np.array([sigma_cache.get(int(x), np.nan) for x in pairs_i_global])
sig_j = np.array([sigma_cache.get(int(x), np.nan) for x in pairs_j_global])

sig_pairs = np.concatenate([sig_i, sig_j])
sig_pairs = sig_pairs[np.isfinite(sig_pairs) & (sig_pairs > 0)]

print(f"\nStellar sigma (FLAMINGO-equivalent):")
print(f"  N:      {len(sig_pairs):,}")
print(f"  Median: {np.median(sig_pairs):.2f} km/s")
print(f"  Range:  {sig_pairs.min():.2f} – {sig_pairs.max():.2f} km/s")
