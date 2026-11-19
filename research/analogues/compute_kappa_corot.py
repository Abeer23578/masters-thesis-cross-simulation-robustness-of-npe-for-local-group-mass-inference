"""compute_kappa_corot.py
TNG300-1 stellar κ_co (kappa corotation) — FLAMINGO-equivalent definition.

Matches BoundSubhalo/KappaCorotStars from FLAMINGO SOAP (Sales et al. 2012):
  - Positions relative to centre of potential (SubhaloPos)
  - Velocities relative to stellar centre of mass velocity (mass-weighted)
  - All bound stellar particles, no aperture cut
  - Wind particles excluded (GFM_StellarFormationTime <= 0)
  - Minimum 100 star particles (FLAMINGO Mask Threshold = 100)
  - κ_co = K_rot / K_tot
      K_tot = Σ ½ m_i |v_i|²
      K_rot = Σ ½ m_i (L_z,i / (m_i R_i))²   [only co-rotating: L_z,i > 0]
      where ẑ = direction of net stellar angular momentum
            R_i = projected distance from spin axis
            L_z,i = m_i (r_i × v_i) · ẑ

Output: dict {global_subhalo_id (int) -> kappa_co (float, dimensionless [0, 1])}
Saved to: CACHE_DIR / "tng_kappa_corot.pkl"
Resumes from checkpoint if pkl already exists.
"""

import io
import pickle
import random
import time
from pathlib import Path

import h5py
import illustris_python as il
import numpy as np
import requests
import os


# ── Config ────────────────────────────────────────────────────────────────────

TNG_API_KEY = os.environ.get("TNG_API_KEY")
if not TNG_API_KEY:
    raise RuntimeError("Set TNG_API_KEY in the environment; credentials are never stored in source.")
HEADERS = {"api-key": TNG_API_KEY}

SIM      = "TNG300-1"
SNAP     = 99
BASE_URL = f"https://www.tng-project.org/api/{SIM}/snapshots/{SNAP}/subhalos"

# local TNG300 path for fast groupcat access
BASE_PATH = os.environ.get("TNG_BASE")
if not BASE_PATH:
    raise RuntimeError("Set TNG_BASE to the local TNG300 group-catalogue directory.")

PROJECT = Path(os.environ.get("LG_PROJECT", str(Path(__file__).resolve().parents[1])))
TNG_DIR = PROJECT / "analogues/tng_for_comparison_v3"
CACHE_DIR = PROJECT / "analogues/cache/tng_features_v3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH  = CACHE_DIR / "tng_kappa_corot.pkl"

BOX_CKPCH     = 205000.0   # TNG300 box in ckpc/h (205 Mpc/h)
MIN_PARTICLES = 100         # matches FLAMINGO Mask Threshold
CHECKPOINT    = 50          # save every N subhalos
SLEEP_MIN     = 0.5
SLEEP_MAX     = 1.5


# ── Load pair indices ─────────────────────────────────────────────────────────

keep_idx    = np.load(TNG_DIR / "keep_idx.npy")
tng_pairs_i = np.load(TNG_DIR / "pairs_i.npy")
tng_pairs_j = np.load(TNG_DIR / "pairs_j.npy")

pairs_i_global = keep_idx[tng_pairs_i]
pairs_j_global = keep_idx[tng_pairs_j]

unique_subhalos = np.unique(np.concatenate([pairs_i_global, pairs_j_global]))
print(f"Unique subhalos to compute: {len(unique_subhalos):,}")


# ── Load SubhaloPos from local groupcat (fast, no API needed) ─────────────────
# SubhaloPos = position of minimum potential particle, in ckpc/h
# This is the centre of potential FLAMINGO uses for position reference

print("Loading SubhaloPos from local groupcat...")
sub_pos_all = il.groupcat.loadSubhalos(BASE_PATH, SNAP, fields=["SubhaloPos"])
# sub_pos_all shape: (N_subhalos, 3), units ckpc/h
print(f"SubhaloPos loaded: {sub_pos_all.shape}")


# ── Resume from checkpoint ────────────────────────────────────────────────────

if OUT_PATH.exists():
    with open(OUT_PATH, "rb") as f:
        kappa_cache = pickle.load(f)
    print(f"Resuming from checkpoint: {len(kappa_cache):,} already cached")
else:
    kappa_cache = {}

remaining = [int(x) for x in unique_subhalos if int(x) not in kappa_cache]
print(f"Remaining: {len(remaining):,}")


# ── Fetch function ────────────────────────────────────────────────────────────

def fetch_kappa_corot(sub_id: int, max_retries: int = 5) -> float:
    """Fetch star particles for one subhalo and compute κ_co.

    Returns np.nan if:
      - subhalo not found (404)
      - fewer than MIN_PARTICLES real star particles
      - net angular momentum is near zero (degenerate case)
      - any fetch/parse error
    """
    url    = f"{BASE_URL}/{sub_id}/cutout.hdf5"
    params = {"stars": "Coordinates,Velocities,Masses,GFM_StellarFormationTime"}

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
            pos       = f["PartType4"]["Coordinates"][:]              # ckpc/h
            vel       = f["PartType4"]["Velocities"][:]               # km/s (a=1 at z=0)
            mass      = f["PartType4"]["Masses"][:]                   # 1e10 Msun/h
            form_time = f["PartType4"]["GFM_StellarFormationTime"][:]
    except Exception:
        return np.nan

    # exclude wind particles
    real = form_time > 0
    pos  = pos[real]
    vel  = vel[real]
    mass = mass[real]

    if len(mass) < MIN_PARTICLES:
        return np.nan

    # ── Reference frames ──────────────────────────────────────────────────────

    # positions: relative to centre of potential (SubhaloPos), with periodic wrap
    centre = sub_pos_all[sub_id]                          # ckpc/h
    pos    = pos - centre
    pos   -= BOX_CKPCH * np.round(pos / BOX_CKPCH)       # minimum image

    # velocities: relative to mass-weighted stellar CoM velocity
    total_mass = mass.sum()
    vel_com    = np.sum(mass[:, None] * vel, axis=0) / total_mass
    vel        = vel - vel_com

    # ── κ_co (Sales et al. 2012) ─────────────────────────────────────────────

    # net stellar angular momentum direction (spin axis ẑ)
    L_vec  = np.sum(mass[:, None] * np.cross(pos, vel), axis=0)
    L_norm = np.linalg.norm(L_vec)
    if L_norm < 1e-30:
        return np.nan
    z_hat = L_vec / L_norm

    # per-particle L_z (scalar projection onto spin axis)
    L_z = np.sum(np.cross(pos, vel) * z_hat, axis=1) * mass   # ckpc/h * km/s * 1e10 Msun/h

    # projected radius from spin axis
    pos_along_z = np.outer(np.dot(pos, z_hat), z_hat)
    pos_perp    = pos - pos_along_z
    R           = np.linalg.norm(pos_perp, axis=1)             # ckpc/h

    # circular velocity for co-rotating particles only (L_z > 0)
    v_phi = np.zeros(len(mass))
    corot = (L_z > 0) & (R > 0)
    v_phi[corot] = L_z[corot] / (mass[corot] * R[corot])      # km/s

    # kinetic energies (mass prefactor cancels in ratio)
    K_tot = np.sum(mass * np.sum(vel ** 2, axis=1))
    K_rot = np.sum(mass * v_phi ** 2)

    if K_tot <= 0:
        return np.nan

    return float(K_rot / K_tot)


# ── Main loop ─────────────────────────────────────────────────────────────────

for i, sid in enumerate(remaining, start=1):
    kappa_cache[sid] = fetch_kappa_corot(sid)
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    if i % CHECKPOINT == 0 or i == len(remaining):
        with open(OUT_PATH, "wb") as f:
            pickle.dump(kappa_cache, f)
        print(f"cached: {len(kappa_cache):,}/{len(unique_subhalos):,}")

print(f"\nDONE → {OUT_PATH}")


# ── Summary ───────────────────────────────────────────────────────────────────

kap_i = np.array([kappa_cache.get(int(x), np.nan) for x in pairs_i_global])
kap_j = np.array([kappa_cache.get(int(x), np.nan) for x in pairs_j_global])

kap_pairs = np.concatenate([kap_i, kap_j])
kap_pairs = kap_pairs[np.isfinite(kap_pairs) & (kap_pairs >= 0) & (kap_pairs <= 1)]

print(f"\nκ_co (FLAMINGO-equivalent):")
print(f"  N:              {len(kap_pairs):,}")
print(f"  Median:         {np.median(kap_pairs):.4f}")
print(f"  Range:          {kap_pairs.min():.4f} – {kap_pairs.max():.4f}")
print(f"  Disc frac (>0.35): {np.mean(kap_pairs > 0.35):.3f}")
