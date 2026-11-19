"""Regenerate analogues/tng_for_comparison_v3/grnr.npy.

Provenance record for the TNG300 FoF group numbers used as host identifiers in
pair grouping (splits.pair_groups, and host_i/host_j in the TNG feature table).

Requires illustris_python and the raw TNG-300-1 group catalogue, neither of
which is available on Roihu -- run this on the machine holding the TNG data.
Originally executed as an inline cell in inference_npe.ipynb; that cell was
removed when the notebook was ported to the cluster.
"""
import os
from pathlib import Path

import numpy as np
import illustris_python as il

TNG_BASE = os.environ.get("TNG_BASE", r"<TNG_BASE>")
PROJECT  = Path(os.environ.get("LG_PROJECT", r"<research-project>"))
TNG_DIR  = PROJECT / "analogues" / "tng_for_comparison_v3"
SNAPSHOT = 99

keep_idx = np.load(TNG_DIR / "keep_idx.npy")
grnr = il.groupcat.loadSubhalos(TNG_BASE, SNAPSHOT, fields=["SubhaloGrNr"])[keep_idx]

out = TNG_DIR / "grnr.npy"
if out.exists():
    existing = np.load(out)
    if np.array_equal(existing, grnr):
        print(f"{out.name}: regenerated array matches the stored one exactly "
              f"(shape {grnr.shape}, {len(np.unique(grnr)):,} unique groups)")
    else:
        print(f"WARNING: regenerated array DIFFERS from {out.name} -- not overwriting.")
        raise SystemExit(1)
else:
    np.save(out, grnr)
    print(f"wrote {out}  shape={grnr.shape}  unique groups={len(np.unique(grnr)):,}")
