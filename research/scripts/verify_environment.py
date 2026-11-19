#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lg_npe.registry import environment_record

record = environment_record()
print(json.dumps(record, indent=2))
required = {"python": "3.12.14", "torch": "2.13.0", "sbi": "0.27.0"}
actual = {"python": record["python"], "torch": record["packages"]["torch"], "sbi": record["packages"]["sbi"]}
mismatch = {key: {"actual": actual[key], "required": value} for key, value in required.items()
            if actual[key] is None or actual[key].split("+")[0] != value}
if mismatch:
    raise SystemExit("Environment mismatch: " + json.dumps(mismatch))
if not record["cuda_available"]:
    raise SystemExit("CUDA is not available")
