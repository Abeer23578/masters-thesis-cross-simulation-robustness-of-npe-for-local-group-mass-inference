#!/usr/bin/env python
"""Capture SOAP field attributes when the raw FLAMINGO catalogue is accessible."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analogues"))
import flamingo_lg_pipeline as pipeline


def serializable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [serializable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "data/derived/soap_field_metadata_v1.json",
    )
    args = parser.parse_args()
    soap = pipeline.open_default_soap()
    requested = ("stellar_mass", "bound_total_mass", "m200c")
    fields = {}
    for name in requested:
        path = pipeline.FIELD_PATHS[name]
        dataset = soap[path]
        fields[name] = {
            "path": path,
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "attributes": {str(key): serializable(value) for key, value in dataset.attrs.items()},
            "pipeline_conversion": (
                "multiply by 1e10 to physical Msun; no h correction"
                if name in {"stellar_mass", "bound_total_mass", "m200c"} else None
            ),
        }
    record = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalogue": pipeline.DEFAULT_SOAP_PATH,
        "fields": fields,
        "systematic_note": (
            "FLAMINGO HBT-HERONS/SOAP and TNG SUBFIND use different structure finders; "
            "the catalogue definitions are matched but no correction is applied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        comparable_previous = {k: v for k, v in previous.items() if k != "created_utc"}
        comparable_new = {k: v for k, v in record.items() if k != "created_utc"}
        if comparable_previous != comparable_new:
            raise FileExistsError(f"Refusing to replace different SOAP metadata: {args.output}")
        print(f"Verified existing metadata: {args.output}")
        return
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
