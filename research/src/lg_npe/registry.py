from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = None
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
        "packages": {name: _version(name) for name in (
            "torch", "sbi", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib",
        )},
        "roihu_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        import torch
        record["cuda_available"] = bool(torch.cuda.is_available())
        record["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        record["cuda_available"] = None
        record["cuda_device"] = None
    return record


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    def convert(value):
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except ImportError:
            pass
        if isinstance(value, Path):
            return str(value)
        return str(value)

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert), encoding="utf-8")
