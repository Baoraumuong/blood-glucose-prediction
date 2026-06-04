"""
Model artifact persistence helpers.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any


def safe_name(value: str) -> str:
    """Return a filesystem-friendly artifact name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "artifact"


def artifact_dir(cfg: dict[str, Any], model_name: str, feature_tag: str) -> Path:
    root = Path(cfg["paths"].get("checkpoints_dir", "output/checkpoints"))
    path = root / safe_name(model_name) / safe_name(feature_tag)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle_artifact(
    cfg: dict[str, Any],
    model_name: str,
    feature_tag: str,
    filename: str,
    payload: Any,
) -> Path:
    path = artifact_dir(cfg, model_name, feature_tag) / filename
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_pickle_artifact(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)
