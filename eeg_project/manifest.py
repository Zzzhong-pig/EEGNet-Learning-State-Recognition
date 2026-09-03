"""Model deployment manifest for traceability."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    model_path: str | Path,
    preprocessing_path: str | Path,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path)
    preprocessing_path = Path(preprocessing_path)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
            "bytes": model_path.stat().st_size,
        },
        "preprocessing": {
            "path": str(preprocessing_path),
            "sha256": file_sha256(preprocessing_path),
            "bytes": preprocessing_path.stat().st_size,
        },
    }
    if config:
        manifest["training_config"] = config
    if metrics:
        manifest["validation_metrics"] = metrics
    return manifest


def save_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path
