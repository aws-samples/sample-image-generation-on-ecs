from __future__ import annotations

import hashlib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_GRAPH_AFFECTING_FILES: list[str] = [
    "runtime/neuron/config/inference.py",
    "runtime/neuron/config/model.py",
    "runtime/neuron/runtime.py",
    "runtime/neuron/utils/attention.py",
    "runtime/neuron/utils/helpers.py",
    "runtime/neuron/utils/loader.py",
    "runtime/serving/production/pipeline_1024.py",
    "runtime/serving/production/pipeline_768.py",
]


def compute_cache_version(project_root: Path | None = None) -> str:
    root = project_root or _PROJECT_ROOT
    h = hashlib.sha256()

    for rel_path in _GRAPH_AFFECTING_FILES:
        full = root / rel_path
        if full.is_file():
            h.update(full.read_bytes())
        else:
            h.update(rel_path.encode())

    return h.hexdigest()[:12]
