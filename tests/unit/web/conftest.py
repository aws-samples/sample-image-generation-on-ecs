from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_WEB_BACKEND = str(Path(__file__).resolve().parents[3] / "web" / "backend")
if _WEB_BACKEND not in sys.path:
    sys.path.insert(0, _WEB_BACKEND)

os.environ.setdefault("ZIMAGE_LOCAL", "true")
os.environ.setdefault("ZIMAGE_GALLERY_BUCKET", "test-bucket")
os.environ.setdefault("ZIMAGE_GALLERY_TABLE", "test-table")
os.environ.setdefault("ZIMAGE_REGION", "us-east-1")
os.environ.setdefault("ZIMAGE_PREFIX", "test-zimg")

_config_spec = importlib.util.spec_from_file_location(
    "config", os.path.join(_WEB_BACKEND, "config.py")
)
_config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules["config"] = _config_mod
_config_spec.loader.exec_module(_config_mod)
