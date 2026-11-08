from __future__ import annotations


def is_flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def is_flash_attn_3_available() -> bool:
    try:
        import flash_attn

        return hasattr(flash_attn, "flash_attn_func_v3")
    except ImportError:
        return False


def is_torch_version(op: str, version: str) -> bool:
    import torch
    from packaging.version import Version

    installed = Version(torch.__version__.split("+")[0])
    target = Version(version)

    if op == ">=":
        return installed >= target
    elif op == "<=":
        return installed <= target
    elif op == ">":
        return installed > target
    elif op == "<":
        return installed < target
    elif op == "==":
        return installed == target
    elif op == "!=":
        return installed != target
    else:
        raise ValueError(f"Unsupported operator: {op!r}")
