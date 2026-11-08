import hashlib
import os
from pathlib import Path

import torch
from loguru import logger

from config import BYTES_PER_GB


def format_bytes(size: float) -> str:
    n = size / BYTES_PER_GB
    return f"{n:.2f} GB"


def print_memory_stats(stage: str) -> None:
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping memory stats")
        return

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    current_allocated = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()

    logger.info(f"[{stage}] Memory Stats:")
    logger.info(f"  Current Allocated: {format_bytes(current_allocated)}")
    logger.info(f"  Current Reserved:  {format_bytes(current_reserved)}")
    logger.info(f"  Peak Allocated:    {format_bytes(allocated)}")
    logger.info(f"  Peak Reserved:     {format_bytes(reserved)}")


def compute_file_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


compute_file_md5 = compute_file_sha256


def load_manifest(manifest_file: Path) -> dict[str, str | None]:
    manifest = {}
    if not manifest_file.exists():
        return manifest

    with open(manifest_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) == 1:
                file_path = parts[0]
                manifest[file_path] = None
            elif len(parts) == 2:
                if len(parts[0]) == 32 and all(c in "0123456789abcdef" for c in parts[0].lower()):
                    md5_hash, file_path = parts
                else:
                    file_path, md5_hash = parts
                manifest[file_path] = md5_hash
            else:
                logger.warning(f"Invalid manifest format at line {line_num}: {line}")
                continue

    return manifest


def verify_file_integrity(
    base_dir: Path, manifest: dict[str, str | None], verify_checksums: bool = True
) -> tuple[bool, list[str], list[str]]:
    missing = []
    corrupted = []

    for rel_path, expected_md5 in manifest.items():
        file_path = base_dir / rel_path

        if not file_path.exists():
            missing.append(rel_path)
            continue

        if verify_checksums and expected_md5 is not None:
            try:
                actual_md5 = compute_file_md5(file_path)
                if actual_md5 != expected_md5:
                    corrupted.append(rel_path)
                    logger.debug(
                        f"Checksum mismatch for {rel_path}: expected {expected_md5}, got {actual_md5}"
                    )
            except Exception as e:
                logger.error(f"Failed to compute checksum for {rel_path}: {e}")
                corrupted.append(rel_path)

    all_valid = len(missing) == 0 and len(corrupted) == 0
    return all_valid, missing, corrupted


def ensure_model_weights(
    model_path: str,
    repo_id: str = "Tongyi-MAI/Z-Image-Turbo",
    revision: str | None = None,
    verify: bool = False,
    manifest_name: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    target_dir = Path(model_path)

    if manifest_name:
        manifest_path = Path(__file__).parent.parent / "config" / "manifests" / manifest_name
    else:
        model_name = target_dir.name.lower()
        config_manifest = (
            Path(__file__).parent.parent / "config" / "manifests" / f"{model_name}.txt"
        )

        if config_manifest.exists():
            manifest_path = config_manifest
        else:
            manifest_path = target_dir / "manifest.txt"

    manifest = load_manifest(manifest_path)

    if not manifest:
        logger.warning(f"Manifest file not found: {manifest_path}")
        logger.warning("Skipping file verification (assuming model exists)")
        if target_dir.exists():
            logger.info(f"Model directory exists: {target_dir}")
            return target_dir
        else:
            logger.warning(f"Model directory not found: {target_dir}")
            missing_files = ["entire model directory"]
            corrupted_files = []
    else:
        files_with_checksums = sum(1 for v in manifest.values() if v is not None)

        if verify and files_with_checksums == 0:
            logger.info("Verify requested but no checksums in manifest, only checking existence")
        elif verify and files_with_checksums > 0:
            logger.info(f"Verifying {files_with_checksums} file(s) with MD5 checksums...")

        all_valid, missing_files, corrupted_files = verify_file_integrity(
            target_dir, manifest, verify_checksums=verify
        )

        if all_valid:
            if verify and files_with_checksums > 0:
                logger.success(f"All files verified with MD5 checksums in {target_dir}")
            else:
                logger.info(f"All {len(manifest)} required files exist in {target_dir}")
            return target_dir

    if missing_files:
        logger.warning(f"Missing {len(missing_files)} file(s):")
        for f in missing_files[:10]:
            logger.warning(f"  - {f}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")

    if corrupted_files:
        logger.error(f"Corrupted {len(corrupted_files)} file(s) (checksum mismatch):")
        for f in corrupted_files[:10]:
            logger.error(f"  - {f}")
        if len(corrupted_files) > 10:
            logger.error(f"  ... and {len(corrupted_files) - 10} more")

    pin = revision or os.environ.get("HF_MODEL_REVISION")
    logger.info(f"\nAttempting to download from {repo_id}@{pin or 'HEAD'}...")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(  # nosec B615 - trusted repo from config
            repo_id=repo_id,
            revision=pin if pin else None,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.success("Download completed")
    except Exception as e:
        logger.warning(f"Download failed: {e}. Manually place weights in {target_dir.absolute()}")
        raise RuntimeError(f"Failed to download model weights: {e}") from e

    if manifest:
        all_valid, missing_after, corrupted_after = verify_file_integrity(
            target_dir, manifest, verify_checksums=verify
        )

        if not all_valid:
            error_msg = []
            if missing_after:
                error_msg.append(f"Still missing {len(missing_after)} file(s)")
            if corrupted_after:
                error_msg.append(f"Still corrupted {len(corrupted_after)} file(s)")

            raise FileNotFoundError(
                f"After download: {', '.join(error_msg)}. "
                f"Manually place files in {target_dir.absolute()}"
            )

    logger.success("All model weights validated successfully")
    return target_dir
