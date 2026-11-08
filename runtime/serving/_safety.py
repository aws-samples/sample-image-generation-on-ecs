from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class Limits:
    min_ram_gb: float = 10.0
    min_disk_gb: float = 20.0
    max_workdir_gb: float = 60.0
    interval_s: float = 10.0


_MONITOR_THREAD: threading.Thread | None = None
_MONITOR_STOP = threading.Event()


def _bytes_free(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return -1


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except OSError:
        pass
    return -1


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def _neuron_memory_usage() -> dict | None:
    try:
        out = subprocess.check_output(
            ["/opt/aws/neuron/bin/neuron-ls", "-j"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        import json as _json

        data = _json.loads(out)
        return data[0] if isinstance(data, list) and data else None
    except Exception:
        return None


def install_resource_monitor(
    log_path: Path | None = None,
    limits: Limits = Limits(),
    watch_dirs: list | None = None,
    watch_glob_dirs: list | None = None,
) -> None:
    global _MONITOR_THREAD
    if _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
        return

    watch_dirs = [Path(d) for d in (watch_dirs or [])]
    watch_glob_dirs = [(Path(p), g) for (p, g) in (watch_glob_dirs or [])]

    def _tick() -> None:
        samples = 0
        while not _MONITOR_STOP.is_set():
            samples += 1
            try:
                mem = _mem_available_bytes()
                disk = _bytes_free("/")
                mem_gb = mem / 1e9 if mem > 0 else -1
                disk_gb = disk / 1e9 if disk > 0 else -1
                wd_bytes = sum(_dir_size(d) for d in watch_dirs)
                wd_gb = wd_bytes / 1e9

                glob_bytes = 0
                for parent, pat in watch_glob_dirs:
                    for d in parent.glob(pat):
                        glob_bytes += _dir_size(d)
                glob_gb = glob_bytes / 1e9

                line = (
                    f"t={samples * limits.interval_s:.0f}s "
                    f"ram_avail={mem_gb:.1f}GB disk_free={disk_gb:.1f}GB "
                    f"workdir={wd_gb:.2f}GB stray_compile_dirs={glob_gb:.2f}GB"
                )
                logger.info(line)
                if log_path is not None:
                    try:
                        with open(log_path, "a") as fh:
                            fh.write(line + "\n")
                    except OSError:
                        pass

                if disk_gb > 0 and disk_gb < limits.min_disk_gb:
                    logger.error(
                        f"FREE DISK {disk_gb:.1f}GB < "
                        f"{limits.min_disk_gb}GB -> aborting to protect host"
                    )
                    os._exit(137)
                if wd_gb > limits.max_workdir_gb:
                    logger.error(f"WORKDIR {wd_gb:.1f}GB > {limits.max_workdir_gb}GB -> aborting")
                    os._exit(137)
                if glob_gb > limits.max_workdir_gb:
                    logger.error(
                        f"STRAY COMPILE DIRS {glob_gb:.1f}GB > "
                        f"{limits.max_workdir_gb}GB -> aborting"
                    )
                    os._exit(137)
                if mem_gb > 0 and mem_gb < limits.min_ram_gb:
                    logger.error(
                        f"FREE RAM {mem_gb:.1f}GB < "
                        f"{limits.min_ram_gb}GB -> aborting to prevent OOM"
                    )
                    os._exit(137)
            except Exception as e:
                logger.warning(f"tick error: {e}")

            _MONITOR_STOP.wait(limits.interval_s)

    _MONITOR_THREAD = threading.Thread(target=_tick, name="resource-monitor", daemon=True)
    _MONITOR_THREAD.start()
    logger.info(
        f"installed — ram>={limits.min_ram_gb}GB, "
        f"disk>={limits.min_disk_gb}GB, workdir<={limits.max_workdir_gb}GB, "
        f"every {limits.interval_s:.0f}s"
    )


def stop_resource_monitor() -> None:
    _MONITOR_STOP.set()


def cleanup_stale_neuronxcc_dirs(
    root: Path, max_age_min: float = 120.0, dry_run: bool = False
) -> int:
    now = time.time()
    freed = 0
    for item in root.glob("neuronxcc-*"):
        try:
            age_min = (now - item.stat().st_mtime) / 60
        except OSError:
            continue
        if age_min < max_age_min:
            continue
        size = _dir_size(item)
        logger.info(f"removing {item.name} ({size / 1e9:.2f}GB, age {age_min:.0f}min)")
        if not dry_run:
            try:
                shutil.rmtree(item)
                freed += size
            except OSError as e:
                logger.warning(f"could not remove {item}: {e}")
    return freed


class DiskGuard:
    def __init__(self, path: Path, max_gb: float = 40.0, interval_s: float = 15.0):
        self.path = Path(path)
        self.max_bytes = int(max_gb * 1e9)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            size = _dir_size(self.path)
            if size > self.max_bytes:
                logger.warning(
                    f"{self.path.name} = {size / 1e9:.2f}GB "
                    f"exceeds cap {self.max_bytes / 1e9:.2f}GB -> abort"
                )
                os._exit(137)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._th = threading.Thread(
            target=self._run, daemon=True, name=f"DiskGuard[{self.path.name}]"
        )
        self._th.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1)
