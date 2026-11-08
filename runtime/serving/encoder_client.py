from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

from loguru import logger


class EncoderClient:
    _PY_CANDIDATES = (
        "/opt/conda/bin/python3",
        "python3",
    )

    def __init__(self, server_script: Path):
        env = dict(os.environ)
        env["NEURON_RT_VISIBLE_CORES"] = "1"
        env.setdefault("PYTHONPATH", "/opt/ml/code")
        env.pop("PJRT_DEVICE", None)
        env.pop("NEURON_CC_FLAGS", None)
        env["NEURON_RT_EXEC_TIMEOUT"] = "86400"

        py = self._resolve_python()
        self.proc = subprocess.Popen(
            [py, str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
            bufsize=0,
        )
        rb = self._read_exact(1)
        if rb != b"R":
            raise RuntimeError(
                f"encoder server didn't emit R (got {rb!r}). "
                f"Check preceding [enc-srv] stderr lines for details."
            )

    @classmethod
    def _resolve_python(cls) -> str:
        for c in cls._PY_CANDIDATES:
            if c == "python3":
                resolved = shutil.which("python3")
                if resolved:
                    return resolved
            elif Path(c).exists():
                return c
        raise RuntimeError(
            "could not find a python interpreter to launch encoder subprocess; "
            f"tried: {cls._PY_CANDIDATES}"
        )

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                raise RuntimeError(
                    f"encoder EOF after {n - remaining}/{n} bytes "
                    f"(stderr was inherited; check preceding container logs)."
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _write_frame(self, json_data: dict, payload: bytes = b"") -> None:
        json_bytes = json.dumps(json_data).encode()
        self.proc.stdin.write(struct.pack(">I", len(json_bytes)))
        self.proc.stdin.write(json_bytes)
        self.proc.stdin.write(struct.pack(">I", len(payload)))
        if payload:
            self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def _read_frame(self) -> tuple[dict, bytes]:
        json_len = struct.unpack(">I", self._read_exact(4))[0]
        json_bytes = self._read_exact(json_len)
        payload_len = struct.unpack(">I", self._read_exact(4))[0]
        payload = self._read_exact(payload_len) if payload_len > 0 else b""
        return json.loads(json_bytes), payload

    def encode(self, prompt: str, seed: int = 0) -> dict:
        self._write_frame({"op": "encode", "prompt": prompt, "seed": seed})
        resp, payload = self._read_frame()
        if "error" in resp:
            raise RuntimeError(f"encoder error: {resp['error']}")
        resp["embeds_pt_bytes"] = payload
        return resp

    def decode_vae(self, latent_tensor):
        import io

        import torch

        buf = io.BytesIO()
        torch.save(latent_tensor.cpu().contiguous(), buf)
        self._write_frame({"op": "vae_decode"}, buf.getvalue())
        resp, payload = self._read_frame()
        if "error" in resp:
            raise RuntimeError(f"Core 1 VAE decode error: {resp['error']}")
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception as exc:
            logger.debug(f"suppressed: {exc}")
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


__all__ = ["EncoderClient"]
