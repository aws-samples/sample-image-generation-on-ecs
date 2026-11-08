from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

os.environ["NEURON_RT_VISIBLE_CORES"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_ORIG_STDOUT_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_data_stream = os.fdopen(_ORIG_STDOUT_FD, "wb", buffering=0)

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="DEBUG")

from lib.paths import (
    COMPILED_MODELS_DIR,
    DEFAULT_ARTIFACTS_DIR,
    NEFF_ENCODER_256,
    NEFF_VAE_768,
    NEFF_VAE_1024,
)

ARTIFACTS_DIR = Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
TRACED_ENCODER_PT = ARTIFACTS_DIR / COMPILED_MODELS_DIR / NEFF_ENCODER_256
TRACED_VAE_1024_PT = ARTIFACTS_DIR / COMPILED_MODELS_DIR / NEFF_VAE_1024
TRACED_VAE_768_PT = ARTIFACTS_DIR / COMPILED_MODELS_DIR / NEFF_VAE_768
MAX_SEQ_LEN_ENCODER = 256
OUT_SEQ_LEN = 512


def _read_frame(stream) -> tuple[dict, bytes] | None:
    raw = stream.read(4)
    if len(raw) < 4:
        return None
    json_len = struct.unpack(">I", raw)[0]
    json_bytes = stream.read(json_len)
    if len(json_bytes) < json_len:
        return None
    raw = stream.read(4)
    if len(raw) < 4:
        return None
    payload_len = struct.unpack(">I", raw)[0]
    payload = stream.read(payload_len) if payload_len > 0 else b""
    if len(payload) < payload_len:
        return None
    return json.loads(json_bytes), payload


def _write_frame(stream, json_data: dict, payload: bytes = b"") -> None:
    json_bytes = json.dumps(json_data).encode()
    stream.write(struct.pack(">I", len(json_bytes)))
    stream.write(json_bytes)
    stream.write(struct.pack(">I", len(payload)))
    if payload:
        stream.write(payload)
    stream.flush()


def main() -> int:
    logger.info("[enc-srv] starting, NEURON_RT_VISIBLE_CORES=1")

    import io

    import torch
    from transformers import AutoTokenizer

    if not TRACED_ENCODER_PT.exists():
        logger.error(f"[enc-srv] missing {TRACED_ENCODER_PT}")
        return 1

    tok = AutoTokenizer.from_pretrained(  # nosec B615 - local path
        str(ARTIFACTS_DIR / "ckpts" / "Z-Image-Turbo" / "tokenizer"),
        trust_remote_code=True,
    )

    logger.info(
        f"[enc-srv] loading traced encoder ({TRACED_ENCODER_PT.stat().st_size / 1e9:.1f} GB)…"
    )
    enc = torch.jit.load(str(TRACED_ENCODER_PT)).eval()

    dummy_ids = torch.zeros(1, MAX_SEQ_LEN_ENCODER, dtype=torch.long)
    dummy_mask = torch.zeros(1, MAX_SEQ_LEN_ENCODER, dtype=torch.long)
    with torch.no_grad():
        _ = enc(dummy_ids, dummy_mask)
    logger.info("[enc-srv] encoder warm, ready")

    vae_neff = None
    for vae_path in (TRACED_VAE_1024_PT, TRACED_VAE_768_PT):
        if vae_path.exists():
            logger.info(f"[enc-srv] loading VAE ({vae_path.name})…")
            vae_neff = torch.jit.load(str(vae_path)).eval()
            logger.info("[enc-srv] VAE loaded on Core 1")
            break

    _data_stream.write(b"R")
    _data_stream.flush()

    stdin_bin = sys.stdin.buffer
    stdout_bin = _data_stream

    while True:
        try:
            result = _read_frame(stdin_bin)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[enc-srv] bad frame: {e}")
            _write_frame(stdout_bin, {"error": str(e)})
            continue
        if result is None:
            break
        request, payload = result
        try:
            op = request.get("op", "")

            if op == "encode":
                prompt = request["prompt"]
                formatted = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                inputs = tok(
                    [formatted],
                    padding="max_length",
                    max_length=MAX_SEQ_LEN_ENCODER,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    emb = enc(inputs.input_ids, inputs.attention_mask.long())
                out = torch.zeros((1, OUT_SEQ_LEN, emb.shape[-1]), dtype=emb.dtype)
                valid_len = int((inputs.attention_mask[0] == 1).sum().item())
                out[:, : min(MAX_SEQ_LEN_ENCODER, OUT_SEQ_LEN), :] = emb

                buf = io.BytesIO()
                torch.save(out.cpu().contiguous(), buf)
                _write_frame(
                    stdout_bin,
                    {"n_tokens": valid_len},
                    buf.getvalue(),
                )

            elif op == "vae_decode":
                latent = torch.load(
                    io.BytesIO(payload),
                    map_location="cpu",
                    weights_only=True,
                )
                if vae_neff is not None:
                    with torch.no_grad():
                        img = vae_neff(latent)
                    buf = io.BytesIO()
                    torch.save(img.cpu().contiguous(), buf)
                    _write_frame(stdout_bin, {"ok": True}, buf.getvalue())
                else:
                    _write_frame(stdout_bin, {"error": "no VAE loaded"})

            else:
                _write_frame(stdout_bin, {"error": "unknown request type"})

        except Exception as e:
            logger.opt(exception=True).error(f"[enc-srv] request failed: {e}")
            _write_frame(stdout_bin, {"error": str(e)})

    logger.info("[enc-srv] stdin closed, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
