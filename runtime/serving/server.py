from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from loguru import logger

from lib.paths import (
    COMPILED_MODELS_DIR,
    EFS_ARTIFACTS_PATH,
    NEURON_NEFFS,
    WEIGHTS_DIR,
)

_PORT = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))
_RESOLUTION = int(os.environ.get("ZIMAGE_DEFAULT_RESOLUTION", "512"))
_BACKEND = os.environ.get("ZIMAGE_BACKEND", "neuron").lower()
_ARTIFACTS_DIR = Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", EFS_ARTIFACTS_PATH))

_pipeline = None
_load_error: str | None = None

_inference_lock = threading.Lock()


def _validate_artifacts() -> None:
    weights_dir = _ARTIFACTS_DIR / WEIGHTS_DIR
    if not weights_dir.is_dir() or not any(weights_dir.iterdir()):
        raise FileNotFoundError(
            f"Model weights not found at {weights_dir}. "
            "Ensure EFS is mounted and the compile pipeline has run."
        )

    if _BACKEND == "neuron":
        compiled_dir = _ARTIFACTS_DIR / COMPILED_MODELS_DIR
        missing = []
        for neff in NEURON_NEFFS.get(_RESOLUTION, []):
            if not (compiled_dir / neff).is_file():
                missing.append(neff)
        if missing:
            raise FileNotFoundError(
                f"Missing NEFFs in {compiled_dir}: {missing}. "
                "Run the compile state machine to generate them."
            )

    logger.info(f"artifacts validated: {_ARTIFACTS_DIR}")


def _construct_pipeline():
    if _BACKEND == "gpu":
        allowed = tuple(
            int(x)
            for x in os.environ.get("ZIMAGE_ALLOWED_RESOLUTIONS", str(_RESOLUTION)).split(",")
            if x.strip()
        )
        from serving.production.pipeline_gpu import ZImagePipelineGpu

        pipe = ZImagePipelineGpu.from_default_artifacts(allowed_resolutions=allowed)
        pipe.warmup()
        return pipe
    if _RESOLUTION == 512:
        from serving.production.pipeline_512 import ZImagePipeline

        encoder_mode = os.environ.get("ZIMAGE_ENCODER_MODE", "subprocess")
        return ZImagePipeline.from_default_artifacts(encoder_mode=encoder_mode)
    if _RESOLUTION == 768:
        from serving.production.pipeline_768 import ZImagePipeline768

        pipe = ZImagePipeline768.from_default_artifacts()
        pipe.warmup()
        return pipe
    if _RESOLUTION == 1024:
        from serving.production.pipeline_1024 import ZImagePipeline1024

        pipe = ZImagePipeline1024.from_default_artifacts()
        pipe.warmup()
        return pipe
    raise ValueError(f"unsupported ZIMAGE_DEFAULT_RESOLUTION={_RESOLUTION}")


def _startup() -> None:
    global _pipeline, _load_error
    try:
        t0 = time.perf_counter()
        logger.info("startup: validating artifacts...")
        _validate_artifacts()
        logger.info(f"startup: loading pipeline backend={_BACKEND} resolution={_RESOLUTION}")
        _pipeline = _construct_pipeline()
        logger.info(f"startup: ready in {time.perf_counter() - t0:.1f}s")
    except RuntimeError as exc:
        if "PJRT_Client_Create" in str(exc) or "nrt_init" in str(exc):
            logger.error(
                f"FATAL: Neuron driver incompatible — exiting to force new instance. {exc}"
            )
            os._exit(1)
        logger.exception("startup FAILED")
        _load_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        logger.exception("startup FAILED")
        _load_error = f"{type(exc).__name__}: {exc}"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info(fmt % args)

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != "/ping":
            self._json(404, {"error": "not found"})
            return
        if _load_error:
            self._json(500, {"status": "failed", "error": _load_error})
            return
        if _pipeline is None:
            self._json(503, {"status": "loading"})
            return
        if hasattr(_pipeline, "_encoder_client"):
            client = _pipeline._encoder_client
            if client is not None and not client.is_alive():
                from serving.encoder_client import EncoderClient as _EC

                try:
                    _pipeline._encoder_client = _EC(_pipeline._paths.encoder_server)
                    _pipeline._encoder_path = "subprocess"
                    logger.info("encoder subprocess restarted via /ping")
                except Exception:
                    self._json(500, {"status": "degraded", "error": "encoder subprocess dead"})
                    return
        self._json(200, {"status": "ok"})

    def do_POST(self):
        if self.path != "/invocations":
            self._json(404, {"error": "not found"})
            return
        if _pipeline is None:
            self._json(503, {"error": "model still loading"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            MAX_BODY_SIZE = 65536
            if length > MAX_BODY_SIZE:
                self._json(413, {"error": f"Request body too large (max {MAX_BODY_SIZE} bytes)"})
                return
            raw = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

            if "inputs" in raw and "prompt" not in raw:
                raw["prompt"] = raw.pop("inputs")

            prompt = raw.get("prompt", "")
            if not prompt:
                self._json(400, {"error": "prompt is required"})
                return
            seed = raw.get("seed", 42)
            denoise_viz = raw.get("denoise_viz", True)

            accept = self.headers.get("Accept", "")
            use_streaming = (
                denoise_viz
                and accept == "application/octet-stream"
                and hasattr(_pipeline, "generate_streaming")
            )

            logger.info(
                "invocation start: mode={} resolution={} denoise_viz={} seed={}",
                "streaming" if use_streaming else "json",
                _RESOLUTION,
                denoise_viz,
                seed,
            )

            with _inference_lock:
                if use_streaming:
                    self._handle_streaming(prompt, seed, raw)
                else:
                    self._handle_json(prompt, seed, raw)

        except Exception as exc:
            logger.exception("invocations failed")
            self._json(500, {"error": str(exc), "type": type(exc).__name__})

    def _handle_json(self, prompt: str, seed: int, req: dict) -> None:
        kwargs: dict = {"seed": seed}
        if _BACKEND == "gpu":
            res = int(req.get("resolution", 0))
            if res:
                kwargs.setdefault("height", res)
                kwargs.setdefault("width", res)
            for k in ("height", "width", "num_inference_steps"):
                if k in req:
                    kwargs[k] = int(req[k])

        t0 = time.perf_counter()
        result = _pipeline.generate(prompt, **kwargs)
        image = result[0] if isinstance(result, tuple) else result
        t_gen = time.perf_counter() - t0

        t_enc = time.perf_counter()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        t_enc = time.perf_counter() - t_enc

        total = time.perf_counter() - t0
        logger.info(
            "invocation done: generate={:.2f}s encode={:.3f}s total={:.2f}s size={}KB",
            t_gen,
            t_enc,
            total,
            len(buf.getvalue()) // 1024,
        )
        self._json(
            200,
            {
                "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "format": "png",
                "resolution": _RESOLUTION,
                "elapsed_s": round(total, 2),
            },
        )

    def _handle_streaming(self, prompt: str, seed: int, req: dict) -> None:
        import struct

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        FT_INTERMEDIATE = 1
        FT_FINAL = 2

        def _write_frame(frame_type: int, step: int, payload: bytes) -> None:
            header = struct.pack(">III", frame_type, step, len(payload))
            chunk = header + payload
            # HTTP chunked encoding: hex length + CRLF + data + CRLF
            self.wfile.write(f"{len(chunk):x}\r\n".encode())
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        kwargs: dict = {"seed": seed}
        if _BACKEND == "gpu":
            for k in ("height", "width", "num_inference_steps"):
                if k in req:
                    kwargs[k] = int(req[k])

        try:
            t0 = time.perf_counter()
            t_first_frame = None
            frame_count = 0
            for frame_type_str, step, image, _timing in _pipeline.generate_streaming(
                prompt, denoise_viz=True, **kwargs
            ):
                if frame_type_str == "intermediate":
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=40)
                    if t_first_frame is None:
                        t_first_frame = time.perf_counter() - t0
                        logger.info("first intermediate frame at {:.2f}s", t_first_frame)
                    _write_frame(FT_INTERMEDIATE, step, buf.getvalue())
                    frame_count += 1
                elif frame_type_str == "final":
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    _write_frame(FT_FINAL, step, buf.getvalue())

            total = time.perf_counter() - t0
            logger.info(
                "streaming done: first_frame={:.2f}s total={:.2f}s frames={}",
                t_first_frame or 0,
                total,
                frame_count,
            )
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except BrokenPipeError:
            logger.warning("streaming client disconnected (broken pipe)")
        except Exception as exc:
            logger.exception("streaming generation failed")
            try:
                err_payload = str(exc).encode("utf-8")
                err_header = struct.pack(">III", 4, 0, len(err_payload))
                chunk = err_header + err_payload
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk)
            except BrokenPipeError:
                pass
            self.wfile.write(b"\r\n0\r\n\r\n")
            self.wfile.flush()


def main() -> None:
    logger.info(f"starting server on :{_PORT} (resolution={_RESOLUTION}, backend={_BACKEND})")
    threading.Thread(target=_startup, name="startup", daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", _PORT), _Handler).serve_forever()  # nosec B104


if __name__ == "__main__":
    main()
