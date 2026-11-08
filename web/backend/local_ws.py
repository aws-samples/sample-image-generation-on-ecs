from __future__ import annotations

import asyncio
import contextlib
import struct
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

import inference_client
from config import settings
from gallery import gallery_manager
from mock_data import mock_generate_frames
from streaming import FrameBuilder, FrameParser, FrameType

router = APIRouter()


@router.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    request_id = str(uuid.uuid4())
    parser = FrameParser()

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        steps = data.get("steps", 8)
        seed = data.get("seed")
        resolution = data.get("resolution", 512)
        denoise_viz = data.get("denoise_viz", True)
        backend = data.get("backend", settings.default_backend)
        guidance = data.get("guidance", 0.0)
        negative_prompt = data.get("negative_prompt")

        if not prompt:
            await websocket.send_bytes(FrameBuilder.build_error("Prompt is required"))
            await websocket.close(code=4000, reason="Missing prompt")
            return

        if settings.local:
            await websocket.send_json(
                {
                    "type": "request_id",
                    "request_id": request_id,
                    "backend": "mock",
                    "endpoint": "local-mock",
                }
            )
            frames = mock_generate_frames(steps=steps, resolution=resolution)
            for frame_bytes in frames:
                frame_type = struct.unpack(">I", frame_bytes[:4])[0]
                if frame_type == FrameType.INTERMEDIATE and not denoise_viz:
                    continue
                await websocket.send_bytes(frame_bytes)
                await asyncio.sleep(0.3 if denoise_viz else 0.05)
            await websocket.close()
            return

        try:
            endpoint, _ = settings.pick_endpoint(resolution=resolution, backend=backend)
        except ValueError as exc:
            await websocket.send_bytes(FrameBuilder.build_error(str(exc)))
            await websocket.close(code=4000, reason="bad backend/resolution")
            return

        await websocket.send_json(
            {
                "type": "request_id",
                "request_id": request_id,
                "backend": backend,
                "endpoint": endpoint,
            }
        )

        frame_count = 0
        final_image_bytes: bytes | None = None

        payload = {
            "prompt": prompt,
            "steps": steps,
            "seed": seed,
            "resolution": resolution,
            "denoise_viz": denoise_viz,
            "guidance": guidance,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        for chunk in inference_client.invoke_streaming(resolution, payload):
            for frame in parser.feed(chunk):
                frame_bytes = FrameBuilder.build_frame(
                    frame.frame_type, frame.step_index, frame.payload, frame.metadata
                )
                await websocket.send_bytes(frame_bytes)
                frame_count += 1
                if frame.frame_type == FrameType.FINAL:
                    final_image_bytes = frame.payload

        await websocket.send_bytes(
            FrameBuilder.build_complete(
                {
                    "request_id": request_id,
                    "frames_received": frame_count,
                    "backend": backend,
                    "endpoint": endpoint,
                }
            )
        )

        if final_image_bytes:
            try:
                await gallery_manager.save_image(
                    request_id=request_id,
                    image_bytes=final_image_bytes,
                    prompt=prompt,
                    resolution=resolution,
                    steps=steps,
                    seed=seed,
                )
            except Exception as e:
                logger.error(f"failed to save to gallery: {e}")

    except WebSocketDisconnect:
        logger.info(f"client disconnected during gen {request_id}")
    except Exception as e:
        logger.error(f"gen error {request_id}: {e}")
        with contextlib.suppress(Exception):
            await websocket.send_bytes(FrameBuilder.build_error(str(e)))
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
