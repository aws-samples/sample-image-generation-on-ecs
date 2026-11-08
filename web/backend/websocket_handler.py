from __future__ import annotations

import base64
import json
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import boto3
import urllib3
from botocore.config import Config as BotoConfig
from loguru import logger

GALLERY_BUCKET = os.environ.get("ZIMAGE_GALLERY_BUCKET", "")
GALLERY_TABLE = os.environ.get("ZIMAGE_GALLERY_TABLE", "")
REGION = os.environ["ZIMAGE_REGION"]
PREFIX = os.environ.get("ZIMAGE_PREFIX", "dev-zimg")
INFERENCE_ALB_URL = os.environ.get("ZIMAGE_INFERENCE_ALB_URL", "")

s3_client = boto3.client("s3", region_name=REGION, config=BotoConfig(signature_version="s3v4"))
dynamodb_resource = boto3.resource("dynamodb", region_name=REGION)
_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10, read=120))


def _get_apigw_client(domain: str, stage: str):
    endpoint_url = f"https://{domain}/{stage}"
    return boto3.client("apigatewaymanagementapi", endpoint_url=endpoint_url, region_name=REGION)


_WS_FRAME_LIMIT = 128_000


def _post_to_connection(apigw, connection_id: str, data: dict) -> None:
    try:
        payload = json.dumps(data).encode()
        if len(payload) > _WS_FRAME_LIMIT:
            logger.warning(f"Payload exceeds 128KB ({len(payload)} bytes), dropping")
            return
        apigw.post_to_connection(ConnectionId=connection_id, Data=payload)
    except apigw.exceptions.GoneException:
        logger.warning(f"Connection {connection_id} is gone")
    except Exception as e:
        logger.warning(f"PostToConnection failed: {e}")


def _resolve_endpoint(resolution: int, backend: str) -> str:
    if backend == "gpu":
        return "ecs:gpu"
    return f"ecs:{resolution}"


def handler(event, context):
    if event.get("warmup"):
        return {"statusCode": 200, "body": "warm"}
    route_key = event.get("requestContext", {}).get("routeKey", "$default")
    connection_id = event["requestContext"]["connectionId"]
    domain = event["requestContext"]["domainName"]
    stage = event["requestContext"]["stage"]

    logger.info(f"WS handler: route={route_key} conn={connection_id}")

    if route_key == "$connect":
        logger.info("$connect — authorized, accepting")
        return {"statusCode": 200}

    if route_key == "$disconnect":
        logger.info("$disconnect")
        return {"statusCode": 200}

    apigw = _get_apigw_client(domain, stage)

    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        _post_to_connection(apigw, connection_id, {"type": "error", "message": "Invalid JSON"})
        return {"statusCode": 400}

    prompt = body.get("prompt", "")
    if not prompt:
        _post_to_connection(
            apigw, connection_id, {"type": "error", "message": "Prompt is required"}
        )
        return {"statusCode": 400}

    request_id = str(uuid.uuid4())
    try:
        width = max(256, min(2048, int(body.get("width", 512))))
        height = max(256, min(2048, int(body.get("height", 512))))
        steps = max(1, min(50, int(body.get("steps", 8))))
        guidance = max(0.0, min(20.0, float(body.get("guidance", 0.0))))
    except (ValueError, TypeError):
        _post_to_connection(
            apigw, connection_id, {"type": "error", "message": "Invalid numeric parameter"}
        )
        return {"statusCode": 400}
    if width % 8 != 0 or height % 8 != 0:
        _post_to_connection(
            apigw, connection_id, {"type": "error", "message": "Dimensions must be divisible by 8"}
        )
        return {"statusCode": 400}
    seed = body.get("seed")
    backend = body.get("backend", os.environ.get("ZIMAGE_DEFAULT_BACKEND", "gpu"))
    negative_prompt = body.get("negative_prompt")
    denoise_viz = body.get("denoise_viz", True)

    endpoint = _resolve_endpoint(width, backend)
    target_header = "gpu" if backend == "gpu" else str(width)
    t_start = time.perf_counter()

    _post_to_connection(
        apigw,
        connection_id,
        {"type": "request_id", "request_id": request_id, "backend": backend, "endpoint": endpoint},
    )
    logger.info(
        f"[{request_id[:8]}] start endpoint={endpoint} {width}x{height} denoise={denoise_viz}"
    )

    inference_payload = {
        "prompt": prompt,
        "steps": steps,
        "height": height,
        "width": width,
        "denoise_viz": denoise_viz,
        "guidance": guidance,
    }
    if seed is not None:
        inference_payload["seed"] = seed
    if negative_prompt:
        inference_payload["negative_prompt"] = negative_prompt

    def _friendly_error(e: Exception) -> str:
        err_str = str(e)
        if "Read timed out" in err_str or "ReadTimeout" in err_str or "timeout" in err_str.lower():
            return "The model is still warming up. Please try again in a few minutes."
        if "ConnectionRefused" in err_str or "refused" in err_str.lower():
            return "Cannot reach the inference service. It may be starting up."
        if "NewConnectionError" in err_str or "Failed to establish" in err_str:
            return "Cannot connect to inference. The service may be starting up."
        if "server error (500)" in err_str or "status 500" in err_str:
            return "The model encountered an internal error. Try again or switch resolution."
        return "Generation failed. Please try again."

    final_image_bytes: bytes | None = None
    resp_body: dict = {}
    frame_count = 0

    if denoise_viz:
        from streaming import FrameParser, FrameType

        t_invoke = time.perf_counter()
        try:
            response = _http.request(
                "POST",
                f"{INFERENCE_ALB_URL}/invocations",
                body=json.dumps(inference_payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/octet-stream",
                    "X-Target-Resolution": target_header,
                },
                preload_content=False,
            )
            if response.status != 200:
                raise RuntimeError(f"Inference returned {response.status}: {response.data[:200]}")
        except Exception as e:
            logger.error(f"ECS inference invoke failed: {e}")
            _post_to_connection(
                apigw, connection_id, {"type": "error", "message": _friendly_error(e)}
            )
            return {"statusCode": 500}
        logger.info(f"[{request_id[:8]}] invoke={time.perf_counter() - t_invoke:.3f}s")

        parser = FrameParser()
        t_first_frame = None
        frame_count = 0
        try:
            for chunk in response.stream(4096):
                if not chunk:
                    continue
                for frame in parser.feed(chunk):
                    if frame.frame_type == FrameType.INTERMEDIATE:
                        if t_first_frame is None:
                            t_first_frame = time.perf_counter() - t_start
                            logger.info(f"[{request_id[:8]}] first_frame={t_first_frame:.2f}s")
                        t_post = time.perf_counter()
                        img_bytes = frame.payload
                        if img_bytes[:4] == b"\x89PNG" or len(img_bytes) > 80_000:
                            import io

                            from PIL import Image

                            img = Image.open(io.BytesIO(img_bytes))
                            if img.width > 384 or img.height > 384:
                                img.thumbnail((384, 384), Image.LANCZOS)
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=35)
                            img_bytes = buf.getvalue()
                        jpeg_b64 = base64.b64encode(img_bytes).decode()
                        data_url = f"data:image/jpeg;base64,{jpeg_b64}"
                        _post_to_connection(
                            apigw,
                            connection_id,
                            {
                                "type": "frame",
                                "frame_type": "intermediate",
                                "step": frame.step,
                                "url": data_url,
                            },
                        )
                        frame_count += 1
                        if frame_count <= 2:
                            logger.info(
                                f"[{request_id[:8]}] frame {frame.step} "
                                f"post={time.perf_counter() - t_post:.3f}s "
                                f"size={len(frame.payload) // 1024}KB"
                            )
                    elif frame.frame_type == FrameType.FINAL:
                        t_final = time.perf_counter()
                        final_image_bytes = frame.payload
                        final_key = f"images/{request_id}/final.png"
                        s3_client.put_object(
                            Bucket=GALLERY_BUCKET,
                            Key=final_key,
                            Body=final_image_bytes,
                            ContentType="image/png",
                        )
                        url = s3_client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": GALLERY_BUCKET, "Key": final_key},
                            ExpiresIn=3600,
                        )
                        _post_to_connection(
                            apigw,
                            connection_id,
                            {
                                "type": "frame",
                                "frame_type": "final",
                                "step": frame.step,
                                "url": url,
                            },
                        )
                        logger.info(
                            f"[{request_id[:8]}] final s3+post={time.perf_counter() - t_final:.3f}s"
                        )
                    elif frame.frame_type == FrameType.ERROR:
                        _post_to_connection(
                            apigw,
                            connection_id,
                            {
                                "type": "error",
                                "message": frame.payload.decode("utf-8", errors="replace"),
                            },
                        )
                        return {"statusCode": 500}
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            _post_to_connection(
                apigw, connection_id, {"type": "error", "message": _friendly_error(e)}
            )
            return {"statusCode": 500}
        finally:
            response.release_conn()
    else:
        try:
            response = _http.request(
                "POST",
                f"{INFERENCE_ALB_URL}/invocations",
                body=json.dumps(inference_payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Target-Resolution": target_header,
                },
            )
            if response.status != 200:
                raise RuntimeError(f"Inference returned {response.status}")
        except Exception as e:
            logger.error(f"ECS inference invoke failed: {e}")
            _post_to_connection(
                apigw, connection_id, {"type": "error", "message": _friendly_error(e)}
            )
            return {"statusCode": 500}

        try:
            resp_body = json.loads(response.data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse endpoint response: {e}")
            _post_to_connection(
                apigw, connection_id, {"type": "error", "message": "Invalid response"}
            )
            return {"statusCode": 500}

        if "error" in resp_body:
            _post_to_connection(
                apigw, connection_id, {"type": "error", "message": resp_body["error"]}
            )
            return {"statusCode": 500}

        image_b64 = resp_body.get("image_b64", "")
        if image_b64:
            final_image_bytes = base64.b64decode(image_b64)
            final_key = f"images/{request_id}/final.png"
            s3_client.put_object(
                Bucket=GALLERY_BUCKET,
                Key=final_key,
                Body=final_image_bytes,
                ContentType="image/png",
            )
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": GALLERY_BUCKET, "Key": final_key},
                ExpiresIn=3600,
            )
            _post_to_connection(
                apigw,
                connection_id,
                {
                    "type": "frame",
                    "frame_type": "final",
                    "step": width,
                    "url": url,
                },
            )

    if final_image_bytes and GALLERY_TABLE:
        now = datetime.now(UTC).isoformat()
        item: dict[str, Any] = {
            "user_id": "default",
            "created_at": now,
            "id": request_id,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "backend": backend,
            "denoise_viz": denoise_viz,
            "deleted": False,
            "image_url": f"images/{request_id}/final.png",
        }
        if seed is not None:
            item["seed"] = seed
        elapsed = resp_body.get("elapsed_s")
        if elapsed is not None:
            item["elapsed_s"] = str(elapsed)

        try:
            table = dynamodb_resource.Table(GALLERY_TABLE)
            table.put_item(Item=item)
        except Exception as e:
            logger.error(f"Failed to save metadata to DynamoDB: {e}")

    t_total = time.perf_counter() - t_start
    logger.info(
        f"[{request_id[:8]}] done total={t_total:.2f}s frames={frame_count if denoise_viz else 0}"
    )
    _post_to_connection(
        apigw,
        connection_id,
        {
            "type": "complete",
            "request_id": request_id,
            "elapsed_s": resp_body.get("elapsed_s"),
            "lambda_elapsed_s": round(t_total, 2),
            "backend": backend,
            "endpoint": endpoint,
        },
    )

    return {"statusCode": 200}
