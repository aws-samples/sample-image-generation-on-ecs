from __future__ import annotations

import io
import random
import struct
from datetime import UTC, datetime, timedelta


def mock_endpoints() -> list[dict]:
    return [
        {
            "name": "dev-zimg-endpoint-512",
            "status": "InService",
            "instance_type": "ml.inf2.8xlarge",
            "current_instance_count": 1,
            "desired_instance_count": 1,
        },
        {
            "name": "dev-zimg-endpoint-768",
            "status": "InService",
            "instance_type": "ml.inf2.8xlarge",
            "current_instance_count": 1,
            "desired_instance_count": 2,
        },
        {
            "name": "dev-zimg-endpoint-1024",
            "status": "Creating",
            "instance_type": "ml.inf2.8xlarge",
            "current_instance_count": 0,
            "desired_instance_count": 1,
        },
        {
            "name": "dev-zimg-endpoint-gpu",
            "status": "InService",
            "instance_type": "ml.g5.xlarge",
            "current_instance_count": 1,
            "desired_instance_count": 1,
        },
    ]


def mock_metrics() -> dict:
    now = datetime.now(UTC)
    timestamps = [(now - timedelta(minutes=i)).isoformat() for i in range(60, 0, -1)]
    return {
        "invocations": [random.uniform(0, 10) for _ in range(60)],
        "latency_p99_ms": [random.uniform(3000, 20000) for _ in range(60)],
        "errors_4xx": [random.uniform(0, 2) for _ in range(60)],
        "errors_5xx": [0.0] * 58 + [random.uniform(0, 1) for _ in range(2)],
        "timestamps": timestamps,
    }


def mock_metrics_timeseries(hours: int = 6) -> dict:
    now = datetime.now(UTC)
    num_points = (hours * 60) // 5
    timestamps = [(now - timedelta(minutes=i * 5)).isoformat() for i in range(num_points, 0, -1)]
    return {
        "latency": {
            "timestamps": timestamps,
            "values": [random.uniform(3000, 18000) for _ in range(num_points)],
        },
        "invocations": {
            "timestamps": timestamps,
            "values": [random.uniform(0, 15) for _ in range(num_points)],
        },
        "errors5xx": {
            "timestamps": timestamps,
            "values": [
                0.0 if random.random() > 0.1 else random.uniform(0, 3) for _ in range(num_points)
            ],
        },
    }


def mock_alarms() -> list[dict]:
    return [
        {
            "name": "dev-zimg-endpoint-512-5xx",
            "state": "OK",
            "endpoint": "dev-zimg-endpoint-512",
            "metric": "Invocation5XXErrors",
        },
        {
            "name": "dev-zimg-endpoint-512-latency",
            "state": "OK",
            "endpoint": "dev-zimg-endpoint-512",
            "metric": "ModelLatencyP99",
        },
        {
            "name": "dev-zimg-endpoint-768-5xx",
            "state": "OK",
            "endpoint": "dev-zimg-endpoint-768",
            "metric": "Invocation5XXErrors",
        },
        {
            "name": "dev-zimg-endpoint-1024-5xx",
            "state": "ALARM",
            "endpoint": "dev-zimg-endpoint-1024",
            "metric": "Invocation5XXErrors",
        },
        {
            "name": "dev-zimg-endpoint-gpu-latency",
            "state": "INSUFFICIENT_DATA",
            "endpoint": "dev-zimg-endpoint-gpu",
            "metric": "ModelLatencyP99",
        },
    ]


def mock_pipeline() -> list[dict]:
    return [
        {"stage": "Source", "status": "Succeeded"},
        {"stage": "QualityGate", "status": "Succeeded"},
        {"stage": "Build", "status": "Succeeded"},
        {"stage": "Deploy", "status": "Succeeded"},
        {"stage": "CompileModels", "status": "InProgress"},
        {"stage": "DeployEndpoints", "status": None},
        {"stage": "Promote", "status": None},
    ]


def mock_sfn_executions() -> list[dict]:
    now = datetime.now(UTC)
    return [
        {
            "execution_arn": "arn:aws:states:us-east-1:123456789:execution:dev-zimg-compile-sfn:local-001",
            "name": "local-001",
            "status": "RUNNING",
            "start_date": (now - timedelta(minutes=12)).isoformat(),
            "stop_date": None,
        },
        {
            "execution_arn": "arn:aws:states:us-east-1:123456789:execution:dev-zimg-compile-sfn:local-000",
            "name": "local-000",
            "status": "SUCCEEDED",
            "start_date": (now - timedelta(hours=2)).isoformat(),
            "stop_date": (now - timedelta(hours=1, minutes=40)).isoformat(),
        },
    ]


def mock_instances() -> list[dict]:
    return [
        {
            "name": "dev-zimg-inference-512",
            "resolution": "512",
            "running_count": 2,
            "desired_count": 2,
            "avg_utilization": round(random.uniform(20, 65), 1),
            "targets": [
                {
                    "id": "10.0.1.42",
                    "availability_zone": "us-east-1a",
                    "health_state": "healthy",
                    "reason": None,
                    "active_connections": round(random.uniform(0, 1), 2),
                },
                {
                    "id": "10.0.2.18",
                    "availability_zone": "us-east-1b",
                    "health_state": "healthy",
                    "reason": None,
                    "active_connections": round(random.uniform(0, 1), 2),
                },
            ],
        },
        {
            "name": "dev-zimg-inference-768",
            "resolution": "768",
            "running_count": 1,
            "desired_count": 2,
            "avg_utilization": round(random.uniform(40, 80), 1),
            "targets": [
                {
                    "id": "10.0.1.87",
                    "availability_zone": "us-east-1a",
                    "health_state": "healthy",
                    "reason": None,
                    "active_connections": round(random.uniform(0, 1), 2),
                },
                {
                    "id": "10.0.2.55",
                    "availability_zone": "us-east-1b",
                    "health_state": "draining",
                    "reason": "Target.DeregistrationInProgress",
                    "active_connections": None,
                },
            ],
        },
        {
            "name": "dev-zimg-inference-1024",
            "resolution": "1024",
            "running_count": 1,
            "desired_count": 1,
            "avg_utilization": round(random.uniform(10, 40), 1),
            "targets": [
                {
                    "id": "10.0.1.103",
                    "availability_zone": "us-east-1a",
                    "health_state": "healthy",
                    "reason": None,
                    "active_connections": round(random.uniform(0, 1), 2),
                },
            ],
        },
        {
            "name": "dev-zimg-inference-gpu",
            "resolution": "gpu",
            "running_count": 1,
            "desired_count": 1,
            "avg_utilization": round(random.uniform(30, 70), 1),
            "targets": [
                {
                    "id": "10.0.3.22",
                    "availability_zone": "us-east-1c",
                    "health_state": "healthy",
                    "reason": None,
                    "active_connections": round(random.uniform(0, 1), 2),
                },
            ],
        },
    ]


def mock_gallery() -> dict:
    now = datetime.now(UTC)
    images = []
    prompts = [
        "A serene mountain lake at dawn, mist rising",
        "Cyberpunk cityscape with neon lights and rain",
        "Portrait of a cat wearing a space suit",
        "Abstract geometric patterns in gold and blue",
        "Japanese garden in autumn with red maple leaves",
        "Underwater scene with bioluminescent jellyfish",
    ]
    for i, prompt in enumerate(prompts):
        images.append(
            {
                "id": f"mock-{i:04d}",
                "prompt": prompt,
                "resolution": random.choice([512, 768, 1024]),
                "seed": random.randint(1, 99999),
                "steps": 8,
                "created_at": (now - timedelta(hours=i * 2)).isoformat(),
                "deleted": False,
                "url": f"https://picsum.photos/seed/{i}/512/512",
                "frames": [],
            }
        )
    return {"images": images, "next_cursor": None, "count": len(images)}


def mock_generate_frames(steps: int = 8, resolution: int = 512) -> list[bytes]:
    from PIL import Image

    frames = []

    for step in range(1, steps + 1):
        img = Image.new("RGB", (resolution, resolution))
        pixels = img.load()
        progress = step / steps
        for y in range(resolution):
            for x in range(min(32, resolution)):
                r = int(255 * progress * (x / 32))
                g = int(100 * (y / resolution))
                b = int(200 * (1 - progress))
                pixels[x, y] = (r, g, b)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        frame_type = 1 if step < steps else 2
        header = struct.pack(">III", frame_type, step, len(png_bytes))
        frames.append(header + png_bytes)

    import json

    metadata = json.dumps(
        {
            "steps": steps,
            "resolution": resolution,
            "seed": 42,
            "latency_s": round(random.uniform(3, 8), 2),
        }
    ).encode()
    header = struct.pack(">III", 0x03, steps, len(metadata))
    frames.append(header + metadata)

    return frames
