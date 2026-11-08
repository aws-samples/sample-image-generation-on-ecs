import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

import inference_client
from admin import router as admin_router
from admin_instances import router as admin_instances_router
from auth import Role, TokenClaims, require_role
from auth_routes import router as auth_router
from config import settings
from gallery_routes import router as gallery_router

FRONTEND_DIR = Path(__file__).parent / "static"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Z-Image UI server")
    logger.info(
        f"Endpoints: 512={settings.endpoint_512} 768={settings.endpoint_768} "
        f"1024={settings.endpoint_1024} gpu={settings.endpoint_gpu}"
    )
    logger.info(f"Default backend: {settings.default_backend}")
    logger.info(f"Gallery bucket: {settings.gallery_bucket}")
    logger.info(f"Running in {'Lambda' if settings.is_lambda else 'local'} mode")
    yield
    logger.info("Shutting down Z-Image UI server")


app = FastAPI(title="Z-Image UI", version="1.0.0", lifespan=lifespan)


def _should_register_cors(s) -> bool:
    return not s.is_lambda


if _should_register_cors(settings):
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://localhost:\d+$|^http://127\.0\.0\.1:\d+$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )

app.include_router(auth_router)

app.include_router(admin_router)
app.include_router(admin_instances_router)

if not settings.is_lambda:
    from local_ws import router as local_ws_router

    app.include_router(local_ws_router)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    width: int = Field(default=512, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    seed: int | None = None
    steps: int = Field(default=8, ge=1, le=20)
    backend: str = Field(default_factory=lambda: settings.default_backend)


@app.post("/api/generate")
async def generate_image(
    req: GenerateRequest,
    user: Annotated[TokenClaims, Depends(require_role(Role.USER))],
):
    import random

    width = req.width
    height = req.height
    backend = req.backend
    try:
        endpoint, _ = settings.pick_endpoint(width, backend)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    is_gpu = endpoint == settings.endpoint_gpu

    if is_gpu:
        step = settings.gpu_dimension_step
        if width % step != 0 or height % step != 0:
            raise HTTPException(422, f"Dimensions must be divisible by {step}")
        if width * height > settings.gpu_max_pixels:
            raise HTTPException(
                422,
                f"Resolution {width}x{height} ({width * height:,} px) exceeds "
                f"GPU budget of {settings.gpu_max_pixels:,} px",
            )
        if width < settings.gpu_min_dimension or height < settings.gpu_min_dimension:
            raise HTTPException(422, f"Minimum dimension is {settings.gpu_min_dimension}px")
    else:
        if width != height or width not in (512, 768, 1024):
            raise HTTPException(422, "Neuron backend only supports 512x512, 768x768, or 1024x1024")

    seed = req.seed if req.seed is not None else random.randint(0, 2**31)
    payload = {
        "prompt": req.prompt,
        "steps": req.steps,
        "height": height,
        "width": width,
        "seed": seed,
        "denoise_viz": False,
    }
    target = "gpu" if is_gpu else str(width)
    try:
        result = inference_client.invoke_json(target, payload)
    except Exception as e:
        raise HTTPException(502, f"Inference failed: {e}") from e
    return result


@app.post("/api/ws-ticket")
async def create_ws_ticket(
    user: Annotated[TokenClaims, Depends(require_role(Role.USER))],
):
    import time

    from jose import jwt

    from ws_ticket import get_ws_ticket_secret

    payload = {
        "sub": user.sub,
        "email": user.email,
        "groups": user.groups,
        "iat": int(time.time()),
        "exp": int(time.time()) + 30,
        "purpose": "ws-connect",
    }
    secret = get_ws_ticket_secret()
    ticket = jwt.encode(payload, secret, algorithm="HS256")
    return {"ticket": ticket}


@app.get("/api/health")
async def health_check():
    resolutions = [512, 768, 1024]
    results = {f"neuron_{r}": inference_client.health_check(r) for r in resolutions}
    overall = "ok" if all(r.get("status") == "healthy" for r in results.values()) else "degraded"
    return {
        "status": overall,
        "version": "1.0.0",
        "endpoints": results,
    }


@app.get("/api/config")
async def get_config():
    from fastapi.responses import JSONResponse

    return JSONResponse(
        content={
            "default_backend": settings.default_backend,
            "gpu_enabled": settings.gpu_enabled,
            "inferentia_enabled": settings.inferentia_enabled,
            "gpu_max_pixels": settings.gpu_max_pixels,
            "gpu_min_dimension": settings.gpu_min_dimension,
            "gpu_dimension_step": settings.gpu_dimension_step,
            "region": settings.region,
            "ws_heartbeat_interval": settings.ws_heartbeat_interval,
            "websocket_url": settings.websocket_url,
        },
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/api/endpoint-status")
async def get_endpoint_status():

    import aioboto3
    from fastapi.responses import JSONResponse

    target_groups: dict[str, str] = {}
    if settings.inferentia_enabled:
        target_groups["neuron_512"] = f"{settings.prefix}-inf-512-tg"
        target_groups["neuron_768"] = f"{settings.prefix}-inf-768-tg"
        target_groups["neuron_1024"] = f"{settings.prefix}-inf-1024-tg"
    if settings.gpu_enabled:
        target_groups["gpu"] = f"{settings.prefix}-inf-gpu-tg"

    async def _check(key: str, tg_name: str) -> str:
        session = aioboto3.Session(region_name=settings.region)
        async with session.client("elbv2", region_name=settings.region) as elbv2:
            try:
                tgs = await elbv2.describe_target_groups(Names=[tg_name])
                tg_arn = tgs["TargetGroups"][0]["TargetGroupArn"]
                health = await elbv2.describe_target_health(TargetGroupArn=tg_arn)
                targets = health.get("TargetHealthDescriptions", [])
                if not targets:
                    return "offline"
                if any(t["TargetHealth"]["State"] == "healthy" for t in targets):
                    return "ready"
                return "warming"
            except Exception as exc:
                logger.warning(f"operation failed: {exc}")
                return "offline"

    results = await asyncio.gather(*[_check(k, n) for k, n in target_groups.items()])
    status_map = dict(zip(target_groups.keys(), results, strict=True))
    return JSONResponse(content=status_map, headers={"Cache-Control": "max-age=30"})


app.include_router(gallery_router)


if not settings.is_lambda and FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        file_path = FRONTEND_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")

elif not settings.is_lambda:
    logger.warning(f"Frontend not built at {FRONTEND_DIR}")

    @app.get("/")
    async def root():
        return {"message": "Z-Image UI API", "frontend": "not built"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.local,
        log_level="info",
    )
