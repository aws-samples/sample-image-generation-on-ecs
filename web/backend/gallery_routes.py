from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from audit import audit
from auth import Role, TokenClaims, require_role
from config import settings
from gallery import gallery_manager

router = APIRouter(prefix="/api/gallery", tags=["gallery"])


@router.get("")
async def list_gallery(
    cursor: str | None = None,
    page_size: int = Query(default=20, ge=1, le=100),
    user: Annotated[TokenClaims, Depends(require_role(Role.USER))] = None,
):
    if settings.local:
        from mock_data import mock_gallery

        return mock_gallery()
    try:
        images, next_cursor = await gallery_manager.list_images(
            cursor=cursor, include_deleted=False, page_size=page_size
        )
        return {"images": images, "next_cursor": next_cursor, "count": len(images)}
    except Exception as e:
        logger.error(f"Failed to list gallery: {e}")
        raise HTTPException(status_code=500, detail="Gallery operation failed") from None


@router.get("/{image_id}")
async def get_image(
    image_id: str,
    user: Annotated[TokenClaims, Depends(require_role(Role.USER))] = None,
):
    try:
        image = await gallery_manager.get_image(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return image
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Gallery operation failed") from None


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    hard: bool = Query(default=False),
    user: Annotated[TokenClaims, Depends(require_role(Role.USER))] = None,
):
    if hard and Role.ADMIN.value not in user.groups:
        raise HTTPException(status_code=403, detail="Hard delete requires admin role")
    try:
        success = await gallery_manager.delete_image(image_id, hard_delete=hard)
        if not success:
            raise HTTPException(status_code=404, detail="Image not found")
        if hard:
            audit(
                actor_sub=user.sub,
                actor_email=user.email,
                action="hard_delete_image",
                resource=image_id,
            )
        return {"success": True, "image_id": image_id, "hard_deleted": hard}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Gallery operation failed") from None
