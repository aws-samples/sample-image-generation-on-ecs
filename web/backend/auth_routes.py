from __future__ import annotations

import urllib.parse

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from config import settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/logout")
async def auth_logout(request: Request):
    scheme = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("host", "localhost")
    logout_uri = f"{scheme}://{host}/"
    cognito_logout_url = (
        f"https://{settings.cognito_domain}/logout?"
        f"client_id={settings.cognito_app_client_id}&"
        f"logout_uri={urllib.parse.quote(logout_uri)}"
    )
    return RedirectResponse(cognito_logout_url, status_code=302)
