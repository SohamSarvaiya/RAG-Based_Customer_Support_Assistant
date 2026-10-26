import secrets

from fastapi import Header, HTTPException, Request


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    valid = request.app.state.settings.api_key_list
    if not x_api_key or not any(secrets.compare_digest(x_api_key, k) for k in valid):
        raise HTTPException(status_code=401, detail="Invalid or missing API key (X-API-Key header).")
