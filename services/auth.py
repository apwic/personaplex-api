"""Authentication utilities for PersonaPlex API."""

import os
from fastapi import HTTPException, Header

API_KEY = os.getenv("API_KEY", "default-api-key-change-me")


async def verify_api_key(authorization: str = Header(None)) -> str:
    """Verify API key from Authorization header.

    Args:
        authorization: Authorization header value

    Returns:
        The validated token

    Raises:
        HTTPException: If authorization is missing or invalid
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Use 'Bearer <token>'")

    token = authorization[7:]  # Remove "Bearer " prefix

    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return token
