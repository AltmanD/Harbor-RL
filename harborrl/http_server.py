from __future__ import annotations

from typing import Any, Dict

from fastapi import Request


async def json_payload(request: Request) -> Dict[str, Any]:
    """Safely parse JSON body from a FastAPI request.

    Returns an empty dict if parsing fails or the payload is not a JSON object.
    """
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
