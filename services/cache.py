from __future__ import annotations

from typing import Any, Optional

try:
    import redis
except ImportError:
    redis = None

from cfg import settings


_redis_client: Optional[Any] = None


def redis_client() -> Optional[Any]:
    global _redis_client

    if redis is None:
        print("[CACHE] redis package is not installed. Cache disabled.")
        return None

    if not settings.REDIS_URL:
        return None

    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    return _redis_client


def cache_get(key: str) -> Optional[str]:
    client = redis_client()
    if client is None:
        return None

    try:
        return client.get(key)
    except Exception as e:
        print(f"[CACHE] Redis get failed: {e}")
        return None


def cache_set_json(key: str, value: str, ttl_seconds: int) -> None:
    client = redis_client()
    if client is None:
        return

    try:
        client.setex(key, ttl_seconds, value)
    except Exception as e:
        print(f"[CACHE] Redis set failed: {e}")
