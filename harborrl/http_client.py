"""Self-contained async HTTP POST with retry/backoff for harborrl.

This is a faithful local-only port of ``slime.utils.http_utils._post`` (plus
its retry helpers), so the environments and inference layers no longer import
the third-party training backend's internals.  Semantics — retry counting,
exponential backoff with jitter, Retry-After handling, retry-status
allowlists, and log throttling — are identical to the slime implementation;
the unused Ray distributed-POST dispatch is intentionally not ported
(``use_distributed_post`` is never enabled by HarborRL launchers).

The module owns a lazily-created ``httpx.AsyncClient`` sized by
``ENV_HTTP_MAX_CONNECTIONS`` (default 256); ``trust_env=False`` matches slime
so cluster proxies cannot interfere with worker traffic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random

import httpx

from harborrl.env import env_int as _env_int

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max(1, _env_int("ENV_HTTP_MAX_CONNECTIONS", 256))
            ),
            timeout=httpx.Timeout(None),
            trust_env=False,
        )
    return _http_client


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _retry_sleep_seconds(
    retry_count: int,
    *,
    base_delay: float,
    max_delay: float,
    backoff_factor: float,
    jitter: float,
    retry_after: float | None,
) -> float:
    delay = max(0.0, base_delay)
    if backoff_factor > 1.0 and retry_count > 1:
        delay *= backoff_factor ** (retry_count - 1)
    if max_delay > 0:
        delay = min(delay, max_delay)
    if retry_after is not None:
        delay = max(delay, retry_after)
    if jitter > 0 and delay > 0:
        delay += random.uniform(0.0, delay * jitter)
    return delay


def _should_log_retry(retry_count: int, max_retries: int) -> bool:
    """Keep retry storms readable while preserving early/final diagnostics."""
    if retry_count <= 3 or retry_count == max_retries:
        return True
    if retry_count in {5, 10, 20, 50}:
        return True
    every_n = _env_int("HTTP_RETRY_LOG_EVERY_N", 25)
    return every_n > 0 and retry_count % every_n == 0


def _compact_response_text(text: str | None) -> str | None:
    if text is None:
        return None
    limit = max(0, _env_int("HTTP_RETRY_LOG_RESPONSE_CHARS", 512))
    if limit <= 0:
        return None
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"... [truncated {len(text) - limit} chars]"


async def post(
    url,
    payload,
    max_retries=60,
    timeout=None,
    headers=None,
    *,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 1.0,
    retry_backoff_factor: float = 1.0,
    retry_jitter: float = 0.0,
    retry_statuses: set[int] | None = None,
    non_retry_statuses: set[int] | None = None,
):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await _client().post(
                url,
                json=payload or {},
                timeout=timeout,
                headers=headers,
            )
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            status_code = None
            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                if non_retry_statuses is not None and status_code in non_retry_statuses:
                    logger.info(
                        "Non-retryable HTTP status %s for url=%s, failing immediately.",
                        status_code,
                        url,
                    )
                    raise e
                if retry_statuses is not None and status_code not in retry_statuses:
                    logger.info(
                        "HTTP status %s is outside retry_statuses=%s for url=%s, failing immediately.",
                        status_code,
                        sorted(retry_statuses),
                        url,
                    )
                    raise e
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            if _should_log_retry(retry_count, max_retries):
                response_text = _compact_response_text(response_text)
                if response_text is None:
                    logger.info(
                        "Error: %s, retrying... (attempt %s/%s, url=%s)",
                        e,
                        retry_count,
                        max_retries,
                        url,
                    )
                else:
                    logger.info(
                        "Error: %s, retrying... (attempt %s/%s, url=%s, response=%s)",
                        e,
                        retry_count,
                        max_retries,
                        url,
                        response_text,
                    )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            sleep_seconds = _retry_sleep_seconds(
                retry_count,
                base_delay=retry_base_delay,
                max_delay=retry_max_delay,
                backoff_factor=retry_backoff_factor,
                jitter=retry_jitter,
                retry_after=_retry_after_seconds(response),
            )
            await asyncio.sleep(sleep_seconds)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output
