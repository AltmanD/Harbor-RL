import asyncio
import ipaddress
import json
import logging
import multiprocessing
import os
import random
import socket

import httpx

logger = logging.getLogger(__name__)

SLIME_HOST_IP_ENV = "SLIME_HOST_IP"


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip

    def _is_loopback(ip):
        return ip.startswith("127.") or ip == "::1"

    def _resolve_ip(family, test_target_ip):
        """
        Attempt to get the local LAN IP for the specific family (IPv4/IPv6).
        Strategy: UDP Probe (Preferred) -> Hostname Resolution (Fallback) -> None
        """

        # Strategy 1: UDP Connect Probe (Most accurate, relies on routing table)
        # Useful when the machine has a default gateway or internet access.
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                # The IP doesn't need to be reachable, but the routing table must exist.
                s.connect((test_target_ip, 80))
                ip = s.getsockname()[0]
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass  # Route unreachable or network error, move to next strategy.

        # Strategy 2: Hostname Resolution (Fallback for offline clusters)
        # Useful for offline environments where UDP connect fails but /etc/hosts is configured.
        try:
            # getaddrinfo allows specifying the family (AF_INET or AF_INET6)
            # Result format: [(family, type, proto, canonname, sockaddr), ...]
            infos = socket.getaddrinfo(hostname, None, family=family, type=socket.SOCK_STREAM)

            for info in infos:
                ip = info[4][0]  # The first element of sockaddr is the IP
                # Must filter out loopback addresses to avoid "127.0.0.1" issues
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass

        return None

    prefer_ipv6 = os.getenv("SLIME_PREFER_IPV6", "0").lower() in ("1", "true", "yes", "on")
    local_ip = None
    final_fallback = "127.0.0.1"

    if prefer_ipv6:
        # [Strict Mode] IPv6 Only
        # 1. Try UDP V6 Probe
        # 2. Try Hostname Resolution (V6)
        # If failed, fallback to V6 loopback. Never mix with V4.
        local_ip = _resolve_ip(socket.AF_INET6, "2001:4860:4860::8888")
        final_fallback = "::1"
    else:
        # [Strict Mode] IPv4 Only (Default)
        # 1. Try UDP V4 Probe
        # 2. Try Hostname Resolution (V4)
        # If failed, fallback to V4 loopback. Never mix with V6.
        local_ip = _resolve_ip(socket.AF_INET, "8.8.8.8")
        final_fallback = "127.0.0.1"

    return hostname, local_ip or final_fallback


def _wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host


def run_router(args):
    try:
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        logger.info(e)
        return 1


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    """Terminate a process gracefully, with forced kill as fallback.

    Args:
        process: The process to terminate
        timeout: Seconds to wait for graceful termination before forcing kill
    """
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()


_http_client: httpx.AsyncClient | None = None
_client_concurrency: int = 0

# Optional Ray-based distributed POST dispatch
_distributed_post_enabled: bool = False
_post_actors: list[object] = []
_post_actor_idx: int = 0


def _next_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


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


async def _post(
    client,
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
            response = await client.post(
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


def init_http_client(args):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    if not args.rollout_num_gpus:
        return

    _client_concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
            trust_env=False,
        )

    # Optionally initialize distributed POST via Ray without changing interfaces
    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args):
    """Initialize one or more Ray async actors per node for HTTP POST.

    Uses NodeAffinitySchedulingStrategy to place actors on distinct nodes.
    Controlled by SLIME_HTTP_POST_ACTORS_PER_NODE.
    """
    global _post_actors
    if _post_actors:
        return  # Already initialized

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Discover alive nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    # Define the async actor
    @ray.remote
    class _HttpPosterActor:
        def __init__(self, concurrency: int):
            # Lazy creation to this actor's event loop
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(1, concurrency)),
                timeout=httpx.Timeout(None),
                trust_env=False,
            )

        async def do_post(
            self,
            url,
            payload,
            max_retries=60,
            timeout=None,
            headers=None,
            retry_base_delay=1.0,
            retry_max_delay=1.0,
            retry_backoff_factor=1.0,
            retry_jitter=0.0,
            retry_statuses=None,
            non_retry_statuses=None,
        ):
            return await _post(
                self._client,
                url,
                payload,
                max_retries,
                timeout=timeout,
                headers=headers,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
                retry_backoff_factor=retry_backoff_factor,
                retry_jitter=retry_jitter,
                retry_statuses=set(retry_statuses) if retry_statuses is not None else None,
                non_retry_statuses=set(non_retry_statuses) if non_retry_statuses is not None else None,
            )

    # Create actors per node
    created = []
    # Distribute client concurrency across actors (at least 1 per actor)
    per_actor_conc = (_client_concurrency + len(nodes)) // len(nodes)

    for node in nodes:
        node_id = node["NodeID"]
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        for _ in range(args.num_gpus_per_node):
            actor = _HttpPosterActor.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=scheduling,
                max_concurrency=per_actor_conc,
                # Use tiny CPU to schedule
                num_cpus=0.001,
            ).remote(per_actor_conc)
            created.append(actor)

    _post_actors = created


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
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    if _distributed_post_enabled and _post_actors:
        try:
            import ray

            actor = _next_actor()
            if actor is not None:
                # Use a thread to avoid blocking the event loop on ray.get
                obj_ref = actor.do_post.remote(
                    url,
                    payload,
                    max_retries,
                    timeout,
                    headers,
                    retry_base_delay,
                    retry_max_delay,
                    retry_backoff_factor,
                    retry_jitter,
                    sorted(retry_statuses) if retry_statuses is not None else None,
                    sorted(non_retry_statuses) if non_retry_statuses is not None else None,
                )
                try:
                    if timeout is not None:
                        return await asyncio.to_thread(ray.get, obj_ref, timeout=timeout)
                    return await asyncio.to_thread(ray.get, obj_ref)
                except TimeoutError as e:
                    raise TimeoutError(
                        f"Distributed POST timed out after {timeout}s (url={url})"
                    ) from e
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    return await _post(
        _http_client,
        url,
        payload,
        max_retries,
        timeout=timeout,
        headers=headers,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        retry_backoff_factor=retry_backoff_factor,
        retry_jitter=retry_jitter,
        retry_statuses=retry_statuses,
        non_retry_statuses=non_retry_statuses,
    )


async def get(url):
    response = await _http_client.get(url)
    response.raise_for_status()
    content = await response.aread()
    output = json.loads(content)
    return output
