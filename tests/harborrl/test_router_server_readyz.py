from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class _FastAPI:
    def get(self, *_args, **_kwargs):
        return lambda fn: fn

    def post(self, *_args, **_kwargs):
        return lambda fn: fn

    def on_event(self, *_args, **_kwargs):
        return lambda fn: fn


class _JSONResponse(dict):
    def __init__(self, content=None, status_code=200):
        super().__init__(content or {})
        self.status_code = status_code


class _ClientError(Exception):
    pass


def _install_import_stubs(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = _FastAPI
    fastapi_mod.Request = object

    responses_mod = types.ModuleType("fastapi.responses")
    responses_mod.JSONResponse = _JSONResponse

    aiohttp_mod = types.ModuleType("aiohttp")
    aiohttp_mod.ClientError = _ClientError

    class _ClientTimeout:
        def __init__(self, total=None):
            self.total = total

    class _TCPConnector:
        def __init__(self, **_kwargs):
            pass

    aiohttp_mod.ClientTimeout = _ClientTimeout
    aiohttp_mod.TCPConnector = _TCPConnector

    request_utils_mod = types.ModuleType("harborrl.http_server")

    async def _json_payload(_request):
        return {}

    request_utils_mod.json_payload = _json_payload

    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_mod)
    monkeypatch.setitem(sys.modules, "harborrl.http_server", request_utils_mod)
    sys.modules.pop("harborrl.platform.router_app", None)
    return importlib.import_module("harborrl.platform.router_app")


class _FakeRouter:
    def __init__(self, results):
        self.workers = [f"http://worker-{idx}" for idx, _ in enumerate(results)]
        self._results = results

    async def maybe_reload_workers(self):
        return None

    @property
    def num_workers(self):
        return len(self.workers)

    async def worker_readiness(self, worker_url, timeout=5.0):
        idx = self.workers.index(worker_url)
        result = self._results[idx]
        if isinstance(result, BaseException):
            raise result
        return result


def test_readyz_returns_ok_when_any_worker_is_ready(monkeypatch):
    async def _case():
        router_server = _install_import_stubs(monkeypatch)
        router_server.ROUTER = _FakeRouter(
            [
                ({"ok": False, "code": "WORKER_STALE_RUNS"}, 503, "/readyz"),
                ({"ok": True}, 200, "/readyz"),
            ]
        )

        response = await router_server.readyz()

        assert response.status_code == 200
        assert response["ok"] is True
        assert response["ready_workers"] == 1
        assert response["workers"][0]["ready"] is False
        assert response["workers"][1]["ready"] is True

    asyncio.run(_case())


def test_readyz_returns_503_when_no_workers_are_ready(monkeypatch):
    async def _case():
        router_server = _install_import_stubs(monkeypatch)
        router_server.ROUTER = _FakeRouter(
            [
                ({"ok": False, "code": "WORKER_STALE_RUNS"}, 503, "/readyz"),
                _ClientError("connect failed"),
            ]
        )

        response = await router_server.readyz()

        assert response.status_code == 503
        assert response["ok"] is False
        assert response["code"] == "NO_READY_WORKERS"
        assert response["ready_workers"] == 0
        assert response["workers"][1]["ready"] is False
        assert "connect failed" in response["workers"][1]["error"]

    asyncio.run(_case())
