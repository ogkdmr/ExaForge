"""Tests for the ExaForge async inference client."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from exaforge.client import ChatRequest, ChatResponse, InferenceClient
from exaforge.config import ClientConfig
from exaforge.endpoints import Endpoint, EndpointPool


def _pool(urls: list[str] | None = None) -> EndpointPool:
    urls = urls or ["http://node1:8000"]
    eps = [Endpoint(url=u, healthy=True) for u in urls]
    return EndpointPool(eps, strategy="round_robin")


def _config(**kwargs) -> ClientConfig:
    defaults = {
        "max_concurrent_requests": 10,
        "timeout": 5.0,
        "max_retries": 1,
        "retry_backoff": 0.01,
    }
    defaults.update(kwargs)
    return ClientConfig(**defaults)


def _chat_ok(content: str = "Hello!") -> dict:
    return {
        "choices": [{"message": {"content": content}}],
    }


class TestInferenceClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self) -> None:
        route = respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("world"))
        )
        client = InferenceClient(_pool(), _config())
        try:
            req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
            resp = await client.generate(req)
            assert resp.success
            assert resp.text == "world"
            assert resp.endpoint_url == "http://node1:8000"
            assert resp.latency > 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_http_error_retries(self) -> None:
        route = respx.post("http://node1:8000/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(500, text="Internal error"),
                httpx.Response(200, json=_chat_ok("ok")),
            ]
        )
        client = InferenceClient(_pool(), _config(max_retries=2))
        try:
            req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
            resp = await client.generate(req)
            assert resp.success
            assert resp.text == "ok"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_all_retries_fail(self) -> None:
        route = respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="error")
        )
        client = InferenceClient(_pool(), _config(max_retries=1))
        try:
            req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
            resp = await client.generate(req)
            assert not resp.success
            assert "attempts failed" in resp.error
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_batch(self) -> None:
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("batch"))
        )
        client = InferenceClient(_pool(), _config())
        try:
            reqs = [
                ChatRequest(messages=[{"role": "user", "content": f"q{i}"}])
                for i in range(5)
            ]
            resps = await client.generate_batch(reqs)
            assert len(resps) == 5
            assert all(r.success for r in resps)
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_concurrency_limited(self) -> None:
        """Verify the semaphore limits concurrent requests."""
        active = 0
        max_active = 0

        async def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(200, json=_chat_ok("ok"))

        respx.post("http://node1:8000/v1/chat/completions").mock(
            side_effect=_handler
        )
        client = InferenceClient(_pool(), _config(max_concurrent_requests=3))
        try:
            reqs = [
                ChatRequest(messages=[{"role": "user", "content": "x"}])
                for _ in range(10)
            ]
            await client.generate_batch(reqs)
            assert max_active <= 3
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_load_balances_across_endpoints(self) -> None:
        urls = ["http://n1:8000", "http://n2:8000"]
        for url in urls:
            respx.post(f"{url}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_chat_ok("ok"))
            )
        client = InferenceClient(_pool(urls), _config())
        try:
            reqs = [
                ChatRequest(messages=[{"role": "user", "content": "x"}])
                for _ in range(6)
            ]
            resps = await client.generate_batch(reqs)
            used = {r.endpoint_url for r in resps}
            assert len(used) == 2
        finally:
            await client.close()


class TestChatRequest:
    def test_defaults(self) -> None:
        req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.temperature == 0.7
        assert req.max_tokens == 2000

    def test_custom_values(self) -> None:
        req = ChatRequest(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            max_tokens=500,
            model="my-model",
        )
        assert req.model == "my-model"


class TestChatResponse:
    def test_success_defaults(self) -> None:
        resp = ChatResponse(text="hello")
        assert resp.success is True
        assert resp.error == ""

    def test_failure(self) -> None:
        resp = ChatResponse(success=False, error="timeout")
        assert resp.text == ""
