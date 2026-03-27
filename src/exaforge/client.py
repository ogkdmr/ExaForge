"""Async HTTP client for OpenAI-compatible vLLM endpoints.

Wraps ``httpx.AsyncClient`` with:

* Semaphore-based concurrency limiting.
* Automatic retries with exponential back-off.
* Load-balanced endpoint selection via :class:`EndpointPool`.
* Request tracking for least-loaded strategy.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from exaforge.config import ClientConfig
from exaforge.endpoints import Endpoint, EndpointPool

logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    """Payload for a single chat-completions call.

    Attributes
    ----------
    messages : list[dict]
        OpenAI-format message list
        (e.g. ``[{"role": "user", "content": "..."}]``).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    top_p : float
        Nucleus sampling threshold.
    model : str
        Model identifier passed in the request body.  Must match the
        model name vLLM was started with (e.g. ``openai/gpt-oss-120b``).
    """

    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    model: str = "default"


@dataclass
class ChatResponse:
    """Result of a single chat-completions call.

    Attributes
    ----------
    text : str
        The generated text (empty string on failure).
    success : bool
        Whether the request succeeded.
    error : str
        Error message if the request failed.
    endpoint_url : str
        Which endpoint served the request.
    latency : float
        Wall-clock seconds for the request.
    """

    text: str = ""
    success: bool = True
    error: str = ""
    endpoint_url: str = ""
    latency: float = 0.0


class InferenceClient:
    """High-throughput async client for chat completions.

    Parameters
    ----------
    pool : EndpointPool
        The endpoint pool to draw vLLM servers from.
    config : ClientConfig
        Client settings (concurrency, timeout, retries, etc.).
    """

    def __init__(
        self,
        pool: EndpointPool,
        config: ClientConfig,
    ) -> None:
        self.pool = pool
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(
                    max_connections=self.config.max_concurrent_requests + 10,
                    max_keepalive_connections=self.config.max_concurrent_requests,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Send a single chat-completions request with retries.

        The request is routed to a healthy endpoint chosen by the
        pool's load-balancing strategy.  On failure the request is
        retried up to ``config.max_retries`` times with exponential
        back-off.
        """
        async with self._semaphore:
            return await self._generate_with_retries(request)

    async def _generate_with_retries(
        self, request: ChatRequest
    ) -> ChatResponse:
        last_error = ""
        for attempt in range(1 + self.config.max_retries):
            try:
                ep = self.pool.select()
            except RuntimeError as exc:
                last_error = str(exc)
                if attempt < self.config.max_retries:
                    await asyncio.sleep(
                        self.config.retry_backoff ** attempt
                    )
                continue

            self.pool.acquire(ep)
            try:
                resp = await self._call(ep, request)
                if resp.success:
                    return resp
                last_error = resp.error
            finally:
                self.pool.release(ep)

            if attempt < self.config.max_retries:
                delay = self.config.retry_backoff ** attempt
                logger.warning(
                    "Retry %d/%d for endpoint %s after %.1fs: %s",
                    attempt + 1,
                    self.config.max_retries,
                    ep.url,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        return ChatResponse(
            success=False,
            error=f"All {1 + self.config.max_retries} attempts failed: {last_error}",
        )

    async def _call(
        self, ep: Endpoint, request: ChatRequest
    ) -> ChatResponse:
        client = await self._get_client()
        url = f"{ep.url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }

        loop = asyncio.get_event_loop()
        t0 = loop.time()
        try:
            resp = await client.post(url, json=payload)
            latency = loop.time() - t0

            if resp.status_code != 200:
                return ChatResponse(
                    success=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    endpoint_url=ep.url,
                    latency=latency,
                )

            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return ChatResponse(
                text=text,
                success=True,
                endpoint_url=ep.url,
                latency=latency,
            )

        except httpx.TimeoutException:
            return ChatResponse(
                success=False,
                error="Request timed out",
                endpoint_url=ep.url,
                latency=loop.time() - t0,
            )
        except httpx.HTTPError as exc:
            return ChatResponse(
                success=False,
                error=str(exc),
                endpoint_url=ep.url,
                latency=loop.time() - t0,
            )

    async def generate_batch(
        self, requests: list[ChatRequest]
    ) -> list[ChatResponse]:
        """Send multiple requests concurrently.

        Returns responses in the same order as the input requests.
        """
        tasks = [self.generate(req) for req in requests]
        return list(await asyncio.gather(*tasks))
