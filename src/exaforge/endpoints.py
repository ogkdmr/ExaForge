"""Endpoint management for Aegis-spawned vLLM instances.

Loads the Aegis endpoints file, performs health checks via the
``/health`` HTTP endpoint exposed by vLLM, and provides load-balanced
endpoint selection (round-robin or least-loaded).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Endpoint:
    """Represents a single vLLM server endpoint.

    Attributes
    ----------
    url : str
        The base URL, e.g. ``http://node1:8000``.
    healthy : bool
        Whether the last health check succeeded.
    active_requests : int
        Approximate number of in-flight requests (for least-loaded).
    last_check : float
        Monotonic timestamp of the last health probe.
    """

    url: str
    healthy: bool = True
    active_requests: int = 0
    last_check: float = 0.0


class EndpointPool:
    """Manages a set of vLLM endpoints with health checking and load balancing.

    Parameters
    ----------
    endpoints : list[Endpoint]
        The initial set of endpoints.
    strategy : str
        Load-balancing strategy: ``"round_robin"`` or ``"least_loaded"``.
    health_interval : float
        Minimum seconds between successive health checks per endpoint.
    """

    def __init__(
        self,
        endpoints: list[Endpoint],
        strategy: str = "round_robin",
        health_interval: float = 30.0,
    ) -> None:
        self.endpoints = endpoints
        self.strategy = strategy
        self.health_interval = health_interval
        self._rr_cycle = itertools.cycle(range(len(endpoints)))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: Path,
        strategy: str = "round_robin",
        health_interval: float = 30.0,
    ) -> "EndpointPool":
        """Create a pool from an Aegis endpoints file.

        The file format is one ``host:port`` per line (as written by
        ``aegis launch``).
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"Endpoints file not found: {path}"
            )
        lines = path.read_text().strip().splitlines()
        endpoints = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            url = line if line.startswith("http") else f"http://{line}"
            endpoints.append(Endpoint(url=url))
        if not endpoints:
            raise ValueError(f"No endpoints found in {path}")
        return cls(
            endpoints,
            strategy=strategy,
            health_interval=health_interval,
        )

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def check_health(
        self,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 10.0,
    ) -> dict[str, bool]:
        """Probe ``/health`` on every endpoint.

        Returns a mapping of URL -> healthy status.
        """
        own_client = client is None
        if own_client:
            client = httpx.AsyncClient()
        try:
            tasks = [
                self._probe(ep, client, timeout) for ep in self.endpoints
            ]
            results = await asyncio.gather(*tasks)
            return dict(results)
        finally:
            if own_client:
                await client.aclose()

    async def _probe(
        self,
        ep: Endpoint,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[str, bool]:
        now = time.monotonic()
        try:
            resp = await client.get(
                f"{ep.url}/health", timeout=timeout
            )
            ep.healthy = resp.status_code == 200
        except (httpx.HTTPError, Exception):
            ep.healthy = False
        ep.last_check = now
        return ep.url, ep.healthy

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    @property
    def healthy_endpoints(self) -> list[Endpoint]:
        """Return only endpoints whose last health check passed."""
        return [ep for ep in self.endpoints if ep.healthy]

    def select(self) -> Endpoint:
        """Pick an endpoint using the configured strategy.

        Raises
        ------
        RuntimeError
            If no healthy endpoints are available.
        """
        healthy = self.healthy_endpoints
        if not healthy:
            raise RuntimeError("No healthy endpoints available")

        if self.strategy == "least_loaded":
            return min(healthy, key=lambda ep: ep.active_requests)

        # Default: round-robin over healthy endpoints
        for _ in range(len(self.endpoints)):
            idx = next(self._rr_cycle)
            ep = self.endpoints[idx]
            if ep.healthy:
                return ep
        raise RuntimeError("No healthy endpoints available")

    # ------------------------------------------------------------------
    # Request tracking (for least-loaded)
    # ------------------------------------------------------------------

    def acquire(self, ep: Endpoint) -> None:
        """Mark that a request is starting on *ep*."""
        ep.active_requests += 1

    def release(self, ep: Endpoint) -> None:
        """Mark that a request on *ep* has completed."""
        ep.active_requests = max(0, ep.active_requests - 1)
