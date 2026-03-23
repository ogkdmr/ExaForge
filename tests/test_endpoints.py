"""Tests for the ExaForge endpoint management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from exaforge.endpoints import Endpoint, EndpointPool


class TestEndpointPool:
    def test_from_file(self, endpoints_file: Path) -> None:
        pool = EndpointPool.from_file(endpoints_file)
        assert len(pool.endpoints) == 2
        assert pool.endpoints[0].url.startswith("http://")

    def test_from_file_missing_raises(self, tmp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            EndpointPool.from_file(tmp_dir / "nope.txt")

    def test_from_file_empty_raises(self, tmp_dir: Path) -> None:
        p = tmp_dir / "empty.txt"
        p.write_text("")
        with pytest.raises(ValueError, match="No endpoints"):
            EndpointPool.from_file(p)

    def test_from_file_with_http_prefix(self, tmp_dir: Path) -> None:
        p = tmp_dir / "ep.txt"
        p.write_text("http://node1:8000\nhttp://node2:8001\n")
        pool = EndpointPool.from_file(p)
        assert pool.endpoints[0].url == "http://node1:8000"

    def test_from_file_skips_blank_lines(self, tmp_dir: Path) -> None:
        p = tmp_dir / "ep.txt"
        p.write_text("node1:8000\n\nnode2:8000\n\n")
        pool = EndpointPool.from_file(p)
        assert len(pool.endpoints) == 2


class TestSelection:
    def _pool(self) -> EndpointPool:
        eps = [
            Endpoint(url="http://n1:8000", healthy=True),
            Endpoint(url="http://n2:8000", healthy=True),
            Endpoint(url="http://n3:8000", healthy=False),
        ]
        return EndpointPool(eps, strategy="round_robin")

    def test_round_robin_cycles(self) -> None:
        pool = self._pool()
        urls = [pool.select().url for _ in range(4)]
        assert "http://n1:8000" in urls
        assert "http://n2:8000" in urls
        assert "http://n3:8000" not in urls

    def test_round_robin_skips_unhealthy(self) -> None:
        pool = self._pool()
        for _ in range(10):
            ep = pool.select()
            assert ep.healthy is True

    def test_least_loaded_picks_minimum(self) -> None:
        eps = [
            Endpoint(url="http://n1:8000", healthy=True, active_requests=5),
            Endpoint(url="http://n2:8000", healthy=True, active_requests=1),
        ]
        pool = EndpointPool(eps, strategy="least_loaded")
        assert pool.select().url == "http://n2:8000"

    def test_no_healthy_raises(self) -> None:
        eps = [Endpoint(url="http://n1:8000", healthy=False)]
        pool = EndpointPool(eps)
        with pytest.raises(RuntimeError, match="No healthy"):
            pool.select()

    def test_healthy_endpoints_property(self) -> None:
        pool = self._pool()
        assert len(pool.healthy_endpoints) == 2


class TestRequestTracking:
    def test_acquire_release(self) -> None:
        ep = Endpoint(url="http://n1:8000")
        pool = EndpointPool([ep])
        assert ep.active_requests == 0
        pool.acquire(ep)
        assert ep.active_requests == 1
        pool.release(ep)
        assert ep.active_requests == 0

    def test_release_does_not_go_negative(self) -> None:
        ep = Endpoint(url="http://n1:8000")
        pool = EndpointPool([ep])
        pool.release(ep)
        assert ep.active_requests == 0


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_check_health_all_healthy(self) -> None:
        eps = [Endpoint(url="http://n1:8000")]
        pool = EndpointPool(eps)

        mock_response = httpx.Response(200)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await pool.check_health(client=mock_client)
        assert result["http://n1:8000"] is True
        assert eps[0].healthy is True

    @pytest.mark.asyncio
    async def test_check_health_marks_unhealthy(self) -> None:
        eps = [Endpoint(url="http://n1:8000")]
        pool = EndpointPool(eps)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = await pool.check_health(client=mock_client)
        assert result["http://n1:8000"] is False
        assert eps[0].healthy is False
