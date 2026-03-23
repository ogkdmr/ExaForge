"""Tests for the Aegis bridge module."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exaforge.config import AegisConfig
from exaforge.aegis_bridge import _aegis_available, get_endpoint_pool


class TestAegisAvailable:
    def test_returns_bool(self) -> None:
        result = _aegis_available()
        assert isinstance(result, bool)


class TestGetEndpointPool:
    def test_loads_from_file(self, endpoints_file: Path) -> None:
        cfg = AegisConfig(
            auto_launch=False,
            endpoints_file=endpoints_file,
        )
        pool = get_endpoint_pool(cfg)
        assert len(pool.endpoints) == 2

    def test_missing_file_raises(self, tmp_dir: Path) -> None:
        cfg = AegisConfig(
            auto_launch=False,
            endpoints_file=tmp_dir / "nonexistent.txt",
        )
        with pytest.raises(FileNotFoundError):
            get_endpoint_pool(cfg)

    def test_auto_launch_without_aegis_raises(
        self, tmp_dir: Path
    ) -> None:
        cfg = AegisConfig(
            auto_launch=True,
            config_path=tmp_dir / "aegis.yaml",
        )
        # Ensure the config file exists so we don't hit the earlier check
        (tmp_dir / "aegis.yaml").write_text("model: test\n")
        with patch(
            "exaforge.aegis_bridge._aegis_available", return_value=False
        ):
            with pytest.raises(ImportError, match="aegis"):
                get_endpoint_pool(cfg)

    def test_auto_launch_missing_config_path_raises(
        self, tmp_dir: Path
    ) -> None:
        cfg = AegisConfig(auto_launch=True, config_path=None)
        with pytest.raises(FileNotFoundError, match="config_path"):
            get_endpoint_pool(cfg)

    def test_auto_launch_mocked_aegis(
        self, tmp_dir: Path, endpoints_file: Path
    ) -> None:
        """Test the full launch path with mocked Aegis modules."""
        aegis_yaml = tmp_dir / "aegis.yaml"
        aegis_yaml.write_text("model: test\naccount: test\n")

        cfg = AegisConfig(
            auto_launch=True,
            config_path=aegis_yaml,
            endpoints_file=endpoints_file,
        )

        mock_aegis_cfg = MagicMock()
        mock_load = MagicMock(return_value=mock_aegis_cfg)
        mock_generate = MagicMock(return_value="#!/bin/bash\n")
        mock_submit = MagicMock(return_value="12345.pbs")
        mock_wait = MagicMock(
            return_value=["node1:8000", "node2:8000"]
        )

        # Create mock aegis modules in sys.modules
        aegis_mod = types.ModuleType("aegis")
        aegis_config_mod = types.ModuleType("aegis.config")
        aegis_scheduler_mod = types.ModuleType("aegis.scheduler")

        aegis_config_mod.load_config = mock_load
        aegis_scheduler_mod.generate_pbs_script = mock_generate
        aegis_scheduler_mod.submit_job = mock_submit
        aegis_scheduler_mod.wait_for_endpoints = mock_wait

        # Remove any cached import of the bridge function
        import importlib
        import exaforge.aegis_bridge

        with patch.dict(
            sys.modules,
            {
                "aegis": aegis_mod,
                "aegis.config": aegis_config_mod,
                "aegis.scheduler": aegis_scheduler_mod,
            },
        ):
            importlib.reload(exaforge.aegis_bridge)
            from exaforge.aegis_bridge import get_endpoint_pool as gep

            pool = gep(cfg)
            assert len(pool.endpoints) == 2
            mock_load.assert_called_once()
            mock_submit.assert_called_once()

        # Restore
        importlib.reload(exaforge.aegis_bridge)
