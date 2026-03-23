"""Programmatic Aegis integration.

Launches vLLM instances on Aurora via Aegis's scheduler and waits
for the endpoints file to appear.  Falls back gracefully when the
``aegis`` package is not installed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from exaforge.config import AegisConfig
from exaforge.endpoints import EndpointPool

logger = logging.getLogger(__name__)


def _aegis_available() -> bool:
    """Return True if the aegis package is importable."""
    try:
        import aegis  # noqa: F401

        return True
    except ImportError:
        return False


def launch_aegis(config: AegisConfig) -> list[str]:
    """Submit an Aegis PBS job and wait for endpoints.

    Parameters
    ----------
    config : AegisConfig
        ExaForge's Aegis sub-config.  Requires ``config_path`` to
        point at a valid Aegis YAML file.

    Returns
    -------
    list[str]
        Lines from the endpoints file (``host:port``).

    Raises
    ------
    ImportError
        If the ``aegis`` package is not installed.
    FileNotFoundError
        If the Aegis config file does not exist.
    RuntimeError
        If the PBS job fails or endpoints never appear.
    """
    if config.config_path is None:
        raise FileNotFoundError(
            "aegis.config_path must be set when auto_launch is enabled"
        )

    if not _aegis_available():
        raise ImportError(
            "The 'aegis' package is required for auto_launch. "
            "Install it with: pip install aegis"
        )

    from aegis.config import load_config as load_aegis_config
    from aegis.scheduler import (
        generate_pbs_script,
        submit_job,
        wait_for_endpoints,
    )

    aegis_config_path = Path(config.config_path)
    if not aegis_config_path.is_file():
        raise FileNotFoundError(
            f"Aegis config not found: {aegis_config_path}"
        )

    logger.info("Loading Aegis config from %s", aegis_config_path)
    aegis_cfg = load_aegis_config(aegis_config_path)

    logger.info("Generating PBS script")
    script = generate_pbs_script(aegis_cfg)

    hf_token = aegis_cfg.hf_token
    logger.info("Submitting PBS job")
    job_id = submit_job(script, hf_token=hf_token)

    endpoints_file = str(config.endpoints_file)
    logger.info("Waiting for endpoints (job %s)", job_id)

    endpoints = wait_for_endpoints(
        endpoints_file=endpoints_file,
        job_id=job_id,
    )
    logger.info("Aegis ready: %d endpoint(s)", len(endpoints))
    return endpoints


def get_endpoint_pool(config: AegisConfig) -> EndpointPool:
    """Obtain an EndpointPool, launching Aegis if configured.

    If ``config.auto_launch`` is True, submits an Aegis PBS job and
    waits for endpoints.  Otherwise, reads endpoints from the file
    specified by ``config.endpoints_file``.

    Parameters
    ----------
    config : AegisConfig
        ExaForge's Aegis sub-config.

    Returns
    -------
    EndpointPool
        A pool ready for inference.
    """
    if config.auto_launch:
        endpoints_lines = launch_aegis(config)
        # The endpoints file has been written by Aegis; load from it
        return EndpointPool.from_file(config.endpoints_file)

    return EndpointPool.from_file(config.endpoints_file)
