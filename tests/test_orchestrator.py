"""Tests for the ExaForge pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from exaforge.config import (
    CheckpointConfig,
    ClientConfig,
    ExaForgeConfig,
    GenerationTaskConfig,
    JsonlWriterConfig,
    TextDirectoryReaderConfig,
)
from exaforge.endpoints import Endpoint, EndpointPool
from exaforge.orchestrator import Orchestrator


def _chat_ok(content: str = "Generated output") -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _pool() -> EndpointPool:
    return EndpointPool(
        [Endpoint(url="http://node1:8000", healthy=True)],
        strategy="round_robin",
    )


def _config(
    tmp_dir: Path, input_dir: Path | None = None
) -> ExaForgeConfig:
    return ExaForgeConfig(
        task=GenerationTaskConfig(
            system_prompt="Test prompt",
            max_tokens=100,
        ),
        reader=TextDirectoryReaderConfig(
            input_dir=input_dir or tmp_dir,
            glob_patterns=["*.txt"],
        ),
        writer=JsonlWriterConfig(
            output_dir=tmp_dir / "output",
            buffer_size=100,
            base_name="results",
        ),
        client=ClientConfig(
            max_concurrent_requests=5,
            timeout=5.0,
            max_retries=0,
            retry_backoff=0.01,
        ),
        checkpoint=CheckpointConfig(
            enabled=True,
            checkpoint_file=tmp_dir / "checkpoint.json",
        ),
    )


class TestOrchestrator:
    @pytest.mark.asyncio
    @respx.mock
    async def test_basic_pipeline(
        self, tmp_dir: Path, sample_texts: list[Path]
    ) -> None:
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("answer"))
        )

        cfg = _config(tmp_dir, input_dir=sample_texts[0].parent)
        orch = Orchestrator(cfg, _pool())
        summary = await orch.run()

        assert summary["completed"] == 5
        assert summary["failed"] == 0
        assert summary["total"] == 5

        output_files = list((tmp_dir / "output").glob("*.jsonl"))
        assert len(output_files) >= 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_checkpoint_resumes(
        self, tmp_dir: Path, sample_texts: list[Path]
    ) -> None:
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("ok"))
        )

        cfg = _config(tmp_dir, input_dir=sample_texts[0].parent)

        # First run — process all 5
        orch1 = Orchestrator(cfg, _pool())
        s1 = await orch1.run()
        assert s1["completed"] == 5

        # Second run — all checkpointed, nothing to do
        orch2 = Orchestrator(cfg, _pool())
        s2 = await orch2.run()
        assert s2["completed"] == 0
        assert s2["total"] == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_failures(
        self, tmp_dir: Path, sample_texts: list[Path]
    ) -> None:
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="error")
        )

        cfg = _config(tmp_dir, input_dir=sample_texts[0].parent)
        orch = Orchestrator(cfg, _pool())
        summary = await orch.run()

        assert summary["failed"] == 5
        assert summary["completed"] == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_output_is_valid_jsonl(
        self, tmp_dir: Path, sample_texts: list[Path]
    ) -> None:
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_chat_ok("test response")
            )
        )

        cfg = _config(tmp_dir, input_dir=sample_texts[0].parent)
        orch = Orchestrator(cfg, _pool())
        await orch.run()

        output_files = list((tmp_dir / "output").glob("*.jsonl"))
        for f in output_files:
            for line in f.read_text().strip().splitlines():
                record = json.loads(line)
                assert "id" in record
                assert "response" in record

    @pytest.mark.asyncio
    @respx.mock
    async def test_progress_callback(
        self, tmp_dir: Path, sample_texts: list[Path]
    ) -> None:
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("x"))
        )

        progress_calls: list[tuple] = []

        def on_progress(
            completed: int,
            total: int,
            item_id: str,
            latency: float,
            endpoint_url: str = "",
        ) -> None:
            progress_calls.append(
                (completed, total, item_id, latency, endpoint_url)
            )

        cfg = _config(tmp_dir, input_dir=sample_texts[0].parent)
        orch = Orchestrator(cfg, _pool(), on_progress=on_progress)
        await orch.run()

        assert len(progress_calls) == 5

    @pytest.mark.asyncio
    async def test_empty_input(self, tmp_dir: Path) -> None:
        cfg = _config(tmp_dir)
        orch = Orchestrator(cfg, _pool())
        summary = await orch.run()
        assert summary["total"] == 0
        assert summary["completed"] == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_batch_processing_completes_all(
        self, tmp_dir: Path, large_sample_texts: list[Path]
    ) -> None:
        """Items spread across multiple batches are all processed."""
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("ok"))
        )

        cfg = _config(tmp_dir, input_dir=large_sample_texts[0].parent)
        cfg = cfg.model_copy(update={"batch_size": 7})  # 20 items → 3 batches
        orch = Orchestrator(cfg, _pool())
        summary = await orch.run()

        assert summary["total"] == 20
        assert summary["completed"] == 20
        assert summary["failed"] == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_batch_size_1_processes_all(
        self, tmp_dir: Path, large_sample_texts: list[Path]
    ) -> None:
        """Degenerate batch_size=1 still produces correct results."""
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("ok"))
        )

        cfg = _config(tmp_dir, input_dir=large_sample_texts[0].parent)
        cfg = cfg.model_copy(update={"batch_size": 1})
        orch = Orchestrator(cfg, _pool())
        summary = await orch.run()

        assert summary["completed"] == 20

    @pytest.mark.asyncio
    @respx.mock
    async def test_batch_larger_than_dataset(
        self, tmp_dir: Path, sample_texts: list[Path]
    ) -> None:
        """batch_size > total items still works correctly (single batch)."""
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("ok"))
        )

        cfg = _config(tmp_dir, input_dir=sample_texts[0].parent)
        cfg = cfg.model_copy(update={"batch_size": 10000})
        orch = Orchestrator(cfg, _pool())
        summary = await orch.run()

        assert summary["completed"] == 5

    @pytest.mark.asyncio
    @respx.mock
    async def test_max_items_respected_across_batches(
        self, tmp_dir: Path, large_sample_texts: list[Path]
    ) -> None:
        """max_items cap applies before batching, not per-batch."""
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("ok"))
        )

        cfg = _config(tmp_dir, input_dir=large_sample_texts[0].parent)
        cfg = cfg.model_copy(update={"max_items": 9, "batch_size": 4})
        orch = Orchestrator(cfg, _pool())
        summary = await orch.run()

        assert summary["total"] == 9
        assert summary["completed"] == 9

    @pytest.mark.asyncio
    @respx.mock
    async def test_checkpoint_across_batches(
        self, tmp_dir: Path, large_sample_texts: list[Path]
    ) -> None:
        """Checkpoint is saved after each batch; a second run skips them."""
        respx.post("http://node1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_ok("ok"))
        )

        cfg = _config(tmp_dir, input_dir=large_sample_texts[0].parent)
        cfg = cfg.model_copy(update={"batch_size": 6})

        orch1 = Orchestrator(cfg, _pool())
        s1 = await orch1.run()
        assert s1["completed"] == 20

        # Second run — all checkpointed
        orch2 = Orchestrator(cfg, _pool())
        s2 = await orch2.run()
        assert s2["completed"] == 0
        assert s2["total"] == 0
