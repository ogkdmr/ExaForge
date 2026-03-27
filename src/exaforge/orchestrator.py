"""Pipeline orchestrator.

Ties together reader, task, client, writer, checkpoint, and monitor
into a single async pipeline:

1. Read all inputs.
2. Filter out already-completed items (via checkpoint).
3. For each pending item, prepare messages (task) and send to vLLM
   (client).
4. Write results (writer) and update the checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from exaforge.checkpoint import CheckpointManager
from exaforge.client import ChatRequest, ChatResponse, InferenceClient
from exaforge.config import ExaForgeConfig
from exaforge.endpoints import EndpointPool
from exaforge.readers import get_reader
from exaforge.readers.base import InputItem
from exaforge.tasks import get_task
from exaforge.tasks.base import BaseTask
from exaforge.writers import get_writer
from exaforge.writers.base import BaseWriter, OutputRecord

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main inference pipeline.

    Parameters
    ----------
    config : ExaForgeConfig
        The full pipeline configuration.
    pool : EndpointPool
        Pre-initialised endpoint pool (caller is responsible for
        obtaining it, e.g. from Aegis bridge or a file).
    on_progress : callable, optional
        Callback ``(completed, total, item_id, latency, endpoint_url)``
        invoked after each item completes.  Used by the monitor.
    """

    def __init__(
        self,
        config: ExaForgeConfig,
        pool: EndpointPool,
        on_progress: Optional[
            Callable[[int, int, str, float, str], Any]
        ] = None,
    ) -> None:
        self.config = config
        self.pool = pool
        self.on_progress = on_progress

        self.reader = get_reader(config.reader)
        self.task: BaseTask = get_task(config.task)
        self.writer: BaseWriter = get_writer(config.writer)
        self.checkpoint = CheckpointManager(config.checkpoint)
        self.client = InferenceClient(pool, config.client)

        self._completed = 0
        self._failed = 0
        self._total = 0
        self._start_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Execute the full pipeline in a streaming batch loop.

        Memory footprint is bounded to ``config.batch_size`` items at a
        time regardless of total dataset size:

        1. **Scan** — discover item IDs (cheap directory listing, no I/O).
        2. **Filter** — remove already-completed items via checkpoint.
        3. **Cap** — honour ``max_items``.
        4. **Batch loop** — for each slice of ``batch_size`` IDs:
              a. Load only those files from disk.
              b. Dispatch all items in the batch concurrently.
              c. Write outputs and save checkpoint.
              d. Release the batch (memory freed before next batch).

        Returns
        -------
        dict
            Summary statistics: total, completed, failed, elapsed.
        """
        self._start_time = time.monotonic()

        all_ids = self.reader.scan()
        logger.info("Scanned %d input items", len(all_ids))

        pending_ids = self.checkpoint.filter_pending(all_ids)

        if self.config.max_items > 0:
            pending_ids = pending_ids[: self.config.max_items]
            logger.info(
                "max_items=%d: capped to %d pending items",
                self.config.max_items,
                len(pending_ids),
            )

        self._total = len(pending_ids)
        self.checkpoint.total_items = len(all_ids)

        if not pending_ids:
            logger.info("All items already completed — nothing to do")
            return self._summary()

        skipped = len(all_ids) - self._total
        if skipped:
            logger.info(
                "Resuming: %d already done, %d pending",
                skipped,
                self._total,
            )

        batch_size = self.config.batch_size
        num_batches = (len(pending_ids) + batch_size - 1) // batch_size
        logger.info(
            "Processing %d items in %d batch(es) of up to %d",
            self._total,
            num_batches,
            batch_size,
        )

        try:
            for batch_num, start in enumerate(
                range(0, len(pending_ids), batch_size), start=1
            ):
                batch_ids = pending_ids[start : start + batch_size]
                logger.info(
                    "Batch %d/%d: loading %d items …",
                    batch_num,
                    num_batches,
                    len(batch_ids),
                )
                batch_items = self.reader.read_by_ids(set(batch_ids))
                await self._process_items(batch_items)
                self.checkpoint.save()
                logger.info(
                    "Batch %d/%d done. Progress: %d/%d completed, %d failed",
                    batch_num,
                    num_batches,
                    self._completed,
                    self._total,
                    self._failed,
                )
        finally:
            self.writer.close()
            self.checkpoint.save()
            await self.client.close()

        return self._summary()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process_items(self, items: list[InputItem]) -> None:
        """Send all items through the pipeline concurrently."""
        tasks = [self._process_one(item) for item in items]
        await asyncio.gather(*tasks)

    async def _process_one(self, item: InputItem) -> None:
        """Process a single input item end-to-end."""
        messages = self.task.prepare_messages(item)

        task_cfg = self.config.task
        request = ChatRequest(
            messages=messages,
            temperature=getattr(task_cfg, "temperature", 0.7),
            max_tokens=getattr(task_cfg, "max_tokens", 2000),
            top_p=getattr(task_cfg, "top_p", 1.0),
        )

        response = await self.client.generate(request)

        if response.success:
            parsed = self.task.parse_response(response.text)
            record = OutputRecord(
                id=item.id,
                response=response.text,
                metadata={**item.metadata, **parsed},
            )
            self.writer.write([record])
            self.checkpoint.mark_done(item.id)
            self._completed += 1
        else:
            logger.warning(
                "Failed item %s: %s", item.id, response.error
            )
            self._failed += 1

        if self.on_progress:
            self.on_progress(
                self._completed,
                self._total,
                item.id,
                response.latency,
                response.endpoint_url,
            )

        if self._completed % self.config.writer.buffer_size == 0:
            self.checkpoint.save()

    def _summary(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self._start_time
        return {
            "total": self._total,
            "completed": self._completed,
            "failed": self._failed,
            "elapsed_seconds": round(elapsed, 2),
            "items_per_second": round(
                self._completed / elapsed, 2
            )
            if elapsed > 0
            else 0,
        }
