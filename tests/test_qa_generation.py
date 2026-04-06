"""Tests for the Q/A generation task."""

from __future__ import annotations

import json

import pytest

from exaforge.config import QAGenerationTaskConfig
from exaforge.readers.base import InputItem
from exaforge.tasks import QAGenerationTask, get_task
from exaforge.tasks.base import ItemSkipped
from exaforge.writers.base import OutputRecord

_NOVEL_TEXT = "Once upon a time there was a brave knight. " * 200  # >5000 chars


def _item(
    text: str = _NOVEL_TEXT,
    item_id: str = "novels_chunk_aa_1",
    source_zim: str = "gutenberg.zim",
    title: str = "The Brave Knight",
) -> InputItem:
    return InputItem(
        id=item_id,
        text=text,
        metadata={
            "source_file": "/data/novels_chunk_aa.jsonl",
            "line_number": 1,
            "original_record": {
                "text": text,
                "source_zim": source_zim,
                "title": title,
            },
        },
    )


def _narrative_response(n: int = 2) -> str:
    pairs = [
        {
            "qa_index": i,
            "question_type": "Who",
            "question": f"Question {i}?",
            "passage": f"passage {i}",
        }
        for i in range(1, n + 1)
    ]
    return json.dumps({"content_type": "narrative", "qa_pairs": pairs})


class TestQAGenerationTask:
    def test_prepare_messages_basic(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        msgs = task.prepare_messages(_item())
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "brave knight" in msgs[1]["content"]

    def test_prompt_includes_question_count(self) -> None:
        cfg = QAGenerationTaskConfig(questions_per_novel=30)
        task = QAGenerationTask(cfg)
        msgs = task.prepare_messages(_item())
        assert "exactly 30" in msgs[1]["content"]

    def test_prompt_includes_question_types(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        msgs = task.prepare_messages(_item())
        prompt = msgs[1]["content"]
        for qtype in ("Who", "Why", "When", "How", "What", "Where"):
            assert qtype in prompt

    def test_prompt_includes_content_type_instruction(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        msgs = task.prepare_messages(_item())
        assert "content_type" in msgs[1]["content"]
        assert "metadata" in msgs[1]["content"]

    # -- too-long filter ---------------------------------------------------

    def test_too_long_raises_item_skipped(self) -> None:
        cfg = QAGenerationTaskConfig(max_input_tokens=10, min_text_chars=0)
        task = QAGenerationTask(cfg)
        long_text = "x" * 100  # 100 chars / 4 = 25 tokens > 10 limit
        with pytest.raises(ItemSkipped) as exc_info:
            task.prepare_messages(_item(text=long_text))
        assert exc_info.value.skip_type == "too_long"
        assert "tokens" in exc_info.value.reason

    # -- too-short filter --------------------------------------------------

    def test_too_short_raises_item_skipped(self) -> None:
        cfg = QAGenerationTaskConfig(min_text_chars=5000)
        task = QAGenerationTask(cfg)
        short_text = '"Brave Knight" (cover) English Project Gutenberg'
        with pytest.raises(ItemSkipped) as exc_info:
            task.prepare_messages(_item(text=short_text))
        assert exc_info.value.skip_type == "too_short"
        assert "chars" in exc_info.value.reason

    def test_short_filter_checked_before_long_filter(self) -> None:
        # A text that is both short AND would exceed token limit — should
        # surface as too_short, not too_long.
        cfg = QAGenerationTaskConfig(min_text_chars=50, max_input_tokens=1)
        task = QAGenerationTask(cfg)
        with pytest.raises(ItemSkipped) as exc_info:
            task.prepare_messages(_item(text="hi"))
        assert exc_info.value.skip_type == "too_short"

    def test_within_limits_succeeds(self) -> None:
        cfg = QAGenerationTaskConfig(max_input_tokens=110000, min_text_chars=5)
        task = QAGenerationTask(cfg)
        msgs = task.prepare_messages(_item())
        assert len(msgs) == 2

    # -- parse_response ----------------------------------------------------

    def test_parse_response_narrative(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        result = task.parse_response(_narrative_response(2))
        assert result["content_type"] == "narrative"
        assert result["extraction_successful"] is True
        assert result["num_questions"] == 2
        assert len(result["qa_pairs"]) == 2
        assert result["qa_pairs"][0]["question_type"] == "Who"

    def test_parse_response_metadata_content_type(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        raw = json.dumps({"content_type": "metadata", "qa_pairs": []})
        result = task.parse_response(raw)
        assert result["content_type"] == "metadata"
        assert result["extraction_successful"] is False
        assert result["num_questions"] == 0

    def test_parse_response_code_fence(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        inner = json.dumps({"content_type": "narrative", "qa_pairs": [
            {"qa_index": 1, "question": "Q?", "passage": "A", "question_type": "What"}
        ]})
        raw = f"```json\n{inner}\n```"
        result = task.parse_response(raw)
        assert result["num_questions"] == 1
        assert result["extraction_successful"] is True

    def test_parse_response_with_surrounding_text(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        inner = json.dumps({"content_type": "narrative", "qa_pairs": [
            {"qa_index": 1, "question": "Q?", "passage": "A", "question_type": "How"}
        ]})
        raw = f"Here is the output:\n{inner}\nDone!"
        result = task.parse_response(raw)
        assert result["num_questions"] == 1

    def test_parse_response_invalid_json(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        result = task.parse_response("This is not JSON at all.")
        assert result["content_type"] == "unknown"
        assert result["extraction_successful"] is False
        assert result["qa_pairs"] == []
        assert result["num_questions"] == 0

    # -- build_records (fan-out) -------------------------------------------

    def test_build_records_fan_out(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        item = _item()
        parsed = task.parse_response(_narrative_response(3))
        records = task.build_records(item, "", parsed)
        assert len(records) == 3
        assert all(isinstance(r, OutputRecord) for r in records)
        # IDs are novel_id + qa_index
        assert records[0].id == "novels_chunk_aa_1_q01"
        assert records[1].id == "novels_chunk_aa_1_q02"

    def test_build_records_fields_are_flat(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        item = _item()
        parsed = task.parse_response(_narrative_response(1))
        records = task.build_records(item, "", parsed)
        assert len(records) == 1
        meta = records[0].metadata
        # Provenance fields present at top level
        assert meta["novel_id"] == "novels_chunk_aa_1"
        assert meta["source_zim"] == "gutenberg.zim"
        assert meta["title"] == "The Brave Knight"
        # Q/A fields present at top level — no nesting needed
        assert "question" in meta
        assert "passage" in meta
        assert "question_type" in meta
        assert "qa_index" in meta
        # No raw qa_pairs list
        assert "qa_pairs" not in meta

    def test_build_records_failed_extraction_one_record(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        item = _item()
        parsed = task.parse_response(
            json.dumps({"content_type": "metadata", "qa_pairs": []})
        )
        records = task.build_records(item, "raw response", parsed)
        assert len(records) == 1
        assert records[0].metadata["extraction_successful"] is False
        assert records[0].metadata["num_questions"] == 0

    # -- metadata extraction -----------------------------------------------

    def test_extract_item_metadata(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        item = _item()
        meta = task.extract_item_metadata(item)
        assert meta["novel_id"] == "novels_chunk_aa_1"
        assert meta["source_zim"] == "gutenberg.zim"
        assert meta["title"] == "The Brave Knight"

    def test_extract_item_metadata_missing_fields(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        item = InputItem(id="test_1", text="text", metadata={})
        meta = task.extract_item_metadata(item)
        assert meta["novel_id"] == "test_1"
        assert meta["source_zim"] == ""
        assert meta["title"] == ""

    def test_system_prompt_customisable(self) -> None:
        cfg = QAGenerationTaskConfig(system_prompt="Custom prompt.")
        task = QAGenerationTask(cfg)
        msgs = task.prepare_messages(_item())
        assert msgs[0]["content"] == "Custom prompt."


class TestQAGenerationRegistry:
    def test_get_task_from_dict(self) -> None:
        task = get_task({"name": "qa_generation"})
        assert isinstance(task, QAGenerationTask)

    def test_get_task_from_config(self) -> None:
        cfg = QAGenerationTaskConfig(questions_per_novel=15)
        task = get_task(cfg)
        assert isinstance(task, QAGenerationTask)
