"""Tests for the Q/A generation task."""

from __future__ import annotations

import json

import pytest

from exaforge.config import QAGenerationTaskConfig
from exaforge.readers.base import InputItem
from exaforge.tasks import QAGenerationTask, get_task
from exaforge.tasks.base import ItemSkipped


def _item(
    text: str = "Once upon a time there was a brave knight.",
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

    def test_too_long_raises_item_skipped(self) -> None:
        cfg = QAGenerationTaskConfig(max_input_tokens=10)
        task = QAGenerationTask(cfg)
        # 100 chars / 4 = 25 tokens > 10 limit
        long_text = "x" * 100
        with pytest.raises(ItemSkipped, match="tokens"):
            task.prepare_messages(_item(text=long_text))

    def test_within_token_limit_succeeds(self) -> None:
        cfg = QAGenerationTaskConfig(max_input_tokens=110000)
        task = QAGenerationTask(cfg)
        msgs = task.prepare_messages(_item())
        assert len(msgs) == 2

    def test_parse_response_valid_json(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        raw = json.dumps([
            {
                "qa_index": 1,
                "question_type": "Who",
                "question": "Who was the knight?",
                "passage": "a brave knight",
            },
            {
                "qa_index": 2,
                "question_type": "Where",
                "question": "Where did the story take place?",
                "passage": "Once upon a time",
            },
        ])
        result = task.parse_response(raw)
        assert result["num_questions"] == 2
        assert len(result["qa_pairs"]) == 2
        assert result["qa_pairs"][0]["question_type"] == "Who"

    def test_parse_response_code_fence(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        raw = '```json\n[{"qa_index": 1, "question": "Q?", "passage": "A", "question_type": "What"}]\n```'
        result = task.parse_response(raw)
        assert result["num_questions"] == 1

    def test_parse_response_with_surrounding_text(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        raw = 'Here are the questions:\n[{"qa_index": 1, "question": "Q?", "passage": "A", "question_type": "How"}]\nDone!'
        result = task.parse_response(raw)
        assert result["num_questions"] == 1

    def test_parse_response_invalid_json(self) -> None:
        cfg = QAGenerationTaskConfig()
        task = QAGenerationTask(cfg)
        result = task.parse_response("This is not JSON at all.")
        assert result["qa_pairs"] == []
        assert result["num_questions"] == 0

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
