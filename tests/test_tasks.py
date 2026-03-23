"""Tests for the ExaForge task system."""

from __future__ import annotations

import pytest

from exaforge.config import CardExtractionTaskConfig, GenerationTaskConfig
from exaforge.readers.base import InputItem
from exaforge.tasks import (
    CardExtractionTask,
    GenerationTask,
    get_task,
)


def _item(text: str = "Sample paper text.") -> InputItem:
    return InputItem(id="test-1", text=text)


class TestGenerationTask:
    def test_prepare_messages(self) -> None:
        cfg = GenerationTaskConfig(system_prompt="Be helpful.")
        task = GenerationTask(cfg)
        msgs = task.prepare_messages(_item("Hello"))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be helpful."
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"

    def test_parse_response(self) -> None:
        cfg = GenerationTaskConfig()
        task = GenerationTask(cfg)
        result = task.parse_response("Generated text here.")
        assert result["generated_text"] == "Generated text here."


class TestCardExtractionTask:
    @pytest.mark.parametrize(
        "mode", ["model_card", "agent_card", "data_card"]
    )
    def test_prepare_messages_all_modes(self, mode: str) -> None:
        cfg = CardExtractionTaskConfig(mode=mode)
        task = CardExtractionTask(cfg)
        msgs = task.prepare_messages(_item("Some paper text."))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Some paper text." in msgs[1]["content"]

    def test_model_card_prompt_contains_keywords(self) -> None:
        cfg = CardExtractionTaskConfig(mode="model_card")
        task = CardExtractionTask(cfg)
        msgs = task.prepare_messages(_item())
        prompt = msgs[1]["content"]
        assert "MODEL CARD EXTRACTION" in prompt
        assert "NO_MODEL_FOUND" in prompt

    def test_agent_card_prompt_contains_keywords(self) -> None:
        cfg = CardExtractionTaskConfig(mode="agent_card")
        task = CardExtractionTask(cfg)
        msgs = task.prepare_messages(_item())
        prompt = msgs[1]["content"]
        assert "AGENT CARD EXTRACTION" in prompt
        assert "NO_AGENT_FOUND" in prompt

    def test_data_card_prompt_contains_keywords(self) -> None:
        cfg = CardExtractionTaskConfig(mode="data_card")
        task = CardExtractionTask(cfg)
        msgs = task.prepare_messages(_item())
        prompt = msgs[1]["content"]
        assert "DATA CARD EXTRACTION" in prompt
        assert "NO_DATASET_FOUND" in prompt

    def test_parse_response_detected(self) -> None:
        cfg = CardExtractionTaskConfig(mode="model_card")
        task = CardExtractionTask(cfg)
        result = task.parse_response("model_detected: true\nmodel_name: MyModel")
        assert result["card_detected"] is True
        assert result["mode"] == "model_card"
        assert "card_text" in result

    def test_parse_response_not_detected(self) -> None:
        cfg = CardExtractionTaskConfig(mode="model_card")
        task = CardExtractionTask(cfg)
        result = task.parse_response("NO_MODEL_FOUND: This text does not describe a model.")
        assert result["card_detected"] is False

    def test_parse_agent_not_detected(self) -> None:
        cfg = CardExtractionTaskConfig(mode="agent_card")
        task = CardExtractionTask(cfg)
        result = task.parse_response("NO_AGENT_FOUND: Not an agent paper.")
        assert result["card_detected"] is False

    def test_parse_data_not_detected(self) -> None:
        cfg = CardExtractionTaskConfig(mode="data_card")
        task = CardExtractionTask(cfg)
        result = task.parse_response("NO_DATASET_FOUND: No dataset here.")
        assert result["card_detected"] is False

    def test_unknown_mode_raises(self) -> None:
        cfg = CardExtractionTaskConfig.__new__(CardExtractionTaskConfig)
        object.__setattr__(cfg, "mode", "unknown_card")
        object.__setattr__(cfg, "system_prompt", "test")
        task = CardExtractionTask(cfg)
        with pytest.raises(ValueError, match="Unknown card mode"):
            task.prepare_messages(_item())

    def test_system_prompt_customisable(self) -> None:
        cfg = CardExtractionTaskConfig(
            system_prompt="Custom system prompt."
        )
        task = CardExtractionTask(cfg)
        msgs = task.prepare_messages(_item())
        assert msgs[0]["content"] == "Custom system prompt."


class TestTaskRegistry:
    def test_get_generation_task(self) -> None:
        task = get_task({"name": "generation"})
        assert isinstance(task, GenerationTask)

    def test_get_card_extraction_task(self) -> None:
        task = get_task({"name": "card_extraction", "mode": "data_card"})
        assert isinstance(task, CardExtractionTask)

    def test_get_task_from_config(self) -> None:
        cfg = GenerationTaskConfig(max_tokens=100)
        task = get_task(cfg)
        assert isinstance(task, GenerationTask)

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            get_task({"name": "summarize"})
