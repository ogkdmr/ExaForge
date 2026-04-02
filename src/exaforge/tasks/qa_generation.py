"""Novel Q/A generation task.

Generates reading-comprehension questions from fiction novels for
pretraining Mamba-based state-space models.  Each novel produces
a set of (question, passage, metadata) triples where the passage
is a verbatim clip from the original text that answers the question.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from exaforge.config import QAGenerationTaskConfig
from exaforge.readers.base import InputItem

from .base import BaseTask, ItemSkipped

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio for English prose.
_CHARS_PER_TOKEN = 4


class QAGenerationTask(BaseTask):
    """Generate Who/Why/When/How/What/Where questions from novels."""

    def __init__(self, config: QAGenerationTaskConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def prepare_messages(
        self, item: InputItem
    ) -> list[dict[str, str]]:
        text = item.text

        if len(text) < self.config.min_text_chars:
            raise ItemSkipped(
                f"Novel {item.id} is only {len(text)} chars "
                f"(min {self.config.min_text_chars}) — likely metadata/cover page",
                skip_type="too_short",
            )

        estimated_tokens = len(text) // _CHARS_PER_TOKEN
        if estimated_tokens > self.config.max_input_tokens:
            raise ItemSkipped(
                f"Novel {item.id} is ~{estimated_tokens} tokens "
                f"(limit {self.config.max_input_tokens})",
                skip_type="too_long",
            )

        n = self.config.questions_per_novel
        user_prompt = self._build_prompt(text, n)

        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the model's JSON response object."""
        parsed = self._extract_json(raw)

        content_type = parsed.get("content_type", "unknown")
        qa_pairs = parsed.get("qa_pairs", [])
        num_questions = len(qa_pairs)
        extraction_successful = (
            content_type == "narrative" and num_questions > 0
        )

        if not extraction_successful:
            logger.warning(
                "Extraction unsuccessful: content_type=%r, num_questions=%d",
                content_type,
                num_questions,
            )

        return {
            "content_type": content_type,
            "extraction_successful": extraction_successful,
            "qa_pairs": qa_pairs,
            "num_questions": num_questions,
        }

    def extract_item_metadata(self, item: InputItem) -> dict[str, Any]:
        """Promote source_zim and title to top-level output fields."""
        original = item.metadata.get("original_record", {})
        return {
            "novel_id": item.id,
            "source_zim": original.get("source_zim", ""),
            "title": original.get("title", ""),
        }

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str, n: int) -> str:
        return f"""You are given the full text of a fiction novel below. Your task is to generate exactly {n} reading-comprehension questions that can be answered from the novel.

**Step 1 — Assess the content.**

First, decide whether the text is genuine narrative prose (a novel, short story, or play with plot, characters, and dialogue). If it is only a cover page, table of contents, bibliographic notice, index, or other non-narrative material, you must still return valid JSON but with an empty qa_pairs array and content_type set to "metadata".

**Step 2 — Generate questions (narrative content only).**

If the text is narrative:
1. Generate a diverse mix of question types: Who, Why, When, How, What, Where.
2. Questions should span the entire novel — cover early, middle, and late sections.
3. For each question, extract a **verbatim passage** from the novel text that answers the question. This passage must be copied exactly from the text — do not paraphrase or modify it.
4. The passage should be long enough to fully answer the question (typically 1-5 sentences) but not excessively long.
5. Each Q/A pair has a **qa_index** (starting from 1) that reflects the temporal order in which the answering passages appear in the novel. Q/A pair 1 should reference a passage from early in the text, Q/A pair {n} from late in the text.

**Output format:**

Return a single JSON object (not an array) with these top-level fields:
- "content_type": either "narrative" (genuine story/play/novel) or "metadata" (cover, TOC, bibliographic notice, etc.)
- "qa_pairs": array of exactly {n} objects for narrative content, or an empty array [] for metadata

Each object in qa_pairs must have:
- "qa_index": integer (1 to {n}), temporal order of the passage in the novel
- "question_type": one of "Who", "Why", "When", "How", "What", "Where"
- "question": the question string
- "passage": the verbatim clip from the novel that answers the question

Return ONLY the JSON object, no other text before or after it.

**Novel text:**

{text}"""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Best-effort extraction of a JSON object from the model output."""
        stripped = raw.strip()

        # Try direct parse first.
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # Try to find a JSON object in markdown code fences.
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL
        )
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Last resort: find the outermost { ... } span.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse Q/A JSON from model response")
        return {"content_type": "unknown", "qa_pairs": []}
