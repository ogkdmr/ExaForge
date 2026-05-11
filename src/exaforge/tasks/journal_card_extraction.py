"""Journal / publication card-extraction task.

Extracts structured publication metadata from scientific papers
using the YAML template defined in ``prompt.yaml``.  The template
covers study design, genome information, metagenome samples,
metabolites, sequencing details, bioinformatics tools, results,
and database accessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exaforge.config import JournalCardExtractionTaskConfig
from exaforge.readers.base import InputItem

from .base import BaseTask

_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompt.yaml"

_NEGATIVE_SENTINEL = "NO_PUBLICATION_FOUND"


class JournalCardExtractionTask(BaseTask):
    """Extract journal / publication cards from scientific papers."""

    def __init__(self, config: JournalCardExtractionTaskConfig) -> None:
        self.config = config
        self._prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def prepare_messages(
        self, item: InputItem
    ) -> list[dict[str, str]]:
        prompt = self._build_prompt(item.text)
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def parse_response(self, raw: str) -> dict[str, Any]:
        detected = _NEGATIVE_SENTINEL not in raw
        return {
            "card_detected": detected,
            "card_text": raw,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompt_template() -> str:
        """Read the YAML card template from disk once at init time."""
        return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    def _build_prompt(self, text: str) -> str:
        return f"""Please analyze the following scientific text to determine if it describes a scientific study or publication that could benefit from a journal/publication card.

**FIRST: Determine if this text is a scientific publication**

A text IS a scientific publication if it:
- Presents original research with methods, results, and conclusions
- Describes experimental or computational studies in any scientific domain
- Reports findings from omics studies (genomics, metagenomics, transcriptomics, proteomics, metabolomics)
- Describes clinical, observational, or epidemiological studies
- Presents bioinformatics analyses, pipelines, or workflows
- Reports sequencing data, genome assemblies, or metagenome analyses

A text is NOT a scientific publication if it:
- Is a news article, blog post, or opinion piece without primary data
- Is a table of contents, index, or reference list only
- Contains insufficient content to extract meaningful metadata

**If NO scientific publication is described in this text, respond ONLY with:**
"{_NEGATIVE_SENTINEL}: This text does not describe a scientific publication suitable for a journal/publication card."

**If a scientific publication IS described, extract the following information in YAML-compatible format:**

{self._prompt_template}

**IMPORTANT GUIDELINES:**
- Only extract information that is explicitly stated or can be directly inferred from the text
- Use "UNKNOWN" or "Not specified" for fields where information is not available
- Omit entire sections (e.g., metagenome, metabolite) if they are not relevant to the study
- Pay special attention to: accession numbers, database identifiers, bioinformatics tools and versions
- Carefully search for ALL accession numbers (SRA, BioProject, BioSample, GEO, ENA, PDB)
- Extract all mentioned software tools with their versions when available

Text:
{text}"""
