"""Paper card-extraction task.

Adapts the model-card, agent-card, and data-card extraction prompts
from the paper-card-extraction repository into the ExaForge task
interface.  Each mode produces a structured YAML-compatible card
from a scientific paper's full text.
"""

from __future__ import annotations

from typing import Any

from exaforge.config import CardExtractionTaskConfig
from exaforge.readers.base import InputItem

from .base import BaseTask

_NEGATIVE_SENTINELS = {
    "model_card": "NO_MODEL_FOUND",
    "agent_card": "NO_AGENT_FOUND",
    "data_card": "NO_DATASET_FOUND",
}


class CardExtractionTask(BaseTask):
    """Extract model / agent / data cards from scientific papers."""

    def __init__(self, config: CardExtractionTaskConfig) -> None:
        self.config = config

    def prepare_messages(
        self, item: InputItem
    ) -> list[dict[str, str]]:
        prompt = self._build_prompt(item.text)
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def parse_response(self, raw: str) -> dict[str, Any]:
        sentinel = _NEGATIVE_SENTINELS.get(self.config.mode, "")
        detected = sentinel not in raw if sentinel else True
        return {
            "mode": self.config.mode,
            "card_detected": detected,
            "card_text": raw,
        }

    # ------------------------------------------------------------------
    # Prompt builders — adapted from paper-card-extraction
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        builders = {
            "model_card": self._model_card_prompt,
            "agent_card": self._agent_card_prompt,
            "data_card": self._data_card_prompt,
        }
        builder = builders.get(self.config.mode)
        if builder is None:
            raise ValueError(f"Unknown card mode: {self.config.mode}")
        return builder(text)

    # -- model card ----------------------------------------------------

    def _model_card_prompt(self, text: str) -> str:
        return f"""Please analyze the following scientific text to determine if it describes a machine learning model, AI system, neural network, or computational model that could benefit from a model card.

**FIRST: Determine if this text is about an ML/AI model**

A paper IS about an ML/AI model if it:
- Introduces, develops, or trains a new machine learning model
- Fine-tunes or adapts an existing model for a new task
- Presents a neural network architecture
- Describes a foundation model, language model, or vision model
- Presents computational models for prediction, classification, regression, generation
- Describes model training, validation, and evaluation procedures

A paper is NOT about an ML/AI model if it:
- Only uses existing models as tools without modification (e.g., "we used GPT-4 to analyze...")
- Is about traditional computational methods (molecular dynamics, DFT, etc.) without ML components
- Is purely experimental with no computational modeling
- Is a review paper that surveys models but doesn't introduce one

**If NO ML/AI model is described in this text, respond ONLY with:**
"NO_MODEL_FOUND: This text does not describe a machine learning or AI model suitable for a model card."

**If an ML/AI model IS described, extract the following information in YAML-compatible format:**

```yaml
# MODEL CARD EXTRACTION
model_detected: true
model_name: "[Name of the model]"
model_type: "[e.g., Transformer, CNN, GNN, RNN, Diffusion, etc.]"
language: "[e.g., en, multilingual]"
tags:
  - "[tag1: e.g., science:materials, science:biology, science:climate]"
license: "[SPDX license ID if mentioned, or UNKNOWN]"
base_model: "[Base/parent model if fine-tuned, or null]"
developed_by: "[Primary developers/authors]"
contributed_by: "[Contributing institutions/groups]"
short_description: "[One-line description]"
full_description: |
  [Detailed paragraph about the model architecture, purpose, and capabilities]
inputs:
  - type: "[e.g., text, image, time_series, molecular_structure]"
    format: "[e.g., SMILES, FASTA, PNG, tensor shape]"
    description: "[Details about input requirements]"
outputs:
  - type: "[e.g., classification, regression, generation, embedding]"
    format: "[Output format details]"
    description: "[What the model produces]"
training_data:
  - name: "[Dataset name]"
    size: "[Number of samples/size]"
    source: "[URL or description]"
training_procedure:
  optimizer: "[e.g., AdamW, SGD]"
  learning_rate: "[Value or schedule]"
  batch_size: "[Value]"
  epochs: "[Number]"
  hardware: "[GPUs/TPUs used]"
metrics:
  - name: "[Metric name, e.g., accuracy, F1, RMSE]"
    value: "[Reported value]"
    dataset: "[Evaluation dataset]"
software:
  framework: "[e.g., PyTorch, TensorFlow, JAX]"
  dependencies: "[Key libraries mentioned]"
intended_use:
  primary_uses: "[Main intended applications]"
  out_of_scope: "[Uses not recommended]"
limitations: |
  [Known limitations, biases, failure modes]
code_repository: "[GitHub/GitLab URL if mentioned]"
model_repository: "[HuggingFace URL if mentioned]"
paper_doi: "[DOI if mentioned]"
arxiv_id: "[arXiv ID if mentioned]"
contact:
  - name: "[Corresponding author]"
    email: "[Email if provided]"
```

**IMPORTANT GUIDELINES:**
- Only extract information that is explicitly stated or can be directly inferred
- Use "UNKNOWN" or "Not specified" for fields where information is not available
- Carefully search for ALL URLs in the text (GitHub, HuggingFace, DOIs, arXiv)

Text:
{text}"""

    # -- agent card ----------------------------------------------------

    def _agent_card_prompt(self, text: str) -> str:
        return f"""Please analyze the following scientific text to determine if it describes an AI agent, autonomous system, or agentic framework that could benefit from an agent card.

**FIRST: Determine if this text is about an AI agent**

A paper IS about an AI agent if it:
- Introduces an autonomous system that can take actions, use tools, or interact with environments
- Describes an LLM-based agent with planning, reasoning, or tool-use capabilities
- Presents a multi-agent system or agent framework
- Describes a system that can execute tasks, make decisions, or perform actions autonomously
- Introduces a conversational agent, chatbot, or assistant with specialized capabilities
- Describes a scientific workflow agent or lab automation agent

A paper is NOT about an AI agent if it:
- Only describes a static ML model without autonomous action capabilities
- Is about traditional software without AI-driven decision making
- Only uses agents as a metaphor (e.g., "cleaning agent" in chemistry)
- Is purely theoretical without describing an implemented system

**If NO AI agent is described in this text, respond ONLY with:**
"NO_AGENT_FOUND: This text does not describe an AI agent system suitable for an agent card."

**If an AI agent IS described, extract the following information in YAML-compatible format:**

```yaml
# AGENT CARD EXTRACTION
agent_detected: true
agent_name: "[Name of the agent]"
agent_type: "[e.g., LLM-based, Multi-agent, Tool-using, Conversational, Workflow, Scientific]"
language: "[e.g., en, multilingual]"
tags:
  - "[tag1: e.g., science:materials, science:biology, science:automation]"
license: "[SPDX license ID if mentioned, or UNKNOWN]"
provider:
  organization: "[Organization that developed/hosts the agent]"
  url: "[Base URL or endpoint if mentioned]"
developed_by: "[Primary developers/authors]"
short_description: "[One-line description of what the agent does]"
full_description: |
  [Detailed paragraph about the agent's purpose, architecture, and capabilities]
underlying_models:
  - name: "[Model name, e.g., GPT-4, Claude, Llama]"
    purpose: "[How the model is used in the agent]"
capabilities:
  tool_use: "[true/false]"
  planning: "[true/false]"
  memory: "[short-term/long-term/none/UNKNOWN]"
  multi_turn: "[true/false/UNKNOWN]"
skills:
  - name: "[Skill name]"
    description: "[What this skill does]"
tools:
  - name: "[Tool name]"
    purpose: "[What the tool does]"
inputs:
  - type: "[e.g., text, image, structured query]"
    format: "[e.g., natural language, JSON, API call]"
outputs:
  - type: "[e.g., text, action, artifact]"
    format: "[e.g., natural language, JSON, file]"
runtime:
  framework: "[e.g., LangChain, LangGraph, AutoGen, CrewAI, custom]"
evaluation:
  metrics:
    - name: "[Metric name]"
      value: "[Reported value]"
      task: "[Task being evaluated]"
intended_use:
  primary_uses: "[Main intended applications]"
  out_of_scope: "[Uses not recommended]"
limitations: |
  [Known limitations, failure modes, edge cases]
risks: |
  [Agent-specific risks: tool misuse, prompt injection, data handling]
code_repository: "[GitHub/GitLab URL if mentioned]"
paper_doi: "[DOI if mentioned]"
arxiv_id: "[arXiv ID if mentioned]"
contact:
  - name: "[Corresponding author]"
    email: "[Email if provided]"
```

**IMPORTANT GUIDELINES:**
- Only extract information that is explicitly stated or can be directly inferred
- Use "UNKNOWN" or "Not specified" for fields where information is not available
- Pay special attention to: tools/skills the agent can use, how it plans/reasons
- Carefully search for ALL URLs in the text

Text:
{text}"""

    # -- data card -----------------------------------------------------

    def _data_card_prompt(self, text: str) -> str:
        return f"""Please analyze the following scientific text to determine if it describes or introduces a dataset that could benefit from a data card.

**FIRST: Determine if this text introduces or describes a dataset**

A paper DOES introduce/describe a dataset if it:
- Introduces a new dataset created by the authors
- Describes a benchmark dataset for ML/AI evaluation
- Presents a curated collection of data (genomic, materials, simulation, experimental, etc.)
- Describes a database or data repository with structured data
- Releases training/evaluation data for models

A paper does NOT introduce a dataset if it:
- Only uses existing datasets without creating new ones
- Mentions datasets briefly without detailed description
- Is purely methodological without data contribution

**If NO dataset is introduced or described in this text, respond ONLY with:**
"NO_DATASET_FOUND: This text does not introduce or describe a dataset suitable for a data card."

**If a dataset IS introduced or described, extract the following information in YAML-compatible format:**

```yaml
# DATA CARD EXTRACTION
dataset_detected: true
dataset_name: "[Name of the dataset]"
dataset_type: "[e.g., Benchmark, Training, Simulation, Experimental, Observational]"
language: "[e.g., en, multilingual, N/A for non-text data]"
tags:
  - "[tag1: e.g., science:materials, science:biology, science:climate]"
  - "[tag2: e.g., modality:tabular, modality:image, modality:text]"
license: "[SPDX license ID if mentioned, or UNKNOWN]"
curated_by: "[Primary creators/curators]"
short_description: "[One-line description of the dataset]"
full_description: |
  [Detailed paragraph about the dataset's purpose, content, and significance]
dataset_details:
  domain: "[Scientific domain]"
  data_modality: "[tabular, image, text, time-series, graph, molecular, multi-modal]"
  file_formats: "[e.g., CSV, JSON, HDF5, Parquet]"
  size:
    num_samples: "[Number of samples/records]"
    file_size: "[Total size if mentioned]"
data_collection:
  methodology: |
    [How was the data collected?]
  source_data: "[Original data sources]"
data_processing:
  preprocessing: |
    [Cleaning, filtering, normalization steps]
  quality_control: "[QC procedures applied]"
access:
  repository: "[Where the data is hosted: URL, DOI]"
  access_type: "[open, restricted, upon-request, licensed]"
  download_url: "[Direct download link if available]"
intended_use:
  primary_uses: "[Main intended applications]"
  ml_tasks: "[Applicable ML tasks]"
  out_of_scope: "[Uses not recommended]"
limitations: |
  [Known limitations, biases, data quality issues]
paper_doi: "[DOI if mentioned]"
data_doi: "[DOI of dataset itself if different from paper DOI]"
code_repository: "[GitHub/GitLab URL if mentioned]"
contact:
  - name: "[Contact person]"
    email: "[Email if provided]"
```

**IMPORTANT GUIDELINES:**
- Only extract information that is explicitly stated or can be directly inferred
- Use "UNKNOWN" or "Not specified" for fields where information is not available
- Pay special attention to: data size, format, access method, and intended use
- Carefully search for ALL URLs in the text (data repositories, DOIs, code)

Text:
{text}"""
