# ExaForge

Distributed LLM inference on the Aurora supercomputer, powered by
[Aegis](https://github.com/your-org/Aegis) vLLM endpoints.

ExaForge takes an embarrassingly parallel inference workload, distributes
it across vLLM instances spawned by Aegis, and writes the results as JSONL —
all with Lustre-filesystem-aware I/O and built-in checkpointing.

## Features

- **Aegis integration** — automatically launch vLLM instances via Aegis and
  consume the resulting endpoints, or point at an existing endpoints file.
- **Async HTTP client** — high-throughput `httpx`-based client with
  configurable concurrency, retries, and load balancing across endpoints.
- **Pluggable tasks** — registry-based task system. Ships with generic
  text generation and scientific-paper card extraction; add your own by
  implementing a small interface.
- **Lustre-aware I/O** — buffered JSONL writes, bulk reads, stripe tuning
  helpers, and atomic checkpointing designed for overwhelmed networked
  filesystems.
- **Checkpointing & resume** — interrupted jobs pick up where they left off.
- **Monitoring** — Rich progress bars, throughput counters, error tracking,
  and structured JSONL log files.
- **YAML configuration** — every knob is exposed in a single YAML file.

## Installation

```bash
# Editable install with development extras
pip install -e ".[dev]"

# If you want Aegis auto-launch support
pip install -e ".[aegis]"
```

**Dependencies** (installed automatically):

| Package | Purpose |
|---------|---------|
| `httpx` | Async HTTP client for vLLM endpoints |
| `pydantic` | Configuration validation |
| `pyyaml` | YAML config loading |
| `typer` | CLI framework |
| `rich` | Progress bars and formatted output |

## Quick start

### 1. Using existing Aegis endpoints

If you already ran `aegis submit` and have an `aegis_endpoints.txt` file:

```bash
exaforge run --config configs/generic_generation.yaml
```

### 2. Letting ExaForge launch Aegis

Set `aegis.auto_launch: true` in your config and provide the path to your
Aegis YAML:

```bash
exaforge run --config configs/paper_cards.yaml
```

ExaForge will submit the PBS job, wait for endpoints to become healthy,
then start processing.

### 3. Resuming an interrupted job

Just re-run the same command. ExaForge reads the checkpoint file and
skips already-completed items:

```bash
exaforge run --config configs/paper_cards.yaml
# Output: "Resuming: 12345 already done, 47655 pending"
```

## Configuration

All configuration lives in a single YAML file. See `configs/` for
annotated examples.

### Top-level structure

```yaml
aegis:        # How to obtain vLLM endpoints
task:         # What inference task to run
reader:       # Where to read input data
writer:       # Where and how to write output
client:       # HTTP client settings
monitor:      # Progress display and logging
checkpoint:   # Resume support
```

### Aegis section

```yaml
aegis:
    config_path: /path/to/aegis_config.yaml  # Aegis YAML (required if auto_launch)
    auto_launch: true                         # Submit PBS job automatically
    wait_for_endpoints: true                  # Block until endpoints healthy
    endpoints_file: aegis_endpoints.txt       # Path to endpoints file
```

### Task section

ExaForge supports pluggable tasks selected by `name`:

**Generic generation** (`name: generation`):

```yaml
task:
    name: generation
    system_prompt: "You are a helpful assistant."
    temperature: 0.7
    max_tokens: 2000
    top_p: 1.0
```

**Card extraction** (`name: card_extraction`):

```yaml
task:
    name: card_extraction
    mode: model_card          # model_card | agent_card | data_card
    system_prompt: "You are a helpful assistant..."
    temperature: 0.3
    max_tokens: 4000
```

### Reader section

**Text directory** (`name: text_directory`):

```yaml
reader:
    name: text_directory
    input_dir: /lus/flare/projects/MyProject/papers
    glob_patterns: ["*.txt"]
```

**JSONL** (`name: jsonl`):

```yaml
reader:
    name: jsonl
    input_dir: /lus/flare/projects/MyProject/data
    glob_patterns: ["*.jsonl"]
    text_field: text    # JSON key containing the input text
    id_field: id        # JSON key for unique item IDs
```

### Writer section

```yaml
writer:
    name: jsonl
    output_dir: /lus/flare/projects/MyProject/output
    buffer_size: 500     # Records buffered before flushing to Lustre
    base_name: results   # Output file prefix
```

### Client section

```yaml
client:
    max_concurrent_requests: 64    # Global in-flight cap across ALL endpoints.
                                    # Rule of thumb: num_endpoints × desired_queue_depth
                                    # e.g. 12 endpoints × 16 = 192. Default 64 suits small tests.
    timeout: 300                    # Per-request timeout (seconds)
    max_retries: 3                  # Retry count on failure
    retry_backoff: 2.0              # Exponential back-off base
    load_balance_strategy: round_robin  # round_robin | least_loaded
```

### Monitor section

```yaml
monitor:
    log_file: run.log          # Structured JSONL log (optional)
    progress_interval: 30      # Seconds between console reports
    enable_rich: true          # Rich progress bar
```

### Checkpoint section

```yaml
checkpoint:
    enabled: true
    checkpoint_file: checkpoint.json
```

## CLI reference

```
exaforge run     --config <yaml>           Run a full inference pipeline
exaforge launch  --config <yaml>           Launch Aegis endpoints only
exaforge status  --config <yaml>           Show endpoint health + progress
exaforge merge   <output_dir> [--output]   Merge shard JSONL files
```

### Examples

```bash
# Run card extraction
exaforge run --config configs/paper_cards.yaml --verbose

# Check endpoint health
exaforge status --config configs/paper_cards.yaml

# Merge output shards
exaforge merge /lus/flare/projects/MyProject/output --output merged.jsonl
```

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────┐
│  YAML Config │────▶│               Orchestrator                   │
└──────────────┘     │                                              │
                     │  Reader ──▶ Task ──▶ Client ──▶ Writer       │
                     │              │         │                     │
                     │          Checkpoint  Monitor                 │
                     └──────────────────────────────────────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                     vLLM :8000  vLLM :8001  vLLM :800N
                     (Aegis-spawned on Aurora compute nodes)
```

### Module overview

| Module | Responsibility |
|--------|---------------|
| `config.py` | Pydantic config models with YAML serde |
| `readers/` | Bulk-load input data (text files, JSONL) |
| `writers/` | Buffered JSONL output with fsync |
| `endpoints.py` | Load Aegis endpoints, health check, load balance |
| `client.py` | Async HTTP client for OpenAI chat completions |
| `tasks/` | Pluggable task definitions (generation, card extraction) |
| `checkpoint.py` | Atomic checkpoint persistence for resume |
| `orchestrator.py` | Main async pipeline |
| `monitor.py` | Rich progress, throughput, structured logging |
| `aegis_bridge.py` | Programmatic Aegis launch and wait |
| `cli.py` | Typer CLI entry point |
| `lustre.py` | Lustre-aware I/O utilities |

### Adding a custom task

1. Create a new file in `src/exaforge/tasks/`, e.g. `my_task.py`
2. Implement `BaseTask`:

```python
from exaforge.tasks.base import BaseTask
from exaforge.readers.base import InputItem

class MyTask(BaseTask):
    def __init__(self, config):
        self.config = config

    def prepare_messages(self, item: InputItem) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "Your system prompt"},
            {"role": "user", "content": item.text},
        ]

    def parse_response(self, raw: str) -> dict[str, Any]:
        return {"result": raw}
```

3. Add a config class in `config.py` and register in `tasks/__init__.py`

## Lustre filesystem considerations

ExaForge is designed to work well on Lustre parallel filesystems like
*flare* on Aurora:

- **Bulk reads** — input directories are scanned once; file contents are
  read into memory to avoid repeated metadata operations.
- **Buffered writes** — output records are accumulated in memory (default
  500) and flushed in a single sequential write with `fsync`.
- **Atomic checkpoints** — the checkpoint file is written to a temp file
  then renamed (`os.rename` is atomic on Lustre).
- **Stripe tuning** — `lustre.set_stripe()` and `lustre.ensure_output_dir()`
  can configure Lustre striping on output directories.
- **Minimal file contention** — each output shard is a separate file;
  use `exaforge merge` to combine them.

## Development

```bash
# Run all tests
PYTHONPATH="src" pytest -v

# Run a specific test module
PYTHONPATH="src" pytest tests/test_client.py -v

# Lint
ruff check src/ tests/

# Type-check
mypy src/
```

## License

MIT
