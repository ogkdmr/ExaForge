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
- **Pluggable tasks** — registry-based task system. Ship with generic
  text generation and scientific-paper card extraction; add your own by
  implementing a small interface.
- **Lustre-aware I/O** — buffered JSONL writes, bulk reads, stripe tuning
  helpers, and atomic checkpointing designed for overwhelmed networked
  filesystems.
- **Checkpointing & resume** — interrupted jobs pick up where they left off.
- **Monitoring** — Rich progress bars, throughput counters, error tracking,
  and structured log files.
- **YAML configuration** — every knob is exposed in a single YAML file.

## Quick start

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run an inference job against existing Aegis endpoints
exaforge run --config configs/paper_cards.yaml

# Or let ExaForge launch Aegis for you
exaforge run --config configs/paper_cards.yaml   # with aegis.auto_launch: true
```

## Configuration

See `configs/` for annotated examples. A minimal config looks like:

```yaml
aegis:
  endpoints_file: aegis_endpoints.txt

task:
  name: generation
  system_prompt: "You are a helpful assistant."
  max_tokens: 2000

reader:
  name: text_directory
  input_dir: /lus/flare/projects/MyProject/papers
  glob_patterns: ["*.txt"]

writer:
  name: jsonl
  output_dir: /lus/flare/projects/MyProject/output

client:
  max_concurrent_requests: 64

checkpoint:
  enabled: true
```

## CLI reference

| Command | Description |
|---------|-------------|
| `exaforge run --config <yaml>` | Run a full inference pipeline |
| `exaforge launch --config <yaml>` | Launch Aegis endpoints only |
| `exaforge status --config <yaml>` | Show endpoint health and job progress |
| `exaforge merge --output-dir <dir>` | Merge shard JSONL files into one |

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

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/

# Type-check
mypy src/
```

## License

MIT
