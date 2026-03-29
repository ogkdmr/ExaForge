"""Shared test fixtures for ExaForge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture()
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def sample_texts(tmp_dir: Path) -> list[Path]:
    """Create a handful of sample .txt files and return their paths."""
    files = []
    for i in range(5):
        p = tmp_dir / f"paper_{i:03d}.txt"
        p.write_text(f"This is the full text of sample paper {i}.\n")
        files.append(p)
    return files


@pytest.fixture()
def sample_jsonl(tmp_dir: Path) -> Path:
    """Create a sample JSONL file and return its path."""
    p = tmp_dir / "inputs.jsonl"
    records = [
        {"id": f"doc-{i}", "text": f"Sample document text number {i}."}
        for i in range(10)
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


@pytest.fixture()
def endpoints_file(tmp_dir: Path) -> Path:
    """Create a mock Aegis endpoints file."""
    p = tmp_dir / "aegis_endpoints.txt"
    p.write_text("node1.hsn.cm.aurora.alcf.anl.gov:8000\nnode2.hsn.cm.aurora.alcf.anl.gov:8000\n")
    return p


@pytest.fixture()
def large_sample_texts(tmp_dir: Path) -> list[Path]:
    """Create 20 sample .txt files — useful for testing multi-batch behaviour."""
    files = []
    for i in range(20):
        p = tmp_dir / f"doc_{i:03d}.txt"
        p.write_text(f"Full text of large-corpus document {i}.\n")
        files.append(p)
    return files


@pytest.fixture()
def sample_mmd_dir(tmp_dir: Path) -> Path:
    """Create a directory with 12 .mmd files for preprocessing tests."""
    mmd_dir = tmp_dir / "papers"
    mmd_dir.mkdir()
    for i in range(12):
        p = mmd_dir / f"paper_{i:03d}.mmd"
        p.write_text(f"# Paper {i}\n\nThis is the parsed markdown for paper {i}.\n")
    return mmd_dir


@pytest.fixture()
def preprocessed_jsonl_dir(tmp_dir: Path, sample_mmd_dir: Path) -> Path:
    """Run jsonl preprocessing on sample_mmd_dir and return the output dir."""
    import zipfile

    from exaforge.preprocess import preprocess_to_jsonl

    out = tmp_dir / "preprocessed_jsonl"
    preprocess_to_jsonl(sample_mmd_dir, out, ["*.mmd"], batch_size=5)
    return out


@pytest.fixture()
def preprocessed_zip_dir(tmp_dir: Path, sample_mmd_dir: Path) -> Path:
    """Run zip preprocessing on sample_mmd_dir and return the output dir."""
    from exaforge.preprocess import preprocess_to_zip

    out = tmp_dir / "preprocessed_zip"
    preprocess_to_zip(sample_mmd_dir, out, ["*.mmd"], batch_size=5)
    return out
