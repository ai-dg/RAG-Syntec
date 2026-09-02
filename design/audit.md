# Audit — baseline of the existing system

> Recorded at commit `4197fce` (2026-09-02 17:36:29 +0200), working tree clean
> before this measurement. Every number below is next to the exact command
> that produced it, so it is re-derivable, not just asserted — per
> the project's verification discipline.

## Test suite

Command: `uv run pytest -q`

```
7 passed, 2 warnings in 1.73s
```

Two `DeprecationWarning`s (langchain-community sunset notice, an
`asyncio.iscoroutinefunction` deprecation from chromadb's telemetry module) —
not failures, recorded for completeness.

## Hardware

Command: `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv`

```
NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 29 MiB
```

Command: `free -g`

```
               total        used        free      shared  buff/cache   available
Mem:              15           9           0           1           6           5
Swap:             24           0          24
```

Command: `nproc` → `32`

## Ollama models available

Command: `curl -s localhost:11434/api/tags`

| Model | Size | Params | Quant |
|---|---|---|---|
| `qwen3-embedding:8b` | 4.68 GB | 7.6B | Q4_K_M |
| `gemma4:latest` | 9.61 GB | 8.0B | Q4_K_M |
| `llama3.2:3b` / `llama3.2:latest` | 2.02 GB | 3.2B | Q4_K_M |
| `qwen2.5-coder:7b` / `qwen2.5-coder:latest` | 4.68 GB | 7.6B | Q4_K_M |

**Finding, not yet acted on:** the models currently configured in `.env`
(`EMBEDDING_MODEL_LOCAL=qwen3-embedding:8b`, `CHAT_MODEL_LOCAL=gemma4:latest`)
sum to 4.68 GB + 9.61 GB = **14.29 GB**, which exceeds the 8188 MiB (≈ 8 GB)
of VRAM on this machine on their own — before accounting for KV cache or any
concurrent process. `gemma4:latest` alone (9.61 GB) already exceeds total
VRAM. This is recorded as a fact of the current configuration; whether Ollama
is actually offloading to CPU, and what that costs in latency, is not
measured here — that is a separate, later question (relevant to the T7.1
BGE-M3 + `llama3.2:3b` VRAM arithmetic in `TODO.md`).

## Chunk count (load + split, no embedding)

Command: a small script calling `app.services.ingestion.load_docs` then
`chunk_text`, using the current `.env` (`DOCS_DIR=docs`, `CHUNK_SIZE=500`,
`CHUNK_OVERLAP=100`), stopping before any embedding call — see
`app/services/ingestion.py:64` (`chunk_text`) and `:116` (`load_docs`).

```
docs loaded: 109
chunks: 4183
chunk_size=500 chunk_overlap=100 docs_dir=docs
```

`docs/` currently holds 109 Markdown files (1.7 MB total) — this is already
Syntec-shaped content (`ls docs/ | wc -l`, `du -sh docs/`), not the original
6-file smoke fixture the repo started with. Phase 2 (`T2.1`) still needs to
formally acquire and version this corpus (manifest, hashes, licensing); this
audit only records what `docs_dir` resolves to *today*, unmodified.

## What this baseline is for

Every later phase (`T4.1` onward: Docling, structured chunking, BGE-M3,
Qdrant, hybrid retrieval, reranking, abstention classifier) will claim an
improvement over *something*. This file is that something. A number changed
without a corresponding line here to compare against is not a measured
improvement — it is an impression.
