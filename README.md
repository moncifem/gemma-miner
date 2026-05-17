# gemma42

A structured scraping & dataset-construction agent designed for **small open
models** (Gemma 3, Llama 3, Mistral, Qwen) served through Together AI or any
OpenAI-compatible endpoint.

The point of `gemma42` is to let a 7B–30B model behave like a state-of-the-art
agent on a narrow but useful task: **read a website, identify the repeating
unit of data, extract it row by row, and stop only when the user's spec is
fully met.**

## What it does

- **Reads the open web** with stdlib-grade HTTP + HTML inspection tools.
- **Builds a JSONL dataset** with optional JSON-Schema validation and
  uniqueness constraints.
- **Honours contracts**: declarative requirements (`min_rows=100`,
  `required_fields=[…]`, `unique_field=id`) that gate the `finish` tool.
  The agent literally cannot terminate until every contract is `OK`.
- **Contracts are mutable mid-run**: if you (or the agent) realise the spec
  changed — "actually I need 200 rows, and a `points` field too" — call the
  `add_contract` tool and the loop keeps going.
- **Has memory**: a persistent JSON KV store the agent uses to recall
  selectors, schemas, and other site-specific facts across turns and runs.
- **Has a code tool**: ad-hoc Python in a sandboxed subprocess for parsing,
  transforming, and computing — without giving the model a destructive shell.
  `rm`, `dd`, `mkfs`, `sudo`, etc. are blocked at the tool layer.
- **Drives the LLM as a tool too**: `extract_structured` runs the same
  small model under a strict extraction system prompt against any JSON
  Schema. Great for prose-heavy sources (legal decisions, articles).
- **Exports to Hugging Face** with a single command.

## Install

```bash
uv pip install gemma42                # core
uv pip install "gemma42[parsers]"     # + pdf/docx/xlsx/epub/... extractors
uv pip install "gemma42[hf]"          # + huggingface export
uv pip install "gemma42[all]"         # everything
```

Or from source:

```bash
git clone https://github.com/yourname/gemma42 && cd gemma42
uv pip install -e ".[hf,dev]"
```

## Providers

Anything that speaks the OpenAI chat-completions protocol works. Built-in
presets:

| Provider | Default model | API key env |
|---|---|---|
| `together` | `google/gemma-3n-E4B-it` | `TOGETHER_API_KEY` |
| `ollama` | `gemma3:27b` (local) | — |
| `groq` | `llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| `openrouter` | `google/gemma-3-27b-it` | `OPENROUTER_API_KEY` |
| `fireworks` | `accounts/fireworks/models/gemma2-27b-it` | `FIREWORKS_API_KEY` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `openai-compatible` | (pass `--base-url` + `--model`) | optional |

```python
from gemma42 import make_llm

llm = make_llm("ollama", model="gemma3:27b")        # local 27B-class
llm = make_llm("together", model="google/gemma-3n-E4B-it")
llm = make_llm("openai-compatible",
               base_url="http://my-vllm:8000/v1",
               model="meta-llama/Meta-Llama-3.1-8B-Instruct")
```

For a 31B-ish local model with Ollama:

```bash
ollama pull gemma3:27b      # closest to "Gemma 3 ~31B"
gemma42 run --provider ollama --model gemma3:27b ...
```

## Quick start (CLI)

```bash
export TOGETHER_API_KEY=...

gemma42 run \
  --goal "Build a dataset of the current top 100 Hacker News stories with rank, id, title, domain, and points. Use the public JSON API at hacker-news.firebaseio.com." \
  --min-rows 100 \
  --required-fields rank,id,title,points \
  --unique-field id \
  --workdir ./runs/hn \
  --model "google/gemma-3n-E4B-it"
```

When the run finishes you get `runs/hn/dataset.jsonl`. Push it:

```bash
gemma42 export-hf ./runs/hn/dataset.jsonl --repo-id yourname/hn-top100
```

## Quick start (Python)

```python
from gemma42 import (
    FieldsContract, LLMClient, MinRowsContract,
    UniqueFieldContract, run_agent,
)

result = run_agent(
    goal=(
        "Build a dataset of the top 100 Hacker News stories using the public "
        "JSON API. Each row needs rank, id, title, domain, points."
    ),
    contracts=[
        MinRowsContract(min_rows=100),
        FieldsContract(required_fields=["rank", "id", "title", "points"]),
        UniqueFieldContract(field="id"),
    ],
    unique_key="id",
    workdir="./runs/hn",
    llm=LLMClient(model="google/gemma-3n-E4B-it"),
)
print(result)
```

See `examples/` for two more end-to-end recipes:

- `examples/cnil_sanctions.py` — scrape an HTML table of GDPR fines.
- `examples/competition_decisions.py` — read PDF decisions and emit one row
  per decision matching a deeply nested JSON schema (the project's brief
  example).

## How the agent is structured

```
run_agent(goal, contracts, workdir, llm)
        │
        ▼
 ┌──────────────────────────────────────┐
 │             AgentState               │
 │  goal, dataset, contracts, memory,   │
 │  workdir, history                    │
 └──────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────┐
 │            Agent loop                │
 │  ┌── render state brief ───────────┐ │
 │  │  goal + contracts + last 3 obs  │ │
 │  └────────────┬────────────────────┘ │
 │               ▼                      │
 │       LLM (JSON tool call)           │
 │               ▼                      │
 │       parse → dispatch → record      │
 │               ▼                      │
 │     contracts all OK?  →  finish     │
 └──────────────────────────────────────┘
```

### Tools shipped by default

| Tool | Purpose |
|---|---|
| `http_get` | Fetch a URL; cache body to disk; return preview |
| `html_inspect` | Frequent tags / classes / ids in a page — pick the unit selector |
| `html_extract` | Run a regex over cached HTML, see N matches |
| `extract_text` | Universal text extractor: pdf, docx, pptx, xlsx, odt, rtf, epub, html, xml, json, jsonl, yaml, toml, csv, tsv, zip, tar, gz (recurses into archives). Magic-byte sniffing for extensionless files. |
| `python` | Run a Python snippet in a fresh subprocess |
| `bash` | Run a bash command (destructive ops blocked) |
| `read_file` / `write_file` / `list_dir` | Workdir-scoped FS |
| `dataset_append` / `dataset_stats` / `dataset_sample` | Manage the JSONL output |
| `memory_set` / `memory_get` / `memory_list` | Persistent KV store |
| `add_contract` / `contract_status` | Mutate the contract book |
| `extract_structured` | LLM-driven JSON-schema extraction from prose |
| `finish` | Declare done — only allowed when contracts pass |

### Why "contracts"?

Small models love to call `finish` too early. They will declare victory after
12 rows when you asked for 100, or skip the `points` field because it wasn't
in the first response. Contracts solve this declaratively: `finish` is a
no-op until every check returns `OK`. The agent reads the failing checks in
its state brief on every turn, so the next action is always obvious.

### Why a state brief instead of full chat history?

Small models drift when their context fills with stale observations. We
re-render a compact brief each turn: goal + dataset progress + contract
status + the last three turns. Everything else lives in `memory` (which the
agent can query) or `dataset.jsonl` (which it can sample).

## Storage convention for downloaded attachments

When the agent downloads files (PDFs, XML, CSV, images) and extracts their
text, it is instructed to write them under a predictable, enumerable layout:

```
workdir/
  dataset.jsonl
  items/
    item_0001/
      meta.json
      attachment_01.pdf
      attachment_01.txt        ← extract_text output
      attachment_02.xml
      attachment_02.txt
    item_0002/
      ...
```

This means every extracted corpus is trivially iterable:

```python
from pathlib import Path
for txt in sorted(Path("./runs/myrun/items").glob("*/attachment_*.txt")):
    item_id = txt.parent.name        # "item_0001"
    print(item_id, txt.read_text()[:120])
```

The instruction is baked into the system prompt and into the description of
the `extract_text` tool, so small models reliably follow it.

## Safety

- The `bash` and `python` tools refuse any input that matches a blocklist
  (`rm`, `mv` to root, `dd`, `mkfs`, `sudo`, `chmod -R 777`, fork bombs, …).
- File tools are confined to the workdir.
- `http_get` does not follow `file://` schemes and caches everything inside
  the workdir.

These are guardrails, not a sandbox. Don't run agent code on production
boxes; use a container or VM.

## License

MIT.
