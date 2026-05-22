<div align="center">

<img src="assets/hero.png" alt="Gemma Miner — extract, analyze, discover" width="100%"/>

# ⛏ Gemma Miner

**Turn any website or document corpus into a typed, research-grade dataset — in minutes, autonomously.**

[![PyPI](https://img.shields.io/pypi/v/gemma-miner.svg?color=blue&cache_seconds=300)](https://pypi.org/project/gemma-miner/)
[![Python](https://img.shields.io/pypi/pyversions/gemma-miner.svg?cache_seconds=300)](https://pypi.org/project/gemma-miner/)
[![Downloads](https://img.shields.io/pypi/dm/gemma-miner.svg?color=brightgreen&cache_seconds=3600)](https://pypi.org/project/gemma-miner/)
[![License](https://img.shields.io/badge/license-Apache--2.0-brightgreen.svg)](LICENSE)
[![HF Datasets](https://img.shields.io/badge/🤗-datasets-yellow)](https://huggingface.co/moncefem)

</div>

Gemma Miner is an autonomous agent that takes a one-sentence goal and produces a clean, typed, analysis-ready dataset — handling everything from crawling to schema design to per-row extraction to export.

```
"Build me a dataset of every CNIL sanction since 2011 with date, organisation, sector, and breach type."
```

Thirty minutes later you have a 374-row × 34-column Parquet file with a Markdown codebook, ready for pandas, DuckDB, or Hugging Face.

---

## What it actually does

Most scraping tools give you a raw JSON dump — a pile of strings with no consistent types, no schema, no way to aggregate. That's not a dataset. Gemma Miner closes the loop end-to-end:

1. **Crawls** the source — HTML pages, JSON APIs, PDFs, DOCX, XLSX, ZIP archives, nested pagination, authenticated endpoints.
2. **Designs a codebook** of 20–60 typed analytical variables appropriate for the corpus: booleans, enums, integers, dates, free-text fields — with controlled vocabularies for every categorical.
3. **Extracts every row** through the codebook with strict type discipline: dates normalised to ISO 8601, enums snapped to the nearest valid value, booleans left null when the source is silent (never fabricated), no placeholder stuffing.
4. **Self-verifies** before declaring done. A second LLM pass audits a sample and checks all contracts. If verification fails, the agent re-enters the loop with corrective feedback.
5. **Exports** to Parquet + CSV + a Markdown codebook, and pushes to the Hugging Face Hub on request.

The result drops directly into `pandas.read_parquet()` or `datasets.load_dataset()` with no second cleaning pass.

---

## Real datasets built end-to-end

| Dataset | Rows × Cols | Source |
|---|---|---|
| 🇫🇷 [CNIL Sanctions 2011–2025](https://huggingface.co/datasets/moncefem/cnil-sanctions-2011-2025) | 374 × 34 | [cnil.fr](https://www.cnil.fr/fr/les-sanctions-prononcees-par-la-cnil) |
| 🧬 [Clinical Trials of AI 2000–2025](https://huggingface.co/datasets/moncefem/clinical-trials-ai-2000-2025) | 3 000 × 30 | [clinicaltrials.gov](https://clinicaltrials.gov) |
| 🪶 [Featherless.ai Model Catalog](https://huggingface.co/datasets/moncefem/featherless-ai-models) | 300 × 20 | [featherless.ai](https://featherless.ai) |

```python
from datasets import load_dataset
ds = load_dataset("moncefem/cnil-sanctions-2011-2025")
```

## Example run

```
› scrape the hf to get the last best models of google

  what I understood
  count target:   30
  fields:         ['model_id', 'author', 'likes', 'downloads', 'last_updated', 'model_url']
  source URL:     https://huggingface.co/models?search=google
  codebook phase: yes

🧭  DISCOVER_LISTING  ·  exploring the site
    1  🌐 http_get    url="https://huggingface.co/models?search=google"
       ↳ status: 200  ·  bytes: 339277  ·  1.4s
    2  🐍 python      extract JSON blob from Svelte data-props
       ↳ exit_code: 0  ·  2.0s
    3  •  llm_scrape  fields=[model_id, author, likes, downloads, …]  target=30
       ↳ NET-NEW 30 rows added  ·  29.6s

📓  CODEBOOK  ·  designing the codebook
    7  ✨ codebook_propose  sample_size=4
       ↳ variables: 25  ·  types: integer×5, boolean×9, enum×4, date×2, float×2  ·  4.9s

🧬  EXTRACT  ·  extracting variables per item
       → 27/27 items (100%)  ·  avg fill 36%  ·  46.8s

🏁  FINISH
   17  🏁 finish
       ↳ 30 rows · 25 variables · 1m 44s · contracts 4/4

  dataset  runs/huggingface_co_3/export/huggingface_google_models.parquet  (17 KB)
```

---

## Install

```bash
# recommended — isolated CLI tool on your PATH
uv tool install gemma-miner

# with optional extras
uv tool install "gemma-miner[parsers]"   # PDF / DOCX / XLSX / EPUB / archives
uv tool install "gemma-miner[hf]"        # push datasets to Hugging Face Hub
uv tool install "gemma-miner[analysis]"  # pandas + matplotlib for post-run analysis
uv tool install "gemma-miner[all]"       # everything
```

| Use case | Command |
|---|---|
| Try once without installing | `uv run --with gemma-miner gemma-miner` |
| Add as a library | `uv add gemma-miner` |
| Plain pip | `pipx install gemma-miner` |

---

## Quick start

```bash
gemma-miner
```

On first launch a wizard asks you to pick a provider, paste an API key, and choose a default model. Your config is saved to `~/.config/gemma-miner/config.toml`. Run `gemma-miner configure` any time to change it.

Then just describe what you want:

```
› Build a dataset of the top 100 Hacker News stories — id, title, domain, points, comment count.
```

The agent plans, crawls, designs a schema, extracts, verifies, and exports — all autonomously. A live activity feed shows every step.

### One-shot (no REPL)

```bash
gemma-miner "Build a dataset of every CNIL sanction from \
https://www.cnil.fr/fr/les-sanctions-prononcees-par-la-cnil \
with date, organisation type, breaches, and decision text."
```

### Explicit flags for scripting

```bash
gemma-miner run \
  --goal "Top 100 Hacker News stories" \
  --min-rows 100 \
  --required-fields rank,id,title,points \
  --unique-field id \
  --workdir ./runs/hn \
  --provider ollama \
  --model gemma4:31b
```

---

## Architecture

Gemma Miner is a **ReAct-style agent loop** with a stateful phase machine, a tool registry, and a contract system that defines what "done" means.

### The loop

Every turn follows the same pattern:

```
1. Compute the current phase from observable state
2. Render a state brief (dataset stats, memory, contracts, recent history)
3. Ask the LLM for ONE tool call
4. Dispatch the tool → append the result to state
5. Repeat until finish() passes self-verification
```

No chat history is accumulated. Instead, the entire state (dataset row count, queue depth, codebook definition, memory entries, contract status) is re-rendered into a fresh prompt every turn. This prevents context drift and keeps small models on track.

### Phase machine

The phase is computed deterministically from observable state — the agent doesn't declare it, it's inferred. This means a stuck or confused agent gets automatically nudged toward the right next action.

```
DISCOVER_LISTING   →   understand the site structure, find pagination
      ↓
ENUMERATE          →   build the full list of item URLs (queue)
      ↓
DISCOVER_DETAIL    →   study one item page to map fields
      ↓
PROCESS            →   fetch + harvest raw text for every item
      ↓
CODEBOOK           →   design the typed schema of analytical variables
      ↓
EXTRACT            →   run the codebook extractor over every harvested row
      ↓
EXPORT             →   write Parquet/CSV/codebook.md, push to HF if asked
      ↓
FINISH             →   self-verify → done (or retry with corrective feedback)
```

Each phase exposes a **different subset of tools** to the model. In `DISCOVER_LISTING` the agent sees http/html tools; in `EXTRACT` it sees the extraction tools; in `EXPORT` it sees the export tools. Fewer choices per turn → dramatically better behaviour from smaller models.

### Tool registry (~40 tools, organised by concern)

| Group | Tools | What they do |
|---|---|---|
| **Web** | `http_get`, `html_inspect`, `html_extract`, `html_find` | Fetch pages, inspect DOM structure, pull CSS-selected fields |
| **Parsing** | `extract_text` | Universal text extractor: PDF, DOCX, PPTX, XLSX, EPUB, HTML, JSON, YAML, CSV, ZIP/tar — dispatches by extension then magic bytes |
| **Declarative scrape** | `extractor_define`, `scrape_paginated`, `process_queue` | Define a CSS/regex extraction rule once, apply it to hundreds of pages in batch |
| **Queue** | `queue_add`, `queue_next`, `queue_mark_done`, `queue_status` | Persistent work queue for the list of items to harvest |
| **Codebook** | `codebook_propose`, `codebook_show`, `codebook_edit`, `codebook_test` | Design, inspect and refine the typed analytical schema |
| **Extraction** | `extract_structured`, `extract_items` | Run a codebook over harvested raw text using the extraction LLM |
| **Dataset** | `dataset_append`, `dataset_stats`, `dataset_sample`, `dataset_patch` | Append rows, inspect shape and fill rates, fix individual cells |
| **Export** | `dataset_validate`, `dataset_export`, `hf_push` | Validate, write Parquet/CSV/codebook.md, push to Hugging Face |
| **Memory** | `memory_set`, `memory_get`, `memory_list` | Persistent key-value store (survives tool calls, used for plan + lessons learned) |
| **Plan** | `set_plan`, `show_plan` | Write and display a structured scraping plan |
| **Code** | `python`, `bash` | Execute Python or shell commands inside the workdir (destructive ops blocked) |
| **Attachments** | `save_attachment` | Download a binary (PDF, image), extract its text, and store both under `items/item_NNNN/` |

### Contracts — defining "done"

Contracts are assertions that must hold before the agent is allowed to call `finish`. They are checked continuously and shown in the status bar throughout the run.

| Contract | What it checks |
|---|---|
| `MinRowsContract` | Dataset has at least N rows |
| `FieldsContract` | Every required field is present in every row |
| `UniqueFieldContract` | A specified field has no duplicate values |
| `CodebookContract` | The codebook defines at least N typed variables |
| `CoverageContract` | No variable has a fill rate below a threshold |

Failed contracts re-open the agent loop with the failure message injected into the next prompt, forcing correction rather than a silent finish.

### Two-LLM setup

The agent uses **two separate LLM clients** that can be pointed at different models:

- **Agent LLM** — drives the ReAct loop. Can be a large reasoning model (e.g., Gemini 2.5 Pro) or a capable small model (Gemma 4 31B on Ollama). Makes one tool-call decision per turn.
- **Extraction LLM** — called in batch by `extract_items` to populate the codebook fields for every row. Typically a fast, cheap model (e.g., Gemini Flash, Gemma 3 9B) because it runs N times where N is your row count.

You can point them at different providers — e.g., agent on OpenRouter, extraction on local Ollama — or keep them identical.

### Self-verification

Before accepting `finish`, the agent runs a verification pass:

1. **Contract checks** — all contracts must be satisfied.
2. **Schema homogeneity** — any field missing in more than 30% of rows is flagged as sparse.
3. **LLM critique** — a small sample of rows is audited by the LLM against the original goal (severity: none / low / high / blocker).

On failure the agent is re-launched with the list of problems injected as corrective feedback, up to `max_verify_retries` times.

### Workdir layout

Each run writes its output to a self-contained directory:

```
runs/my-run/
├── dataset.jsonl        # append-only bronze store (raw harvested rows)
├── memory.json          # agent's key-value memory
├── trace.jsonl          # every LLM decision + tool result, machine-readable
├── trace.log            # human-readable turn-by-turn log
├── codebook.json        # typed schema definition
├── items/               # downloaded attachments (PDFs, images, …)
│   └── item_0001/
│       ├── attachment_01.pdf
│       └── attachment_01.txt   # extracted text
└── export/
    ├── dataset.parquet  # typed, final dataset
    ├── dataset.csv
    └── codebook.md      # Markdown codebook for human review
```

---

## REPL commands

Type `/` inside the REPL to see all commands filtered as you type.

| Command | What it does |
|---|---|
| `/help` | Full help panel |
| `/config` | Re-run the provider + API-key setup wizard |
| `/provider [<name>]` | Show or switch the agent LLM provider (persisted) |
| `/model [<id>]` | Show or switch the agent model (persisted per provider) |
| `/extract-provider [<name>]` | Show or switch the extraction LLM provider |
| `/extract-model [<id>]` | Show or switch the extraction model |
| `/gemma-full-local` | Switch both LLMs to Ollama, auto-picking the largest installed Gemma |
| `/datasets` | List datasets produced in `./runs/` |
| `/workdir [<path>]` | Show or change the base workdir |
| `/resume <path>` | Resume a previous run — reloads dataset, codebook, memory |
| `/push <repo_id>` | Push the last dataset to Hugging Face Hub |
| `/trace` | Open the trace log for the last run |
| `/history`, `/clear`, `/quit` | Standard shell controls |

After a run, the agent holds the dataset in memory. Ask follow-up questions — *"which row had the highest fine?"*, *"summarise breaches by sector"* — and it answers from the data without triggering another scrape.

---

## Python API

```python
from gemma_miner import run_agent, make_llm
from gemma_miner.contracts import MinRowsContract, FieldsContract, UniqueFieldContract

result = run_agent(
    goal=(
        "Build a dataset of the top 100 Hacker News stories using the public "
        "JSON API at https://hacker-news.firebaseio.com/v0/. "
        "Each row needs rank, id, title, domain, points, comment_count."
    ),
    contracts=[
        MinRowsContract(min_rows=100),
        FieldsContract(required_fields=["rank", "id", "title", "points"]),
        UniqueFieldContract(field="id"),
    ],
    unique_key="id",
    workdir="./runs/hn",
    llm=make_llm("openrouter", model="google/gemini-3.1-flash-lite"),
    # optional: faster/cheaper model just for the per-row extraction pass
    extraction_llm=make_llm("ollama", model="gemma3:9b"),
)

print(f"{result.n_rows} rows → {result.dataset_path}")
```

You can also inject extra memory entries to pre-load the agent with known context (auth tokens, field mappings, pagination patterns):

```python
result = run_agent(
    goal="...",
    contracts=[...],
    workdir="./runs/x",
    extra_memory={
        "auth_cookie": "session=abc123",
        "listing_url": "https://example.com/items?page={page}",
    },
)
```

---

## Providers

| Provider | Type | Default model | API key env var |
|---|---|---|---|
| **Ollama** | Local | `gemma4:31b` (wizard shows your installed models) | — |
| **OpenRouter** | Cloud (router) | `google/gemini-3.1-flash-lite` | `OPENROUTER_API_KEY` |
| **Together AI** | Cloud (OSS) | `google/gemma-4-31b-it` | `TOGETHER_API_KEY` |
| **Featherless** | Serverless GPU | `google/gemma-4-31B-it` | `FEATHERLESS_API_KEY` |
| **openai-compatible** | Anything else | — | set via `/config` |

Run `gemma-miner providers` to see the full list with base URLs.

---

## Hugging Face export

```bash
# from inside the REPL
› /push moncefem/my-dataset

# from the shell
gemma-miner export-hf ./runs/hn/dataset.jsonl --repo-id you/hn-top100
```

Requires the `hf` extra and `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in the environment.

---

## Safety

- `bash` and `python` tools block destructive patterns (`rm -rf`, `dd`, `mkfs`, `sudo`, fork bombs) at the tool layer before any shell is invoked.
- All file operations are confined to the run's workdir.
- The config file is written with `chmod 600` so API keys are not readable by other users on shared machines.

Do not run the agent on production machines. Use a container or a throwaway VM.

---

## Contributing

Bug reports, ideas, and pull requests welcome at <https://github.com/moncifem/gemma-miner>.

```bash
git clone https://github.com/moncifem/gemma-miner
cd gemma-miner
uv pip install -e ".[dev]"
pytest -q
```

---

## License

[Apache License 2.0](LICENSE).

If you use Gemma Miner in a paper, project, or product, attribution is appreciated:

```bibtex
@software{elmouden_gemma_miner_2025,
  title  = {Gemma Miner: an autonomous text-to-dataset agent},
  author = {EL-Mouden, Moncif and contributors},
  year   = {2025},
  url    = {https://github.com/moncifem/gemma-miner},
}
```

---

<div align="center">

⛏ Made with care by <a href="https://huggingface.co/moncefem">Moncif EL-Mouden</a>.
Powered by your favourite open model.

</div>
