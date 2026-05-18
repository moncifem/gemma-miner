# Changelog

All notable changes to **Gemma Miner** will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2025

First public release. Brand renamed from internal "gemma42" to
**Gemma Miner**; PyPI package is `gemma-miner`.

### Added

- **Claude-Code-style REPL** with a live `/` command palette
  (prompt_toolkit), multi-line heredoc input, slash-command history.
- **Persistent configuration** at `~/.config/gemma-miner/config.toml`
  (XDG-compliant, `chmod 600`). First-run wizard picks provider,
  prompts for API key, and persists the default model per provider.
- **Providers**: Ollama (local), OpenRouter, Together AI, Featherless,
  plus any OpenAI-compatible endpoint via `--base-url`. Default model
  is **Gemma 4 31B** on every external provider except OpenRouter
  (which keeps `google/gemini-3.1-flash-lite`).
- **Ollama model picker**: the wizard queries `/api/tags` and shows the
  user the models they actually have installed.
- **`/gemma-full-local`** slash command — auto-detects the largest
  Gemma model installed in Ollama and switches every phase to it.
- **`/resume <path>`** — re-enter a previous run with its dataset,
  codebook and memory loaded; **`/push <repo_id>`** — one-shot upload
  to Hugging Face Hub from inside the REPL.
- **Phase machine** with hysteresis: once the silver dataset is built,
  the loop refuses to fall back into harvest for marginal gains.
- **Self-verification** pass before `finish` is accepted; on failure
  the agent re-enters the loop with the verifier's feedback.
- **Null-not-false discipline** end-to-end: booleans are `null` when
  the source is silent. `FieldsContract` flags placeholder stuffing
  via a language-agnostic low-cardinality detector.
- **Deterministic content-hash IDs** so bronze ↔ silver join is stable
  across re-runs.
- **Codebook write-once guard** + automatic silver migration on
  rename/drop — cosmetic changes (`set_required`) no longer trigger
  a full re-extract.
- **`extract_items`** guards: pilot-then-scale, insufficient-text gate,
  partial-fill mode when only some variables are new, refuses
  `skip_existing=false` on > 50 already-extracted rows without
  `force=true`.
- **HuggingFace export**: `dataset_export` + `hf_push` push parquet,
  CSV, codebook + README to a public/private dataset repo.

### Reference datasets

Two end-to-end demonstrations live on the Hugging Face Hub:

- [`moncefem/cnil-sanctions-2011-2025`](https://huggingface.co/datasets/moncefem/cnil-sanctions-2011-2025) — 374 GDPR sanctions × 34 typed columns.
- [`moncefem/clinical-trials-ai-2000-2025`](https://huggingface.co/datasets/moncefem/clinical-trials-ai-2000-2025) — 3 000 AI/ML clinical trials × 30 typed columns.

[0.1.0]: https://github.com/moncifem/gemma-miner/releases/tag/v0.1.0
