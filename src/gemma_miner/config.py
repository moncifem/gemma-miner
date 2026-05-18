"""Persistent gemma-miner configuration.

Stored at $XDG_CONFIG_HOME/gemma-miner/config.toml (defaults to
~/.config/gemma-miner/config.toml on macOS/Linux, %APPDATA%/gemma-miner/config.toml
on Windows).

Schema (TOML):

    default_provider = "openrouter"
    default_extract_provider = "openrouter"

    [providers.openrouter]
    api_key = "sk-or-…"
    recent_model = "google/gemini-3.1-flash-lite"

    [providers.together]
    api_key = "…"
    recent_model = "google/gemma-4-31B-it"

    [providers.ollama]
    recent_model = "gemma3:27b"
    # no api_key — local

The file is chmod 600 (owner read/write only) so API keys aren't readable
by other users on a shared machine.

Public surface (used everywhere else):

    load() -> dict
    save(cfg)
    config_path() -> Path
    apply_env(cfg)
        Sets the per-provider API-key env var (TOGETHER_API_KEY etc.) so
        the existing providers.py code keeps working unchanged.

    get_default_provider() / set_default_provider(name)
    get_recent_model(provider) / set_recent_model(provider, model)
    get_api_key(provider) / set_api_key(provider, key)
    ensure_first_run_done() -> bool
        Returns True if config exists, False if it's a first launch.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any

# Python 3.11+ has tomllib in stdlib; older versions use the `tomli` shim.
try:
    import tomllib as _toml_reader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import tomli as _toml_reader  # type: ignore[import-not-found]


# ── Paths ──────────────────────────────────────────────────────────────────


def _config_root() -> Path:
    """Resolve the config directory in an XDG-respecting way."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "gemma-miner"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "gemma-miner"
    return Path.home() / ".config" / "gemma-miner"


def config_path() -> Path:
    return _config_root() / "config.toml"


def _legacy_config_path() -> Path:
    """Pre-rename location used during the `gemma42` dev cycle. We migrate
    once so existing users don't have to re-run the wizard."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "gemma42" / "config.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "gemma42" / "config.toml"
    return Path.home() / ".config" / "gemma42" / "config.toml"


def _migrate_legacy_config() -> bool:
    """Copy legacy `gemma42` config over when the new one doesn't exist."""
    new_p = config_path()
    if new_p.exists():
        return False
    old_p = _legacy_config_path()
    if not old_p.exists():
        return False
    try:
        new_p.parent.mkdir(parents=True, exist_ok=True)
        new_p.write_bytes(old_p.read_bytes())
        try:
            os.chmod(new_p, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return True
    except OSError:
        return False


# ── Schema defaults ────────────────────────────────────────────────────────


# Providers we present in the wizard (subset of providers.PRESETS).
SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "ollama",       # local, no key
    "openrouter",   # cheap router with many models
    "together",     # together.ai
    "featherless",  # featherless.ai
)

PROVIDER_LABELS: dict[str, str] = {
    "ollama":      "Ollama (local, no API key)",
    "openrouter":  "OpenRouter (paid, hundreds of models)",
    "together":    "Together AI (paid, fast open models)",
    "featherless": "Featherless AI (paid, OSS models on serverless GPUs)",
}

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "ollama":      "",                       # none
    "openrouter":  "OPENROUTER_API_KEY",
    "together":    "TOGETHER_API_KEY",
    "featherless": "FEATHERLESS_API_KEY",
}

# Gemma 4 31B for every external provider — except OpenRouter, which stays
# on Gemini 3.1 Flash (cheap, fast, large context). Ollama defaults to the
# Gemma 4 31B local image; if the user doesn't have it pulled,
# `/gemma-full-local` will pick whatever is installed.
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "ollama":      "gemma4:31b",
    "openrouter":  "google/gemini-3.1-flash-lite",
    "together":    "google/gemma-4-31b-it",
    "featherless": "google/gemma-4-31b-it",
}


# ── Read / write ───────────────────────────────────────────────────────────


def _empty() -> dict[str, Any]:
    return {
        "default_provider": None,
        "default_extract_provider": None,
        "providers": {},
    }


def load() -> dict[str, Any]:
    """Load the config dict. Returns an empty skeleton when no file exists.

    Migrates a legacy `~/.config/gemma42/config.toml` to the new path on
    first call, so users coming from the dev version don't have to re-run
    the wizard.
    """
    _migrate_legacy_config()
    p = config_path()
    if not p.exists():
        return _empty()
    try:
        with p.open("rb") as f:
            data = _toml_reader.load(f) or {}
    except Exception:  # noqa: BLE001
        return _empty()
    # Normalise shape.
    if not isinstance(data.get("providers"), dict):
        data["providers"] = {}
    data.setdefault("default_provider", None)
    data.setdefault("default_extract_provider", None)
    return data


def _toml_escape(s: str) -> str:
    """Escape a string for a basic TOML key/value (no triple quotes)."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _dump_toml(cfg: dict[str, Any]) -> str:
    """Hand-rolled TOML emitter for our tiny schema. Avoids adding
    `tomli-w` / `tomlkit` as a runtime dep for ~20 lines of formatting."""
    out: list[str] = []
    out.append(f'# gemma-miner configuration — managed by `gemma-miner configure`')
    out.append("")
    if cfg.get("default_provider"):
        out.append(f'default_provider = "{_toml_escape(cfg["default_provider"])}"')
    if cfg.get("default_extract_provider"):
        out.append(
            f'default_extract_provider = "{_toml_escape(cfg["default_extract_provider"])}"'
        )
    out.append("")
    for name in sorted((cfg.get("providers") or {}).keys()):
        section = cfg["providers"][name] or {}
        out.append(f"[providers.{name}]")
        for k in ("api_key", "recent_model", "base_url"):
            v = section.get(k)
            if isinstance(v, str) and v:
                out.append(f'{k} = "{_toml_escape(v)}"')
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def save(cfg: dict[str, Any]) -> None:
    """Write the config to disk and chmod 600 so other users can't read it."""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(_dump_toml(cfg), encoding="utf-8")
    os.replace(tmp, p)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Windows / weird FS


# ── Accessors / mutators ───────────────────────────────────────────────────


def get_default_provider(cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg if cfg is not None else load()
    return cfg.get("default_provider")


def set_default_provider(name: str) -> None:
    cfg = load()
    cfg["default_provider"] = name
    save(cfg)


def get_default_extract_provider(cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg if cfg is not None else load()
    return cfg.get("default_extract_provider") or cfg.get("default_provider")


def set_default_extract_provider(name: str) -> None:
    cfg = load()
    cfg["default_extract_provider"] = name
    save(cfg)


def get_recent_model(provider: str, cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg if cfg is not None else load()
    return ((cfg.get("providers") or {}).get(provider) or {}).get("recent_model")


def set_recent_model(provider: str, model: str) -> None:
    cfg = load()
    cfg.setdefault("providers", {}).setdefault(provider, {})["recent_model"] = model
    save(cfg)


def get_api_key(provider: str, cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg if cfg is not None else load()
    return ((cfg.get("providers") or {}).get(provider) or {}).get("api_key")


def set_api_key(provider: str, key: str) -> None:
    cfg = load()
    cfg.setdefault("providers", {}).setdefault(provider, {})["api_key"] = key
    save(cfg)


def apply_env(cfg: dict[str, Any] | None = None) -> None:
    """Export configured API keys to environment variables so providers.py
    picks them up via `os.getenv(...)`. Does NOT overwrite existing env vars
    — anything the user exported in their shell wins."""
    cfg = cfg if cfg is not None else load()
    for provider, env_name in PROVIDER_API_KEY_ENV.items():
        if not env_name:
            continue
        key = get_api_key(provider, cfg)
        if key and not os.environ.get(env_name):
            os.environ[env_name] = key


def ensure_first_run_done() -> bool:
    """Return True if a config file already exists. The CLI uses this to
    decide whether to drop into the first-run wizard."""
    return config_path().exists()
