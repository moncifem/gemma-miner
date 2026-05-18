"""Smoke tests for the trimmed gemma42 core.

These cover the public surface that downstream users (and the CLI) depend on:
parser, contracts, dataset, memory, registry wiring, and phase machine.
External I/O (HTTP, LLM) is mocked or avoided.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_miner.contracts import (
    ContractBook,
    FieldsContract,
    MinRowsContract,
    UniqueFieldContract,
)
from gemma_miner.dataset import Dataset
from gemma_miner.memory import Memory
from gemma_miner.parsing import ParseError, parse_tool_call
from gemma_miner.state import AgentState
from gemma_miner.tools.registry import default_registry


# ── parser ──────────────────────────────────────────────────────────────────


def test_parse_clean_tool_call():
    raw = '{"thought": "ok", "tool": "http_get", "args": {"url": "https://x"}}'
    c = parse_tool_call(raw)
    assert c.tool == "http_get"
    assert c.args == {"url": "https://x"}


def test_parse_fenced_tool_call():
    raw = (
        "Let me fetch the page.\n"
        "```json\n"
        '{"thought":"fetch","tool":"http_get","args":{"url":"https://x"}}\n'
        "```\n"
    )
    c = parse_tool_call(raw)
    assert c.tool == "http_get"


def test_parse_python_quoted_args_repaired():
    raw = "{'thought': 'a', 'tool': 'memory_set', 'args': {'k': 'v'}}"
    c = parse_tool_call(raw)
    assert c.tool == "memory_set"
    assert c.args == {"k": "v"}


def test_parse_garbage_raises():
    with pytest.raises(ParseError):
        parse_tool_call("totally not json — sorry")


# ── dataset ─────────────────────────────────────────────────────────────────


def test_dataset_append_and_unique(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl", unique_key="id")
    for r in [{"id": "1", "v": 1}, {"id": "2", "v": 2}]:
        ok, _ = ds.append(r)
        assert ok
    # Duplicate id is handled by the dataset (upsert/dedup semantics).
    ds.append({"id": "1", "v": 99})
    ds.append({"id": "3", "v": 3})
    assert len(ds) == 3
    ids = sorted(r["id"] for r in ds.rows())
    assert ids == ["1", "2", "3"]


def test_dataset_persists(tmp_path: Path):
    p = tmp_path / "d.jsonl"
    ds1 = Dataset(p)
    ds1.append({"a": 1})
    ds1.append({"a": 2})
    ds2 = Dataset(p)
    assert len(ds2) == 2


# ── contracts ───────────────────────────────────────────────────────────────


def test_min_rows_contract(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl")
    c = MinRowsContract(min_rows=2)
    ok, _ = c.check(ds)
    assert not ok
    for r in [{"a": 1}, {"a": 2}]:
        ds.append(r)
    assert c.check(ds)[0]


def test_fields_contract(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl")
    ds.append({"id": "x", "title": "t"})
    assert FieldsContract(required_fields=["id", "title"]).check(ds)[0]
    assert not FieldsContract(required_fields=["id", "missing"]).check(ds)[0]


def test_unique_field_contract(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl")
    for r in [{"id": "1"}, {"id": "2"}, {"id": "2"}]:
        ds.append(r)
    assert not UniqueFieldContract(field="id").check(ds)[0]


def test_contract_book_all_satisfied(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl")
    for r in [{"id": "1", "v": 1}, {"id": "2", "v": 2}]:
        ds.append(r)
    book = ContractBook([
        MinRowsContract(min_rows=2),
        FieldsContract(required_fields=["id"]),
        UniqueFieldContract(field="id"),
    ])
    assert book.all_satisfied(ds)


# ── memory ──────────────────────────────────────────────────────────────────


def test_memory_persists(tmp_path: Path):
    m1 = Memory(tmp_path / "m.json")
    m1.set("k", {"a": 1})
    m2 = Memory(tmp_path / "m.json")
    assert m2.get("k") == {"a": 1}


# ── registry & state ───────────────────────────────────────────────────────


def test_default_registry_has_core_tools():
    reg = default_registry()
    names = set(reg.names())
    expected = {
        "http_get", "html_inspect", "extractor_define", "scrape_paginated",
        "process_queue", "dataset_append", "dataset_export", "hf_push",
        "finish", "python", "bash", "queue_add", "memory_set",
    }
    missing = expected - names
    assert not missing, f"missing tools: {missing}"


def test_agent_state_brief(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl")
    memory = Memory(tmp_path / "m.json")
    state = AgentState(
        goal="build something",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=10)]),
        memory=memory,
        workdir=str(tmp_path),
    )
    brief = state.to_brief()
    assert brief["goal"] == "build something"
    assert brief["dataset_rows"] == 0


# ── phases ─────────────────────────────────────────────────────────────────


def test_phase_starts_in_discover(tmp_path: Path):
    from gemma_miner.phases import current_phase

    ds = Dataset(tmp_path / "d.jsonl")
    memory = Memory(tmp_path / "m.json")
    state = AgentState(
        goal="anything",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=10)]),
        memory=memory,
        workdir=str(tmp_path),
    )
    assert current_phase(state, contract_min_rows=10).name == "DISCOVER_LISTING"


def test_phase_advances_when_target_met(tmp_path: Path):
    from gemma_miner.phases import current_phase

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(15):
        ds.append({"id": str(i)})
    memory = Memory(tmp_path / "m.json")
    state = AgentState(
        goal="anything",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=10)]),
        memory=memory,
        workdir=str(tmp_path),
    )
    assert current_phase(state, contract_min_rows=10).name == "FINISH"


# ── prompt renderer wires up ───────────────────────────────────────────────


def test_phase_hysteresis_locks_to_export(tmp_path: Path):
    """Once silver is populated to ≥90% of target, no fallback to harvest."""
    from gemma_miner.phases import current_phase

    ds = Dataset(tmp_path / "d.jsonl", unique_key="id")
    for i in range(10):
        ds.append({"id": str(i), "text": "x"})
    memory = Memory(tmp_path / "m.json")
    memory.set("_post_extract_done", True)
    state = AgentState(
        goal="anything",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=20)]),  # 10 < 20
        memory=memory,
        workdir=str(tmp_path),
    )
    # Without hysteresis, this would be DISCOVER_LISTING. With it: EXPORT.
    assert current_phase(state, contract_min_rows=20).name in ("EXPORT", "FINISH")


def test_set_plan_downgrades_unreachable_min_rows(tmp_path: Path):
    """If observed target_rows < contract.min_rows, the contract is lowered."""
    from gemma_miner.tools.plan_tool import SetPlanTool

    ds = Dataset(tmp_path / "d.jsonl")
    memory = Memory(tmp_path / "m.json")
    state = AgentState(
        goal="g",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=400)]),
        memory=memory,
        workdir=str(tmp_path),
    )
    tool = SetPlanTool()
    res = tool.run(
        {
            "item": "x", "source": "listing_html",
            "source_url": "https://example.com",
            "pagination": "none", "items_per_page": 396,
            "target_rows": 396, "pages_needed": 1,
            "harvest_strategy": "listing_only",
        },
        state,
    )
    assert not res.error
    [mr] = [c for c in state.contracts.list() if isinstance(c, MinRowsContract)]
    assert mr.min_rows == 396
    assert "auto-downgraded" in res.output


# ── data-quality fixes ────────────────────────────────────────────────────


def test_boolean_coercion_null_not_false():
    """Ambiguous/silent tokens must coerce to None, not False."""
    from gemma_miner.codebook import VariableSpec
    from gemma_miner.coercion import coerce

    v = VariableSpec(name="has_x", type="boolean", description="")
    assert coerce("unknown", v) is None
    assert coerce("not stated", v) is None
    assert coerce("n/a", v) is None
    # 0/1 are AMBIGUOUS in our discipline — refuse to guess.
    assert coerce(0, v) is None
    assert coerce(1, v) is None
    # Explicit positives/negatives still work.
    assert coerce("yes", v) is True
    assert coerce("non", v) is False


def test_date_coercion_handles_ddmmyyyy():
    """The CNIL '08/01/2026' regression — DD/MM/YYYY must coerce to ISO."""
    from gemma_miner.codebook import VariableSpec
    from gemma_miner.coercion import coerce

    v = VariableSpec(name="dn_decision", type="date", description="")
    assert coerce("08/01/2026", v) == "2026-01-08"
    assert coerce("2026-01-08T15:30:00Z", v) == "2026-01-08"
    assert coerce("16 avril 2026", v) == "2026-04-16"


def test_enum_coercion_snaps_to_nearest():
    from gemma_miner.codebook import VariableSpec
    from gemma_miner.coercion import coerce

    v = VariableSpec(
        name="cat_x", type="enum",
        description="",
        enum_values=["data_security_failure", "retention_too_long"],
    )
    # Exact + accent-insensitive + close match all snap.
    assert coerce("data_security_failure", v) == "data_security_failure"
    assert coerce("Data Security Failure", v) == "data_security_failure"
    assert coerce("data-security-failure", v) == "data_security_failure"
    # Garbage stays null.
    assert coerce("totally unrelated thing", v) is None


def test_synthesize_id_is_deterministic():
    """Same content → same id, regardless of key order or trailing _ fields."""
    from gemma_miner.dataset import synthesize_id

    a = {"title": "x", "url": "https://y"}
    b = {"url": "https://y", "title": "x", "_scratch": "ignore"}
    assert synthesize_id(a) == synthesize_id(b)


def test_codebook_finds_duplicate_groups():
    from gemma_miner.codebook import Codebook, VariableSpec

    cb = Codebook(
        name="x", description="",
        variables=[
            VariableSpec(name="has_llm", type="boolean", description=""),
            VariableSpec(name="is_mentions_llm", type="boolean", description=""),
            VariableSpec(name="has_vector_database", type="boolean", description=""),
            VariableSpec(name="has_vector_db", type="boolean", description=""),
            VariableSpec(name="n_authors", type="integer", description=""),
        ],
    )
    groups = cb.find_duplicate_groups()
    assert len(groups) == 2
    flat = {n for g in groups for n in g}
    assert "has_llm" in flat and "is_mentions_llm" in flat
    assert "has_vector_database" in flat and "has_vector_db" in flat
    assert "n_authors" not in flat


def test_dataset_reloads_after_external_mutation(tmp_path: Path):
    """If the JSONL is rewritten outside the Dataset class, the next call
    must see the new state, not the stale in-memory rows."""
    import json as _json
    import time as _time

    p = tmp_path / "d.jsonl"
    ds = Dataset(p)
    ds.append({"id": "a", "v": 1})
    assert len(ds) == 1

    # Pause to make sure the mtime delta is > our 0.0001s threshold,
    # then rewrite the file outside `ds`.
    _time.sleep(0.05)
    with p.open("w", encoding="utf-8") as f:
        f.write(_json.dumps({"id": "a", "v": 1}) + "\n")
        f.write(_json.dumps({"id": "b", "v": 2}) + "\n")

    assert len(ds) == 2
    ids = sorted(r["id"] for r in ds.rows())
    assert ids == ["a", "b"]


def test_fields_contract_surfaces_low_cardinality(tmp_path: Path):
    """Placeholder-stuffing surfaces structurally: 'this field is the same
    value on most rows', regardless of what string the agent used. No
    hardcoded language tokens — works on any source.
    """
    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(12):
        # Agent "fixed" the contract by stuffing the same string everywhere.
        ds.append({"id": str(i), "decision_url": "WHATEVER_PLACEHOLDER"})

    contract = FieldsContract(required_fields=["decision_url"])
    ok, detail = contract.check(ds)
    # Field is non-null on every row → contract technically passes on
    # missing-counts, but the low-cardinality signal must surface.
    assert ok
    assert "low-cardinality" in detail
    assert "WHATEVER_PLACEHOLDER" in detail


def test_fields_contract_passes_real_constants(tmp_path: Path):
    """A short sample (<10 rows) doesn't trigger the cardinality signal."""
    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(5):
        ds.append({"id": str(i), "country_code": "FR"})
    contract = FieldsContract(required_fields=["country_code"])
    ok, detail = contract.check(ds)
    assert ok
    assert "low-cardinality" not in detail


def test_render_state_brief_returns_string(tmp_path: Path):
    from gemma_miner.prompts import render_state_brief

    ds = Dataset(tmp_path / "d.jsonl")
    memory = Memory(tmp_path / "m.json")
    state = AgentState(
        goal="goal",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=5)]),
        memory=memory,
        workdir=str(tmp_path),
    )
    reg = default_registry()
    brief = render_state_brief(state, reg)
    assert "# Goal" in brief
    assert "DISCOVER_LISTING" in brief
