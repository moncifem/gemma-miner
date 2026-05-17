"""Lightweight tests that don't require a live LLM."""

from __future__ import annotations

from pathlib import Path

from gemma42.contracts import ContractBook, FieldsContract, MinRowsContract, UniqueFieldContract
from gemma42.dataset import Dataset
from gemma42.memory import Memory
from gemma42.parsing import parse_tool_call


def test_parse_tool_call_plain():
    txt = '{"thought":"hi","tool":"http_get","args":{"url":"https://x.com"}}'
    c = parse_tool_call(txt)
    assert c.tool == "http_get" and c.args["url"] == "https://x.com"


def test_parse_tool_call_with_fence_and_prose():
    txt = "Sure, here's the call:\n```json\n{\"tool\":\"finish\",\"args\":{\"summary\":\"done\"}}\n```\nlet me know."
    c = parse_tool_call(txt)
    assert c.tool == "finish"
    assert c.args["summary"] == "done"


def test_parse_tool_call_trailing_comma():
    txt = '{"tool":"finish","args":{"summary":"done",},}'
    c = parse_tool_call(txt)
    assert c.tool == "finish"


def test_dataset_append_and_uniqueness(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl", unique_key="id")
    ok, _ = ds.append({"id": 1, "name": "a"})
    assert ok
    ok, reason = ds.append({"id": 1, "name": "again"})
    assert not ok and "duplicate" in reason
    ok, _ = ds.append({"id": 2, "name": "b"})
    assert ok and len(ds) == 2


def test_dataset_upsert_uses_implicit_id_without_unique_key(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl")
    ok, _ = ds.append({"id": "row_1", "title": "before"})
    assert ok
    ok, reason = ds.upsert({"id": "row_1", "amount": 42})
    assert ok, reason
    rows = ds.rows()
    assert len(rows) == 1
    assert rows[0]["title"] == "before"
    assert rows[0]["amount"] == 42


def test_dataset_schema_validation(tmp_path: Path):
    schema = {
        "type": "object",
        "required": ["rank", "points"],
        "properties": {
            "rank": {"type": "integer"},
            "points": {"type": "integer"},
            "domain": {"type": ["string", "null"]},
        },
    }
    ds = Dataset(tmp_path / "d.jsonl", schema=schema)
    ok, _ = ds.append({"rank": 1, "points": 100, "domain": "x.com"})
    assert ok
    ok, reason = ds.append({"rank": 2})
    assert not ok and "required" in reason
    ok, reason = ds.append({"rank": "two", "points": 1})
    assert not ok and "expected integer" in reason


def test_contracts(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl")
    book = ContractBook([MinRowsContract(min_rows=2), FieldsContract(required_fields=["a"])])
    assert not book.all_satisfied(ds)
    ds.append({"a": 1})
    ds.append({"a": 2})
    assert book.all_satisfied(ds)
    book.add(UniqueFieldContract(field="a"))
    assert book.all_satisfied(ds)
    ds.append({"a": 1})
    assert not book.all_satisfied(ds)


def test_file_refs_resolve(tmp_path: Path):
    from gemma42.refs import resolve_refs

    (tmp_path / "x.txt").write_text("HELLO WORLD")
    args = {"rows": [{"id": "1", "body": {"$file": "x.txt"}}], "n": 5}
    out = resolve_refs(args, tmp_path)
    assert out["rows"][0]["body"] == "HELLO WORLD"
    assert out["n"] == 5


def test_dataset_append_resolves_file_refs(tmp_path: Path):
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.registry import default_registry

    (tmp_path / "txt.txt").write_text("BIG PDF TEXT")
    ds = Dataset(tmp_path / "d.jsonl", unique_key="id")
    state = AgentState(
        goal="t",
        dataset=ds,
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    reg = default_registry()
    r = reg.dispatch(
        "dataset_append",
        {"rows": [{"id": "x", "pdf_text": {"$file": "txt.txt"}}]},
        state,
    )
    assert not r.error, r.output
    assert ds.rows()[0]["pdf_text"] == "BIG PDF TEXT"


def test_parser_truncation_detection():
    from gemma42.parsing import ParseError, parse_tool_call

    # A truncated JSON that has the tool name but breaks inside args.
    truncated = (
        '```json\n{\n'
        '  "thought": "appending the row now",\n'
        '  "tool": "dataset_append",\n'
        '  "args": {"rows": [{"id": "X", "pdf_text": "very long content that gets cut '
    )
    try:
        parse_tool_call(truncated)
    except ParseError as e:
        assert "TRUNCATED" in str(e)
        assert "dataset_append" in str(e)
    else:
        raise AssertionError("expected ParseError")


def test_listing_extractor_apply():
    from gemma42.tools.extractor_tool import apply_listing_spec

    html = """
    <div class="row">
      <a href="/x/1">First</a>
      <span class="id">A-1</span>
    </div>
    <div class="row">
      <a href="/x/2">Second</a>
      <span class="id">A-2</span>
    </div>
    """
    spec = {
        "row_pattern": '<div class="row">(.*?)</div>',
        "base_url": "https://example.com",
        "fields": {
            "id": {"regex": '<span class="id">([^<]+)</span>', "transform": "strip"},
            "url": {"regex": 'href="([^"]+)"', "prefix_base": True},
            "title": {"regex": ">([^<]+)</a>", "transform": "strip"},
        },
    }
    rows = apply_listing_spec(html, spec)
    assert len(rows) == 2
    assert rows[0]["id"] == "A-1"
    assert rows[0]["url"] == "https://example.com/x/1"
    assert rows[0]["title"] == "First"


def test_phase_machine(tmp_path: Path):
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    ds = Dataset(tmp_path / "d.jsonl")
    mem = Memory(tmp_path / "m.json")
    state = AgentState(
        goal="t",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=10)]),
        memory=mem,
        workdir=str(tmp_path),
    )
    # No extractor — DISCOVER_LISTING.
    assert current_phase(state, 10).name == "DISCOVER_LISTING"
    # Listing exists, queue empty — ENUMERATE.
    mem.set("extractors", {"listing": {"row_pattern": "x", "fields": {}}})
    assert current_phase(state, 10).name == "ENUMERATE"
    # Queue has items, no detail — DISCOVER_DETAIL.
    mem.set("queue", [{"id": "1"}, {"id": "2"}] + [{"id": str(i)} for i in range(3, 25)])
    assert current_phase(state, 10).name == "DISCOVER_DETAIL"
    # Detail exists — PROCESS.
    mem.set("extractors", {"listing": {"row_pattern": "x", "fields": {}}, "detail": {"fields": {}}})
    assert current_phase(state, 10).name == "PROCESS"


def test_dataset_export_works_without_codebook(tmp_path: Path):
    """User who didn't ask for a codebook should still get a parquet."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.export_tool import DatasetExportTool

    ds = Dataset(tmp_path / "d.jsonl", unique_key="id")
    for i in range(5):
        ds.append({"id": str(i), "title": f"t{i}", "score": i, "comments": i * 2})
    state = AgentState(
        goal="t", dataset=ds, contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"), workdir=str(tmp_path),
    )
    r = DatasetExportTool().run({}, state)
    assert not r.error, r.output
    assert "auto-synthesised" in r.output
    export_dir = tmp_path / "export"
    assert export_dir.exists()
    pqs = list(export_dir.glob("*.parquet"))
    assert pqs, "parquet must be written even without a codebook"
    # Codebook file should exist too (synthesised)
    assert (export_dir / "codebook.json").exists()


def test_dataset_validate_works_without_codebook(tmp_path: Path):
    """Per-column inferred stats when no codebook."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.export_tool import DatasetValidateTool

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(10):
        ds.append({"id": str(i), "title": f"x{i}", "score": i * 3, "active": True})
    state = AgentState(
        goal="t", dataset=ds, contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"), workdir=str(tmp_path),
    )
    r = DatasetValidateTool().run({}, state)
    assert not r.error, r.output
    assert "no codebook" in r.output.lower()
    assert "type=integer" in r.output or "integer" in r.output
    assert "type=string" in r.output or "string" in r.output


def test_finish_with_force_allowed_when_min_rows_met(tmp_path: Path):
    """min_rows met, required_fields fails on a few rows → finish(force=true)
    should succeed."""
    from gemma42.contracts import ContractBook, FieldsContract, MinRowsContract
    from gemma42.state import AgentState
    from gemma42.tools.finish_tool import FinishTool

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(30):
        row = {"id": str(i), "title": f"t{i}", "score": i}
        if i % 15 != 0:
            row["comments"] = i * 2
        ds.append(row)
    state = AgentState(
        goal="t", dataset=ds,
        contracts=ContractBook([
            MinRowsContract(min_rows=30),
            FieldsContract(required_fields=["title", "comments"]),
        ]),
        memory=Memory(tmp_path / "m.json"), workdir=str(tmp_path),
    )
    # Plain finish: refused (with PARTIAL hint about force=true).
    r = FinishTool().run({"summary": "done"}, state)
    assert r.error
    assert "PARTIAL" in r.output or "force=true" in r.output
    # Forced finish: succeeds.
    r2 = FinishTool().run({"summary": "30 rows; 2 missing comments", "force": True}, state)
    assert not r2.error
    assert state.finished


def test_finish_still_refuses_when_min_rows_not_met(tmp_path: Path):
    """Force=true does not bypass min_rows fundamental failure when there
    are no rows yet."""
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.state import AgentState
    from gemma42.tools.finish_tool import FinishTool

    ds = Dataset(tmp_path / "d.jsonl")
    state = AgentState(
        goal="t", dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=30)]),
        memory=Memory(tmp_path / "m.json"), workdir=str(tmp_path),
    )
    r = FinishTool().run({"summary": "done"}, state)
    assert r.error
    assert "REFUSED" in r.output


def test_extractor_define_accepts_type_coercion_transforms(tmp_path: Path):
    """transform='integer'/'int'/'number' are silently treated as no-ops."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    html = "".join(f"<row><n>{i}</n></row>" for i in range(5))
    (tmp_path / "cache" / "p.html").write_text(html)
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<row>(.*?)</row>",
            "fields": {
                "n": {"regex": r"<n>(\d+)</n>", "transform": "integer"},
            },
        },
    }, state)
    assert not r.error, r.output
    assert "matched_rows: 5" in r.output


def test_fields_contract_tolerates_naming_variants(tmp_path: Path):
    """The contract for 'number of comments' must be satisfied by a row that
    has a field named 'comments' or 'n_comments' (and vice versa)."""
    from gemma42.contracts import FieldsContract

    ds = Dataset(tmp_path / "d.jsonl")
    ds.append({"id": "1", "title": "x", "score": 10, "comments": 4})
    ds.append({"id": "2", "title": "y", "score": 5,  "comments": 0})

    # User asked for "n_comments" / "number of comments" / "comments_count"
    for required in ("n_comments", "number_of_comments", "comments_count"):
        c = FieldsContract(required_fields=["title", required])
        ok, msg = c.check(ds)
        assert ok, f"{required!r}: {msg}"

    # And the inverse — required="comments", row has "n_comments"
    ds2 = Dataset(tmp_path / "d2.jsonl")
    ds2.append({"id": "1", "title": "x", "score": 10, "n_comments": 4})
    c = FieldsContract(required_fields=["comments"])
    ok, msg = c.check(ds2)
    assert ok, msg


def test_llm_scrape_dedupes_on_push(tmp_path: Path):
    """Calling llm_scrape twice with the same rows must not inflate the dataset."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.llm_scrape_tool import LLMScrapeTool

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_args, **_kw):
            return '[{"title":"a","score":1},{"title":"b","score":2}]'

    (tmp_path / "page.html").write_text("<html><body>...</body></html>")
    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    tool = LLMScrapeTool(llm=FakeLLM())
    r1 = tool.run({"source": str(tmp_path / "page.html"),
                    "fields": ["title", "score"], "target": 5,
                    "push_to_dataset": True}, state)
    assert "duplicates skipped: 0" in r1.output
    assert len(state.dataset) == 2
    # Second call returns SAME rows → must be skipped entirely
    r2 = tool.run({"source": str(tmp_path / "page.html"),
                    "fields": ["title", "score"], "target": 5,
                    "push_to_dataset": True}, state)
    assert "duplicates skipped: 2" in r2.output
    assert len(state.dataset) == 2


def test_phase_advances_when_dataset_has_rows(tmp_path: Path):
    """If `llm_scrape` filled the dataset directly (no listing extractor saved),
    the phase machine must NOT loop back into DISCOVER_LISTING."""
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(30):
        ds.append({"id": str(i), "title": f"t{i}", "score": i})
    state = AgentState(
        goal="t",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=30)]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    phase = current_phase(state, contract_min_rows=30)
    # Dataset has 30 rows, contract met → FINISH, NOT DISCOVER_LISTING.
    assert phase.name == "FINISH", phase.name


def test_phase_stays_codebook_when_codebook_contract_fails(tmp_path: Path):
    import json

    from gemma42.contracts import CodebookContract, ContractBook, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(30):
        ds.append({"id": str(i), "text": "some decomposable text"})
    (tmp_path / "codebook.json").write_text(json.dumps({
        "name": "small",
        "description": "too small",
        "variables": [
            {"name": "has_x", "type": "boolean", "description": "x"},
        ],
    }))
    state = AgentState(
        goal="build a stats dataset",
        dataset=ds,
        contracts=ContractBook([
            MinRowsContract(min_rows=30),
            CodebookContract(min_variables=20),
        ]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    assert current_phase(state, contract_min_rows=30).name == "CODEBOOK"


def test_phase_extracts_again_when_codebook_changes(tmp_path: Path):
    import hashlib
    import json

    from gemma42.contracts import CodebookContract, ContractBook, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(30):
        ds.append({"id": str(i), "text": "source", "old_var": True})
    variables = [
        {"name": f"has_var_{i}", "type": "boolean", "description": "x"}
        for i in range(20)
    ]
    cb_path = tmp_path / "codebook.json"
    cb_path.write_text(json.dumps({
        "name": "changed",
        "description": "changed",
        "variables": variables,
    }))
    mem = Memory(tmp_path / "m.json")
    mem.set("last_extracted_codebook_hash", hashlib.sha256(b"old").hexdigest())
    state = AgentState(
        goal="build a stats dataset",
        dataset=ds,
        contracts=ContractBook([
            MinRowsContract(min_rows=30),
            CodebookContract(min_variables=20),
        ]),
        memory=mem,
        workdir=str(tmp_path),
    )
    assert current_phase(state, contract_min_rows=30).name == "EXTRACT"


def test_extractor_define_silently_strips_type_on_fields(tmp_path: Path):
    """`{"score": {"regex": "...", "type": "integer"}}` should work; the
    `type` key is a codebook concept but small models add it to extractor specs."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    html = "<html><body>" + "".join(
        f'<tr><td>{i}</td></tr>' for i in range(5)
    ) + "</body></html>"
    (tmp_path / "cache" / "p.html").write_text(html)
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "listing",
        "spec": {
            "row_pattern": "<tr>(.*?)</tr>",
            "fields": {
                "score": {"regex": r"(\d+)", "type": "integer",
                           "description": "score"},
            },
        },
    }, state)
    assert not r.error, r.output
    assert "type" in r.output.lower()  # we noted the strip
    assert "matched_rows: 5" in r.output


def test_planner_normalizes_field_names():
    """`number of comments` and `Publication Date` are normalised."""
    from gemma42.cli import _plan_with_llm

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_args, **_kw):
            return (
                '{"count": 30, '
                '"target_fields": ["title", "score (upvotes)", '
                '                   "Number Of Comments", "Publication Date"], '
                '"source_url": null, "source_hint": "Hacker News", '
                '"wants_codebook": false, "unique_field": null, "notes": ""}'
            )

    plan = _plan_with_llm("get 30 HN stories", FakeLLM())
    assert plan["target_fields"] == [
        "title", "score_upvotes", "n_comments", "publication_date",
    ]


def test_failure_log_records_parse_and_llm_errors(tmp_path: Path):
    """Every parse failure / empty llm_scrape gets logged with the full raw
    response so the user/dev can debug after the fact."""
    from gemma42.failure_log import failure_paths, log_failure

    log_failure(tmp_path, kind="parse_error", tool="_parse", turn=3,
                raw_response='{"thought":"...","tool":"X","args":{',
                payload={"reason": "TRUNCATED"})
    log_failure(tmp_path, kind="llm_scrape_empty_chunk", tool="llm_scrape",
                payload={"chunk_index": 0, "raw_response": "{}"})

    log_p, jsonl_p = failure_paths(tmp_path)
    assert log_p.exists() and jsonl_p.exists()

    # JSONL is machine-readable
    lines = jsonl_p.read_text().splitlines()
    assert len(lines) == 2
    import json as _j
    rec0 = _j.loads(lines[0])
    assert rec0["kind"] == "parse_error"
    assert rec0["raw_response"].startswith('{"thought"')
    assert rec0["raw_length"] > 0

    # Human log contains the full untruncated raw
    log_text = log_p.read_text()
    assert "parse_error" in log_text
    assert '{"thought":"...","tool":"X"' in log_text


def test_llm_scrape_logs_when_chunk_returns_zero(tmp_path: Path):
    """Empty-chunk responses go to failures.log so the agent (or human) can see why."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.llm_scrape_tool import LLMScrapeTool

    (tmp_path / "page.html").write_text("<html><body>...</body></html>")

    class FakeLLM:
        config = type("c", (), {"model": "fake",
                                  "context_window": 128_000,
                                  "max_tokens": 16_384})
        def chat(self, *_args, **_kw):
            # Return something that doesn't parse as a JSON array
            return "I'm just a prose reply, no JSON at all."

    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    tool = LLMScrapeTool(llm=FakeLLM())
    r = tool.run({"source": str(tmp_path / "page.html"),
                   "fields": [{"name": "x"}], "target": 5}, state)
    # 0 rows extracted but no exception
    assert "0 row" in r.output
    # The chunk-was-empty diagnostic must surface in the tool output
    assert "returned ZERO rows" in r.output
    assert "prose reply" in r.output  # raw preview shown
    # AND the failure was logged to disk
    fail_log = tmp_path / "failures.log"
    assert fail_log.exists()
    assert "llm_scrape_empty_chunk" in fail_log.read_text()
    assert "prose reply" in fail_log.read_text()


def test_llm_scrape_parses_envelope_and_single_object():
    """Three real Ollama response shapes must all yield rows."""
    from gemma42.tools.llm_scrape_tool import _parse_array

    # Shape A: clean array
    rows = _parse_array('[{"title": "a", "score": 1}, {"title": "b", "score": 2}]')
    assert len(rows) == 2
    # Shape B: envelope (what we now ask for)
    rows = _parse_array('{"items": [{"title": "a"}, {"title": "b"}, {"title": "c"}]}')
    assert len(rows) == 3
    # Shape C: bare single object (Ollama's collapsed shape) — must NOT
    # return 0 rows. Returns a 1-element list.
    rows = _parse_array('{"title": "Native all the way", "score": 35, "comments": 13}')
    assert len(rows) == 1
    assert rows[0]["score"] == 35


def test_reflection_parses_lessons_and_merges(tmp_path: Path):
    """`reflect()` calls the LLM, parses the lessons JSON, and merges them
    into state.lessons (deduped, capped, newest wins)."""
    from gemma42.contracts import ContractBook
    from gemma42.reflection import reflect
    from gemma42.state import AgentState, TurnRecord

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_args, **_kw):
            return '{"lessons": ["Do not call llm_scrape with source=<URL>", "Always pass cache_path to llm_scrape"]}'

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.history.append(TurnRecord(
        turn=1, thought="", tool="llm_scrape",
        args={"source": "https://example.com"},
        observation="extracted 0 row(s)", error=False,
    ))
    new = reflect(state, FakeLLM())
    assert len(new) == 2
    assert state.lessons == [
        "Do not call llm_scrape with source=<URL>",
        "Always pass cache_path to llm_scrape",
    ]
    assert state.last_reflection_turn == 1


def test_reflection_dedupes_repeat_lessons(tmp_path: Path):
    """Reflecting twice with overlapping lessons keeps the list compact."""
    from gemma42.contracts import ContractBook
    from gemma42.reflection import reflect
    from gemma42.state import AgentState, TurnRecord

    responses = iter([
        '{"lessons": ["Use cache_path not URL with llm_scrape"]}',
        '{"lessons": ["Use cache_path not URL with llm_scrape — already learned", "html_inspect needs source= for cache files"]}',
    ])

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_args, **_kw):
            return next(responses)

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.history.append(TurnRecord(turn=1, thought="", tool="x", args={}, observation="o"))
    reflect(state, FakeLLM())
    state.history.append(TurnRecord(turn=2, thought="", tool="y", args={}, observation="o"))
    reflect(state, FakeLLM())
    # The duplicate "Use cache_path not URL with llm_scrape …" is folded into one entry.
    assert len(state.lessons) == 2
    assert any("html_inspect" in l for l in state.lessons)


def test_reflection_swallows_llm_errors(tmp_path: Path):
    """If the reflection LLM call fails, the agent loop must NOT break."""
    from gemma42.contracts import ContractBook
    from gemma42.reflection import reflect
    from gemma42.state import AgentState, TurnRecord

    class BrokenLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_args, **_kw):
            raise RuntimeError("network down")

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.history.append(TurnRecord(turn=1, thought="", tool="x", args={}, observation="o"))
    new = reflect(state, BrokenLLM())
    assert new == []
    assert state.lessons == []


def test_reflection_should_reflect_cadence(tmp_path: Path):
    from gemma42.contracts import ContractBook
    from gemma42.reflection import should_reflect
    from gemma42.state import AgentState, TurnRecord

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # 0 turns → no reflection.
    assert not should_reflect(state, every=4)
    for i in range(1, 4):
        state.history.append(TurnRecord(turn=i, thought="", tool="x", args={}, observation="o"))
    assert not should_reflect(state, every=4)
    state.history.append(TurnRecord(turn=4, thought="", tool="x", args={}, observation="o"))
    assert should_reflect(state, every=4)
    state.last_reflection_turn = 4
    assert not should_reflect(state, every=4)
    for i in range(5, 9):
        state.history.append(TurnRecord(turn=i, thought="", tool="x", args={}, observation="o"))
    assert should_reflect(state, every=4)


def test_lessons_appear_in_state_brief(tmp_path: Path):
    """When state has lessons, they show up in the rendered brief."""
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.prompts import render_state_brief
    from gemma42.state import AgentState
    from gemma42.tools.registry import ToolRegistry

    state = AgentState(
        goal="g", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([MinRowsContract(min_rows=5)]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.lessons = [
        "Do not pass a URL as source= to llm_scrape",
        "Cache-hash filenames go in source=, not url=",
    ]
    registry = ToolRegistry()
    brief = render_state_brief(state, registry)
    assert "📓 Lessons" in brief
    assert "Do not pass a URL" in brief
    assert "Cache-hash filenames" in brief


def test_process_queue_text_mode_extracts_from_detail_html(tmp_path: Path, monkeypatch):
    """When the detail spec has NO attachment_url field, process_queue should
    fetch the detail page, apply the spec, and build the row from those fields
    directly — no attachment download required."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools import extractor_tool

    # Stub the network: return inline HTML for each detail URL.
    def fake_http_get(url, cache_dir, **_kw):
        body = (
            "<html><body>"
            f"<h1>Title for {url[-1]}</h1>"
            "<p class='body'>This is the detail body text.</p>"
            "</body></html>"
        ).encode("utf-8")
        path = cache_dir / f"{abs(hash(url)) % 10**12}.html"
        path.write_bytes(body)
        return path, body, "text/html"
    monkeypatch.setattr(extractor_tool, "_http_get", fake_http_get)

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("queue", [
        {"id": "1", "detail_url": "https://x.test/1", "stub": "a"},
        {"id": "2", "detail_url": "https://x.test/2", "stub": "b"},
    ])
    state.memory.set("extractors", {
        "detail": {
            "fields": {
                "title": {"regex": r"<h1>([^<]+)</h1>", "group": 1},
                "body":  {"regex": r"<p class='body'>([^<]+)</p>", "group": 1},
            },
        },
    })
    r = extractor_tool.ProcessQueueTool().run({
        "detail_extractor": "detail",
        "row_template": {
            "id":    "{queue.id}",
            "title": "{detail.title}",
            "body":  "{detail.body}",
        },
        "batch_size": 5,
        "delay_ms": 0,
    }, state)
    assert not r.error, r.output
    assert "appended=2" in r.output
    assert "errors=0" in r.output
    rows = state.dataset.rows()
    assert {r_["id"] for r_ in rows} == {"1", "2"}
    assert all("This is the detail body text." in r_["body"] for r_ in rows)
    assert all(r_["title"].startswith("Title for") for r_ in rows)


def test_process_queue_early_bails_on_repeated_errors(tmp_path: Path, monkeypatch):
    """5 consecutive failures with 0 successes should stop the batch."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools import extractor_tool

    def fake_http_get(url, cache_dir, **_kw):
        # Return a page that has NO fields extractable by the spec.
        body = b"<html><body>nothing useful</body></html>"
        path = cache_dir / f"{abs(hash(url)) % 10**12}.html"
        path.write_bytes(body)
        return path, body, "text/html"
    monkeypatch.setattr(extractor_tool, "_http_get", fake_http_get)

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("queue", [
        {"id": str(i), "detail_url": f"https://x.test/{i}"} for i in range(20)
    ])
    state.memory.set("extractors", {
        "detail": {
            "fields": {"title": {"regex": r"<h1>([^<]+)</h1>", "group": 1}},
        },
    })
    r = extractor_tool.ProcessQueueTool().run({
        "detail_extractor": "detail",
        "row_template": {"id": "{queue.id}", "title": "{detail.title}"},
        "batch_size": 20,
        "delay_ms": 0,
    }, state)
    # Bailed at 5 errors instead of grinding through all 20.
    assert "EARLY BAIL" in r.output
    assert "errors=5" in r.output


def test_dataset_from_queue_basic(tmp_path: Path):
    """Push queue items into the dataset, marking each as processed."""
    from gemma42.contracts import ContractBook, FieldsContract
    from gemma42.state import AgentState
    from gemma42.tools.dataset_tool import DatasetFromQueueTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([FieldsContract(required_fields=["date", "org"])]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("queue", [
        {"id": "1", "date": "2026-01-01", "org": "A", "decision_adopted": "fine"},
        {"id": "2", "date": "2026-02-01", "org": "B", "decision_adopted": "fine"},
        {"id": "3", "date": "2026-03-01", "org": "C", "decision_adopted": "fine"},
    ])
    r = DatasetFromQueueTool().run({}, state)
    assert not r.error, r.output
    assert "appended:           3" in r.output
    assert len(state.dataset) == 3
    rows = state.dataset.rows()
    assert {r["org"] for r in rows} == {"A", "B", "C"}
    # All ids marked processed → a second call is a no-op.
    r2 = DatasetFromQueueTool().run({}, state)
    assert "appended:           0" in r2.output


def test_dataset_from_queue_auto_renames_to_canonical(tmp_path: Path):
    """Queue uses 'comments'; FieldsContract requires 'n_comments'. Output
    rows should use the canonical 'n_comments'."""
    from gemma42.contracts import ContractBook, FieldsContract
    from gemma42.state import AgentState
    from gemma42.tools.dataset_tool import DatasetFromQueueTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([FieldsContract(required_fields=["title", "n_comments"])]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("queue", [
        {"id": "1", "title": "a", "comments": 3},
        {"id": "2", "title": "b", "comments": 5},
    ])
    r = DatasetFromQueueTool().run({}, state)
    assert not r.error, r.output
    rows = state.dataset.rows()
    assert all("n_comments" in r for r in rows)
    assert all("comments" not in r for r in rows)
    assert {r["n_comments"] for r in rows} == {3, 5}


def test_dataset_from_queue_drops_empty_when_required(tmp_path: Path):
    """A queue item with every required field empty is skipped (marked processed)."""
    from gemma42.contracts import ContractBook, FieldsContract
    from gemma42.state import AgentState
    from gemma42.tools.dataset_tool import DatasetFromQueueTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([FieldsContract(required_fields=["date"])]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("queue", [
        {"id": "1", "date": "2026-01-01"},
        {"id": "2", "date": None},
        {"id": "3", "date": ""},
    ])
    r = DatasetFromQueueTool().run({}, state)
    assert not r.error
    assert len(state.dataset) == 1
    assert "skipped (empty):    2" in r.output


def test_phase_skips_discover_detail_when_queue_has_required_fields(tmp_path: Path):
    """When queue items already contain all required fields, the phase
    machine routes to PROCESS (where dataset_from_queue lives) instead of
    DISCOVER_DETAIL."""
    from gemma42.contracts import ContractBook, FieldsContract, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([
            MinRowsContract(min_rows=2),
            FieldsContract(required_fields=["date", "org", "decision"]),
        ]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # Pretend a listing extractor exists (any non-empty spec with row_pattern).
    state.memory.set("extractors", {"x": {"row_pattern": "<tr>(.*)</tr>",
                                            "fields": {"date": {"regex": "."}}}})
    state.memory.set("queue", [
        {"id": "1", "date": "2026-01-01", "org": "A", "decision": "fine"},
        {"id": "2", "date": "2026-02-01", "org": "B", "decision": "fine"},
    ])
    phase = current_phase(state, contract_min_rows=2)
    assert phase.name == "PROCESS", phase.name


def test_http_get_flags_4xx_as_error(tmp_path: Path):
    """HTTP 403/404 should return error=True so the agent stops trying to
    scrape an error page."""
    import http.server, threading, socket
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.http_tool import HttpGetTool

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(403)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>Forbidden</body></html>")
        def log_message(self, *_a, **_k):
            pass

    # Pick a free port.
    sock = socket.socket(); sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]; sock.close()
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        state = AgentState(
            goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
            contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
            workdir=str(tmp_path),
        )
        r = HttpGetTool().run({"url": f"http://127.0.0.1:{port}/x"}, state)
        assert r.error, r.output
        assert "403" in r.output
        assert "not the page content" in r.output.lower()
    finally:
        srv.shutdown()


def test_extractor_define_accepts_fields_as_list(tmp_path: Path):
    """Model writes `fields: [{"name":"x","regex":"..."}, ...]` half the time.
    The dict form is canonical, but list form must be silently normalized."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "p.html").write_text(
        "<tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr>"
    )
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<tr>(.*?)</tr>",
            "fields": [
                {"name": "a", "regex": r"<td>([^<]+)</td>", "group": 1},
                {"name": "b", "regex": r"<td>[^<]+</td>\s*<td>([^<]+)</td>",
                  "group": 1},
            ],
        },
    }, state)
    assert not r.error, r.output
    assert "matched_rows: 2" in r.output


def test_extractor_define_rejects_css_selector_with_helpful_error(tmp_path: Path):
    """A field config like `{"selector":"td:nth-child(1)"}` must give a clear
    pointer toward regex, not an opaque 'unknown key' error."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "p.html").write_text("<tr><td>A</td></tr>")
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<tr>(.*?)</tr>",
            "fields": {"a": {"selector": "td:nth-child(1)"}},
        },
    }, state)
    assert r.error
    assert "regex" in r.output.lower()
    assert "css" in r.output.lower() or "css-" in r.output.lower()


def test_dataset_append_auto_ids_missing_ids(tmp_path: Path):
    """A row added via dataset_append without an id should get a stable
    content-hash id so downstream silver writes can find it."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.dataset_tool import DatasetAppendTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl", unique_key="id"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = DatasetAppendTool().run(
        {"rows": [
            {"title": "A", "score": 1},
            {"title": "B", "score": 2},
        ]}, state,
    )
    assert not r.error, r.output
    rows = state.dataset.rows()
    assert len(rows) == 2
    assert all(row.get("id", "").startswith("sig_") for row in rows)
    # Same content twice → same id → second append is a duplicate-reject.
    r2 = DatasetAppendTool().run(
        {"rows": [{"title": "A", "score": 1}]}, state,
    )
    # Stable id means deduped, not appended twice.
    assert len(state.dataset.rows()) == 2


def test_extract_items_surfaces_silver_upsert_failures(tmp_path: Path, monkeypatch):
    """Pilot extract_items should NOT report success when the silver upsert
    fails (e.g. bronze row has no id)."""
    from gemma42.codebook import Codebook, VariableSpec
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools import extract_items_tool

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_a, **_kw):
            return '{"score": 1}'
    monkeypatch.setattr(extract_items_tool, "_row_text",
                          lambda r, w: r.get("title", "x"))

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # Bronze rows WITHOUT ids — simulates dataset_append before this round's fix.
    state.dataset.append({"title": "A"})
    state.dataset.append({"title": "B"})
    cb = Codebook(name="t", description="", variables=[
        VariableSpec(name="score", type="integer", description=""),
    ])
    cb.save(tmp_path / "codebook.json")
    state.memory.set("codebook_path", str(tmp_path / "codebook.json"))

    r = extract_items_tool.ExtractItemsTool(llm=FakeLLM()).run({}, state)
    # The errors block should mention the silver-upsert failure.
    assert "BRONZE row has no `id`" in r.output
    # And no silver rows were actually written.
    assert len(state.extracted_dataset().rows()) == 0


def test_extractor_field_multi_returns_list(tmp_path: Path):
    """multi=true on a field config should return ALL regex matches as a list."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool, apply_listing_spec

    (tmp_path / "cache").mkdir()
    html = (
        '<div class="quote">Q1'
        '<div class="tags"><a class="tag">life</a><a class="tag">love</a>'
        '<a class="tag">truth</a></div></div>'
        '<div class="quote">Q2'
        '<div class="tags"><a class="tag">hope</a></div></div>'
    )
    (tmp_path / "cache" / "p.html").write_text(html)
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": r'<div class="quote">(.*?)(?=<div class="quote">|$)',
            "fields": {
                "tags": {
                    "regex": r'<a class="tag">([^<]+)</a>',
                    "multi": True,
                },
            },
        },
    }, state)
    assert not r.error, r.output
    saved = (state.memory.get("extractors") or {})["x"]
    rows = apply_listing_spec(html, saved)
    assert len(rows) == 2
    assert rows[0]["tags"] == ["life", "love", "truth"]
    assert rows[1]["tags"] == ["hope"]


def test_extractor_field_multi_alias_accepted(tmp_path: Path):
    """The model often writes `multiple`/`find_all`/`list`; all should rename to `multi`."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "p.html").write_text(
        '<div><a class="tag">a</a><a class="tag">b</a></div>'
    )
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": r"<div>(.*?)</div>",
            "fields": {"tags": {"regex": r'<a class="tag">([^<]+)</a>',
                                  "multiple": True}},
        },
    }, state)
    assert not r.error, r.output


def test_assess_sample_flags_broken_extraction(tmp_path: Path):
    """assess_sample should produce FIX_FIRST when a field has the same value
    on every row, when two fields collide, and when expected fields are missing."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.assess_sample_tool import AssessSampleTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # 5 broken rows: date/org both copy the first <td>, decision_text always null.
    for i in range(5):
        state.dataset.append({
            "id": str(i),
            "date":       "08/01/2026",      # same on every row
            "org":        "08/01/2026",      # collides with date
            "decision_text": None,           # always null
        })
    r = AssessSampleTool().run(
        {"expected": ["decision_text"]}, state,
    )
    assert not r.error, r.output
    assert "FIX_FIRST" in r.output
    out = r.output.lower()
    assert "same value" in out
    assert "identical values" in out
    assert "decision_text" in out


def test_assess_sample_scale_ok_on_healthy_data(tmp_path: Path):
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.assess_sample_tool import AssessSampleTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    for i in range(8):
        state.dataset.append({
            "id":            f"row_{i}",
            "title":         f"Title {i}",
            "score":         10 + i,
            "comment_count": 5 + i * 2,
        })
    r = AssessSampleTool().run({}, state)
    assert not r.error, r.output
    assert "SCALE_OK" in r.output
    assert "warnings" not in r.output.lower() or "0 more" in r.output


def test_extract_items_defaults_to_pilot(tmp_path: Path, monkeypatch):
    """First extract_items call (with no limit, silver empty) should run 3
    rows and emit a PILOT verdict in the output."""
    from gemma42.codebook import Codebook, VariableSpec
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools import extract_items_tool

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_a, **_kw):
            return '{"score": 42, "is_published": true}'
    monkeypatch.setattr(extract_items_tool, "_row_text",
                          lambda r, w: r.get("title", "x"))

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    for i in range(10):
        state.dataset.append({"id": f"r{i}", "title": f"T{i}"})
    cb = Codebook(
        name="t", description="", variables=[
            VariableSpec(name="score", type="integer", description=""),
            VariableSpec(name="is_published", type="boolean", description=""),
        ],
    )
    cb.save(tmp_path / "codebook.json")
    state.memory.set("codebook_path", str(tmp_path / "codebook.json"))

    r = extract_items_tool.ExtractItemsTool(llm=FakeLLM()).run({}, state)
    assert not r.error, r.output
    assert "PILOT verdict" in r.output
    assert state.extracted_dataset().rows().__len__() == 3
    # Run again without limit — should extract the remaining 7.
    r2 = extract_items_tool.ExtractItemsTool(llm=FakeLLM()).run({"pilot": False}, state)
    assert "PILOT" not in r2.output
    assert len(state.extracted_dataset().rows()) == 10


def test_contracts_see_joined_bronze_silver_view(tmp_path: Path):
    """The FieldsContract should pass when a required field lives only in the
    silver (extracted) dataset — the contract snapshot joins by `id`."""
    from gemma42.contracts import ContractBook, FieldsContract
    from gemma42.state import AgentState

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "dataset.jsonl"),
        contracts=ContractBook([
            FieldsContract(required_fields=["title", "fine_amount_eur"]),
        ]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # Bronze: title only.
    state.dataset.append({"id": "1", "title": "A"})
    state.dataset.append({"id": "2", "title": "B"})
    # Silver: typed columns only, keyed by id.
    silver = state.extracted_dataset()
    silver.upsert({"id": "1", "fine_amount_eur": 27_000_000})
    silver.upsert({"id": "2", "fine_amount_eur": 5_000_000})

    snapshot = state.contracts_snapshot()
    fields = next(c for c in snapshot if c["name"] == "required_fields")
    assert fields["ok"], fields


def test_extract_items_writes_to_silver_dataset_not_raw(tmp_path: Path, monkeypatch):
    """Bronze (state.dataset) keeps the raw harvest rows untouched. Silver
    (state.extracted_dataset()) gets the typed-only rows keyed by id."""
    import json as _json
    from gemma42.codebook import Codebook, VariableSpec
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools import extract_items_tool

    # Stub the LLM call so we don't need a model.
    class FakeLLM:
        config = type("c", (), {"model": "gemma4-test"})
        def chat(self, *_a, **_kw):
            return '{"fine_amount_eur": 27000000, "has_injunction": true}'
    monkeypatch.setattr(extract_items_tool, "_row_text",
                          lambda r, w: r.get("decision_text", "x"))

    # Set up a bronze dataset with one raw row.
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "dataset.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.dataset.append({
        "id": "row_0001",
        "decision_date": "08/01/2026",
        "organization_type": "OPÉRATEUR DE TÉLÉPHONIE MOBILE",
        "decision_text": "Amende administrative de 27 millions d'euros et injonction",
    })
    # Save a codebook with the variables we expect Gemma to fill in.
    cb = Codebook(
        name="cnil_test", description="", variables=[
            VariableSpec(name="fine_amount_eur", type="integer",
                          description="Fine amount in EUR"),
            VariableSpec(name="has_injunction", type="boolean",
                          description="Includes an injunction"),
        ],
    )
    cb.save(tmp_path / "codebook.json")
    state.memory.set("codebook_path", str(tmp_path / "codebook.json"))

    extract_items_tool.ExtractItemsTool(llm=FakeLLM()).run({}, state)

    # Bronze unchanged — raw fields still present, no typed cols.
    bronze = state.dataset.rows()
    assert len(bronze) == 1
    assert bronze[0]["decision_date"] == "08/01/2026"
    assert "fine_amount_eur" not in bronze[0]
    assert "has_injunction" not in bronze[0]

    # Silver has the typed row, keyed by id, NO raw fields.
    silver = state.extracted_dataset().rows()
    assert len(silver) == 1
    assert silver[0]["id"] == "row_0001"
    assert silver[0]["fine_amount_eur"] == 27000000
    assert silver[0]["has_injunction"] is True
    assert "decision_date" not in silver[0]
    assert "decision_text" not in silver[0]

    # On-disk file exists and matches.
    silver_path = tmp_path / "extracted.jsonl"
    assert silver_path.exists()
    lines = silver_path.read_text().splitlines()
    assert len(lines) == 1
    assert _json.loads(lines[0])["fine_amount_eur"] == 27000000


def test_dataset_append_resolves_file_ref_to_json_list(tmp_path: Path):
    """When dataset_append gets `rows` as a string (because the $file
    resolver expanded {"$file": "..."}), parse it as JSON and proceed."""
    import json as _json
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.dataset_tool import DatasetAppendTool

    rows_file = tmp_path / "extracted_rows.json"
    rows_file.write_text(_json.dumps([
        {"id": "1", "title": "A"},
        {"id": "2", "title": "B"},
        {"id": "3", "title": "C"},
    ]))
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # Simulate what the $file resolver would produce: the JSON TEXT of the file.
    r = DatasetAppendTool().run(
        {"rows": rows_file.read_text()}, state,
    )
    assert not r.error, r.output
    assert "added: 3/3" in r.output
    # Or a plain path string also works.
    rows_file2 = tmp_path / "more_rows.json"
    rows_file2.write_text('[{"id":"4","title":"D"}]')
    r2 = DatasetAppendTool().run({"rows": "more_rows.json"}, state)
    assert not r2.error, r2.output
    assert "added: 1/1" in r2.output
    # And a JSONL string also works.
    jsonl = '{"id":"5","title":"E"}\n{"id":"6","title":"F"}\n'
    r3 = DatasetAppendTool().run({"rows": jsonl}, state)
    assert not r3.error, r3.output
    assert "added: 2/2" in r3.output


def test_dataset_append_single_dict_shorthand(tmp_path: Path):
    """Passing one dict instead of a list of one dict should also work."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.dataset_tool import DatasetAppendTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = DatasetAppendTool().run({"rows": {"id": "1", "title": "A"}}, state)
    assert not r.error, r.output
    assert "added: 1/1" in r.output


def test_codebook_from_dict_drops_unknown_keys(tmp_path: Path):
    """Variables with foreign keys (e.g. `_adversary_note`) should not crash
    the constructor — extras are silently dropped."""
    from gemma42.codebook import Codebook

    cb = Codebook.from_dict({
        "name": "test",
        "variables": [
            {"name": "x", "type": "integer", "description": "the x",
             "_adversary_note": "internal scratch"},
            {"name": "y", "type": "boolean", "description": "the y",
             "junk_field": 123},
        ],
    })
    assert len(cb.variables) == 2
    assert cb.variables[0].name == "x"


def test_brief_warns_when_enumerate_stuck_freelancing(tmp_path: Path):
    """If a `listing` extractor is saved and the model is doing python/
    llm_scrape instead of scrape_paginated, the brief must call it out."""
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.prompts import render_state_brief
    from gemma42.state import AgentState, TurnRecord
    from gemma42.tools.registry import ToolRegistry

    state = AgentState(
        goal="g", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([MinRowsContract(min_rows=1000)]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # Add 30 rows so we leave DISCOVER_LISTING and enter ENUMERATE.
    for i in range(30):
        state.dataset.append({"id": str(i), "title": f"t{i}"})
    state.memory.set("extractors", {
        "listing": {"row_pattern": "<tr>(.*?)</tr>",
                     "fields": {"title": {"regex": "x"}}}
    })
    state.memory.set("plan", {
        "item": "HN story", "source": "paginated_html",
        "source_url": "https://news.ycombinator.com/",
        "pagination": "?p={page}",
        "items_per_page": 30, "target_rows": 1000, "pages_needed": 34,
        "harvest_strategy": "listing_only", "fields": [],
    })
    # Three turns of python/llm_scrape with no scrape_paginated.
    for i in range(3):
        state.history.append(TurnRecord(
            turn=i + 1, thought="", tool="python", args={"code": "x"},
            observation="ok", error=False,
        ))
    brief = render_state_brief(state, ToolRegistry())
    assert "ENUMERATE STUCK" in brief
    assert "scrape_paginated" in brief
    assert "1000" in brief  # target_count surfaced


def test_set_plan_saves_and_validates_math(tmp_path: Path):
    """set_plan stores a structured plan and rejects nonsense math."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.plan_tool import SetPlanTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = SetPlanTool().run({
        "item": "Hacker News story",
        "source": "paginated_html",
        "source_url": "https://news.ycombinator.com/",
        "pagination": "?p={page}",
        "items_per_page": 30,
        "target_rows": 1000,
        "pages_needed": 34,
        "harvest_strategy": "listing_only",
        "fields": [
            {"dataset_field": "title", "source_field": "title", "type": "string"},
            {"dataset_field": "score", "source_field": "score", "type": "integer"},
        ],
    }, state)
    assert not r.error, r.output
    saved = state.memory.get("plan")
    assert saved["items_per_page"] == 30
    assert saved["pages_needed"] == 34
    assert saved["harvest_strategy"] == "listing_only"

    r2 = SetPlanTool().run({
        "item": "x", "source": "paginated_html", "source_url": "u",
        "items_per_page": 30, "target_rows": 1000, "pages_needed": 1,
        "harvest_strategy": "listing_only",
    }, state)
    assert r2.error
    assert "math" in r2.output.lower()


def test_set_plan_rejects_invalid_source(tmp_path: Path):
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.plan_tool import SetPlanTool

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = SetPlanTool().run({
        "item": "x", "source": "ftp_dump", "source_url": "u",
        "harvest_strategy": "listing_only",
    }, state)
    assert r.error
    assert "ftp_dump" in r.output


def test_state_brief_surfaces_plan(tmp_path: Path):
    """When a plan exists, the brief renders a `🗺 Plan` block; if not and
    an http_get has happened, the brief shows a strong 'set_plan first' nag."""
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.prompts import render_state_brief
    from gemma42.state import AgentState, TurnRecord
    from gemma42.tools.registry import ToolRegistry

    state = AgentState(
        goal="g", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([MinRowsContract(min_rows=100)]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.history.append(TurnRecord(
        turn=1, thought="", tool="http_get", args={"url": "x"},
        observation="status: 200", error=False,
    ))
    brief = render_state_brief(state, ToolRegistry())
    assert "🗺 Plan (REQUIRED before harvesting)" in brief
    assert "call `set_plan(...)`" in brief

    state.memory.set("plan", {
        "item": "HN story", "source": "paginated_html",
        "source_url": "https://news.ycombinator.com/",
        "pagination": "?p={page}", "items_per_page": 30,
        "target_rows": 100, "pages_needed": 4,
        "harvest_strategy": "listing_only",
        "fields": [{"dataset_field": "title", "type": "string"}],
    })
    brief2 = render_state_brief(state, ToolRegistry())
    assert "🗺 Plan (stick to this" in brief2
    assert "HN story" in brief2
    assert "30/page × 4 pages" in brief2


def test_discover_assets_finds_pdf_xml_csv(tmp_path: Path):
    """The discovery tool should rank PDF/XML/CSV/archive links above ordinary
    HTML links and surface anchor text."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.discover_assets_tool import DiscoverAssetsTool

    html = """
    <html><body>
      <a href="/page2.html">Next page</a>
      <a href="/decisions/123.pdf">Décision complète</a>
      <a href="/data/annex.xml">Annexe XML</a>
      <a href="/raw/figures.csv">Download raw data CSV</a>
      <a href="/archive.tar.gz">archive</a>
      <a href="mailto:x@y.com">contact</a>
    </body></html>
    """
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = DiscoverAssetsTool().run({
        "source": html,
        "base_url": "https://example.com/listing/1",
    }, state)
    assert not r.error, r.output
    kinds = {c["kind"] for c in r.artifact["candidates"]}
    urls = {c["url"] for c in r.artifact["candidates"]}
    assert "pdf" in kinds
    assert "xml" in kinds
    assert "csv" in kinds
    # Archive present (we accept either "archive" or its specific kind).
    assert any(c["kind"] == "archive" for c in r.artifact["candidates"])
    # PDF should outscore generic HTML.
    pdf = next(c for c in r.artifact["candidates"] if c["kind"] == "pdf")
    assert pdf["score"] >= 40
    # mailto: excluded.
    assert not any("mailto:" in u for u in urls)
    # Resolved to absolute URLs.
    assert all(u.startswith("https://example.com/") for u in urls)


def test_process_queue_multi_asset_mode(tmp_path: Path, monkeypatch):
    """Multi-asset mode: each queued item's detail page is scanned, top assets
    are downloaded into items/<id>/, and the row carries an `assets` list +
    a $file ref to combined text."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools import extractor_tool

    # Fake network: detail page links to one PDF.
    def fake_http_get(url, cache_dir, **_kw):
        if url.endswith(".pdf"):
            body = b"%PDF-1.4\nfake pdf bytes\n"
            ext = ".pdf"
        else:
            body = (
                '<html><body>'
                '<a href="/files/decision.pdf">Décision complète (full text)</a>'
                '<a href="/page2.html">next</a>'
                '</body></html>'
            ).encode("utf-8")
            ext = ".html"
        path = cache_dir / f"{abs(hash(url)) % 10**12}{ext}"
        path.write_bytes(body)
        return path, body, "application/pdf" if ext == ".pdf" else "text/html"
    monkeypatch.setattr(extractor_tool, "_http_get", fake_http_get)

    # Also stub _extract_bytes so the PDF "text" is deterministic.
    def fake_extract_bytes(data, name):
        if name.endswith(".pdf"):
            return "FAKE PDF TEXT — this is the decision body.", None
        return data.decode("utf-8", errors="replace"), None
    monkeypatch.setattr(extractor_tool, "_extract_bytes", fake_extract_bytes)

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("queue", [
        {"id": "a", "detail_url": "https://example.com/i/a"},
        {"id": "b", "detail_url": "https://example.com/i/b"},
    ])
    r = extractor_tool.ProcessQueueTool().run({
        "mode": "multi_asset",
        "batch_size": 5,
        "delay_ms": 0,
        "min_asset_score": 15,
    }, state)
    assert not r.error, r.output
    rows = state.dataset.rows()
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row.get("assets"), list)
        assert len(row["assets"]) >= 1
        a0 = row["assets"][0]
        assert a0["kind"] == "pdf"
        assert "text_path" in a0
        assert a0["n_chars"] > 0
        # text_path on the row points to combined text file.
        assert row["text_path"].endswith("combined.txt")


def test_row_text_falls_back_to_short_in_row_strings(tmp_path: Path):
    """The CNIL case: rows have {date, organization_type, sanction_details}
    but none meet the heavy-text bar. `_row_text` must still return
    something so codebook_design can run on it."""
    from gemma42.tools.codebook_tool import _row_text

    row = {
        "id": "1",
        "date": "2026-01-08",
        "organization_type": "OPÉRATEUR DE TÉLÉPHONIE MOBILE",
        "sanction_details": "Amende administrative de 27 M€ et injonction",
    }
    text = _row_text(row, str(tmp_path))
    assert text != ""
    assert "sanction_details: Amende administrative" in text
    assert "organization_type:" in text
    # id should NOT leak in.
    assert "id: 1" not in text


def test_phase_routes_to_codebook_when_short_decomposable_text(tmp_path: Path):
    """A row like {decision_adopted: 'Amende 27 M€ et injonction'} is short
    but obviously decomposable into typed variables (fine amount, has_injunction).
    When wants_codebook is set, the phase machine must route to CODEBOOK."""
    from gemma42.contracts import ContractBook, CodebookContract, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(3):
        ds.append({
            "id": str(i),
            "date": f"2026-01-0{i+1}",
            "org_type": "TELECOM",
            "decision_adopted": "Amende administrative de 27 M€ et injonction.",
        })
    state = AgentState(
        goal="g", dataset=ds,
        contracts=ContractBook([
            MinRowsContract(min_rows=3),
            # Codebook contract is NOT satisfied (codebook.json doesn't exist),
            # so all_satisfied=False, and we get past the early FINISH return.
            CodebookContract(min_variables=10),
        ]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("wants_codebook", True)
    phase = current_phase(state, contract_min_rows=3)
    assert phase.name == "CODEBOOK", phase.name


def test_phase_does_not_route_to_codebook_when_disabled(tmp_path: Path):
    """Same data, but wants_codebook=False → does NOT go to CODEBOOK
    (it goes to EXPORT or FINISH depending on contracts)."""
    from gemma42.contracts import ContractBook, CodebookContract, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(3):
        ds.append({
            "id": str(i),
            "decision_adopted": "Amende administrative de 27 M€ et injonction.",
        })
    state = AgentState(
        goal="g", dataset=ds,
        contracts=ContractBook([
            MinRowsContract(min_rows=3),
            CodebookContract(min_variables=10),
        ]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set("wants_codebook", False)
    phase = current_phase(state, contract_min_rows=3)
    assert phase.name != "CODEBOOK", phase.name


def test_planner_defaults_wants_codebook_to_true(tmp_path: Path):
    """When the LLM omits wants_codebook, the planner now defaults to True
    (codebook ON), matching the system's goal of stats-ready outputs."""
    from gemma42.cli import _plan_with_llm

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_a, **_kw):
            # Note: wants_codebook is OMITTED — default must kick in.
            return '{"count": 30, "target_fields": ["date"], "source_url": null, "source_hint": "x", "unique_field": null, "notes": ""}'

    plan = _plan_with_llm("Give me a dataset of CNIL sanctions", FakeLLM())
    assert plan["wants_codebook"] is True


def test_extractor_define_auto_fixes_naive_td_regex_per_field(tmp_path: Path):
    """The exact CNIL case: the model writes the SAME <td>(.*?)</td> regex
    for date/org/decision. Instead of rejecting and asking the model to retry
    (which gemma4:8b often can't do correctly), auto-fix by anchoring each
    field on its declared column index."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    html = (
        '<tr><td>2026-01-01</td><td>Acme</td><td>Fine 1k</td></tr>'
        '<tr><td>2026-02-02</td><td>Globex</td><td>Rappel</td></tr>'
        '<tr><td>2026-03-03</td><td>Initech</td><td>Astreinte</td></tr>'
    )
    (tmp_path / "cache" / "p.html").write_text(html)
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<tr>(.*?)</tr>",
            "fields": {
                "date":             {"regex": r"<td[^>]*>(.*?)</td>"},
                "organization_type": {"regex": r"<td[^>]*>(.*?)</td>"},
                "decision_adopted":  {"regex": r"<td[^>]*>(.*?)</td>"},
            },
        },
    }, state)
    # Should NOT error — the auto-fix kicks in.
    assert not r.error, r.output
    assert "AUTO-FIXED" in r.output
    # The saved spec now has positional anchors and produces distinct values.
    extractors = state.memory.get("extractors") or {}
    saved = extractors["x"]
    # Field 0 stays bare; field 1 gets one prefix; field 2 gets two prefixes.
    assert "<td[^>]*>[^<]*</td>" not in saved["fields"]["date"]["regex"]
    assert "<td[^>]*>[^<]*</td>" in saved["fields"]["organization_type"]["regex"]
    # Decision is the 3rd field — its regex uses a {2} quantifier.
    assert "{2}" in saved["fields"]["decision_adopted"]["regex"]
    # The first row no longer has identical values across the 3 fields.
    from gemma42.tools.extractor_tool import apply_listing_spec
    rows = apply_listing_spec(html, saved)
    assert rows[0]["date"] == "2026-01-01"
    assert rows[0]["organization_type"] == "Acme"
    assert rows[0]["decision_adopted"] == "Fine 1k"


def test_extractor_define_auto_fix_handles_paired_duplicates_with_other_unique_field(tmp_path: Path):
    """The CNIL failure case: date/org/decision all use <td>(...)</td>, but
    detail_url uses a different regex. The auto-fix should rewrite the 3
    duplicated fields positionally while leaving detail_url alone."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool, apply_listing_spec

    (tmp_path / "cache").mkdir()
    html = (
        '<tr><td>2026-01-01</td><td>Acme</td><td>Fine</td>'
        '<td><a href="/a1">link</a></td></tr>'
        '<tr><td>2026-02-02</td><td>Globex</td><td>Warning</td>'
        '<td><a href="/a2">link</a></td></tr>'
    )
    (tmp_path / "cache" / "p.html").write_text(html)
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<tr>(.*?)</tr>",
            "fields": {
                "date":             {"regex": r"<td>([^<]+)</td>", "group": 1},
                "organization_type": {"regex": r"<td>([^<]+)</td>", "group": 1},
                "decision_adopted":  {"regex": r"<td>([^<]+)</td>", "group": 1},
                "detail_url":        {"regex": r'href="([^"]+)"', "group": 1},
            },
        },
    }, state)
    assert not r.error, r.output
    assert "AUTO-FIXED" in r.output
    saved = (state.memory.get("extractors") or {})["x"]
    rows = apply_listing_spec(html, saved)
    assert rows[0]["date"] == "2026-01-01"
    assert rows[0]["organization_type"] == "Acme"
    assert rows[0]["decision_adopted"] == "Fine"
    assert rows[0]["detail_url"] == "/a1"


def test_extractor_define_auto_fixes_identical_field_regexes(tmp_path: Path):
    """When every field uses the same <td>(...)</td> regex, the auto-fix
    rewrites them positionally instead of rejecting (model isn't reliably
    able to write positional anchors itself)."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool, apply_listing_spec

    (tmp_path / "cache").mkdir()
    html = (
        '<tr><td>2026-01-01</td><td>Acme Corp</td><td>Fine</td></tr>'
        '<tr><td>2026-02-02</td><td>Globex</td><td>Warning</td></tr>'
    )
    (tmp_path / "cache" / "p.html").write_text(html)
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<tr>(.*?)</tr>",
            "fields": {
                "date":  {"regex": r"<td>([^<]+)</td>", "group": 1},
                "org":   {"regex": r"<td>([^<]+)</td>", "group": 1},
                "decision": {"regex": r"<td>([^<]+)</td>", "group": 1},
            },
        },
    }, state)
    assert not r.error, r.output
    assert "AUTO-FIXED" in r.output
    saved = (state.memory.get("extractors") or {})["x"]
    rows = apply_listing_spec(html, saved)
    assert rows[0] == {"date": "2026-01-01", "org": "Acme Corp", "decision": "Fine"} or \
           rows[0]["date"] == "2026-01-01" and rows[0]["org"] == "Acme Corp" and rows[0]["decision"] == "Fine"


def test_resolve_source_arg_url_uses_cache(tmp_path: Path):
    """If `source` is a URL we've already cached, transparently use the cache."""
    import hashlib
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import _resolve_source_arg

    url = "https://example.com/listing"
    slug = hashlib.sha1(url.encode()).hexdigest()[:12]
    (tmp_path / "cache").mkdir()
    cache_path = tmp_path / "cache" / f"{slug}.html"
    cache_path.write_text("<html><body>cached</body></html>")
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    src, err = _resolve_source_arg(state, {"source": url})
    assert err is None
    assert src == str(cache_path.resolve())


def test_resolve_source_arg_url_not_cached_errors(tmp_path: Path):
    """If source is a URL not in the cache, return an error — don't silently
    treat the URL string as page content."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import _resolve_source_arg

    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    src, err = _resolve_source_arg(
        state, {"source": "https://example.com/never-fetched"}
    )
    assert src is None
    assert err and "http_get" in err


def test_resolve_source_arg_url_with_cache_prefix(tmp_path: Path):
    """`url='cache/<hash>.html'` (the prefix form the model keeps producing)
    must resolve to the file under workdir/cache/."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import _resolve_source_arg

    (tmp_path / "cache").mkdir()
    cache_file = tmp_path / "cache" / "8c4cd16a9c48.html"
    cache_file.write_text("<html><body>x</body></html>")
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    src, err = _resolve_source_arg(state, {"url": "cache/8c4cd16a9c48.html"})
    assert err is None, err
    assert src == str(cache_file.resolve())


def test_resolve_source_arg_path_in_source(tmp_path: Path):
    """Bare path or path-with-cache-prefix in source= also resolves."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import _resolve_source_arg

    (tmp_path / "cache").mkdir()
    cache_file = tmp_path / "cache" / "8c4cd16a9c48.html"
    cache_file.write_text("<html>x</html>")
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # bare filename
    src, err = _resolve_source_arg(state, {"source": "8c4cd16a9c48.html"})
    assert err is None and src == str(cache_file.resolve())
    # cache/ prefix
    src, err = _resolve_source_arg(state, {"source": "cache/8c4cd16a9c48.html"})
    assert err is None and src == str(cache_file.resolve())
    # path= variant
    src, err = _resolve_source_arg(state, {"path": "cache/8c4cd16a9c48.html"})
    assert err is None and src == str(cache_file.resolve())


def test_llm_scrape_accepts_goal_alias_for_context(tmp_path: Path):
    """Model writes `goal=...` instead of `context=...` — accept it."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.llm_scrape_tool import LLMScrapeTool

    captured: dict = {}

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, msgs, *_a, **_kw):
            captured["user"] = msgs[1]["content"]
            return '{"items":[{"title":"a"}]}'

    (tmp_path / "page.html").write_text("<html><body>x</body></html>")
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    LLMScrapeTool(llm=FakeLLM()).run({
        "source": str(tmp_path / "page.html"),
        "fields": ["title"],
        "goal": "one row per Hacker News story",
        "target": 5,
    }, state)
    assert "one row per Hacker News story" in captured["user"]


def test_scrape_paginated_auto_ids_when_extractor_lacks_id(tmp_path: Path):
    """When the extractor has no `id` field, scrape_paginated synthesizes a
    content-signature id so the queue can populate. The same row across
    multiple pages must still dedupe."""
    import json as _json
    import hashlib
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ScrapePaginatedTool

    workdir = tmp_path
    (workdir / "cache").mkdir()
    # Same HTML on every page — proper dedup must collapse to 2 items.
    body = "<html><body><tr><td>A</td></tr><tr><td>B</td></tr></body></html>"
    # Pre-cache 2 different pages with the SAME body via the _http_get slug.
    for url_path in ("/list?page=0", "/list?page=1"):
        url = "https://example.com" + url_path
        slug = hashlib.sha1(url.encode()).hexdigest()[:12]
        (workdir / "cache" / f"{slug}.html").write_text(body)
    state = AgentState(
        goal="t", dataset=Dataset(workdir / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(workdir / "m.json"),
        workdir=str(workdir),
    )
    state.memory.set("extractors", {
        "x": {
            "row_pattern": r"<tr>(.*?)</tr>",
            "fields": {"name": {"regex": r"<td>([^<]+)</td>", "group": 1}},
        },
    })
    r = ScrapePaginatedTool().run({
        "url_template": "https://example.com/list?page={page}",
        "extractor_name": "x",
        "max_pages": 2,
        "target_count": 100,
        "delay_ms": 0,
    }, state)
    assert not r.error, r.output
    queue = state.memory.get("queue") or []
    # 2 unique items deduped across the two identical pages.
    assert len(queue) == 2
    assert {q.get("name") for q in queue} == {"A", "B"}
    for q in queue:
        assert str(q.get("id", "")).startswith("sig_")


def test_resolve_source_arg_cache_hash_filename_in_url_arg(tmp_path: Path):
    """When the model passes a cache filename in `url=` (instead of `source=`),
    look it up in the cache instead of failing."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import _resolve_source_arg

    (tmp_path / "cache").mkdir()
    cache_file = tmp_path / "cache" / "abc123def456.html"
    cache_file.write_text("<html><body>x</body></html>")
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    src, err = _resolve_source_arg(state, {"url": "abc123def456.html"})
    assert err is None
    assert src == str(cache_file.resolve())


def test_llm_scrape_renames_columns_to_canonical_on_push(tmp_path: Path):
    """If the user's FieldsContract requires `n_comments` but the LLM
    extracts `comments`, the row pushed to the dataset must use the
    canonical name `n_comments`."""
    from gemma42.contracts import ContractBook, FieldsContract
    from gemma42.state import AgentState
    from gemma42.tools.llm_scrape_tool import LLMScrapeTool

    class FakeLLM:
        config = type("c", (), {"model": "fake"})
        def chat(self, *_args, **_kw):
            return '{"items":[{"title":"a","score":1,"comments":3},{"title":"b","score":2,"comments":4}]}'

    (tmp_path / "page.html").write_text("<html><body>x</body></html>")
    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([FieldsContract(required_fields=["title", "score", "n_comments"])]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    LLMScrapeTool(llm=FakeLLM()).run({
        "source": str(tmp_path / "page.html"),
        "fields": ["title", "score", "comments"],
        "target": 5,
        "push_to_dataset": True,
    }, state)
    rows = state.dataset.rows()
    assert len(rows) == 2
    assert "n_comments" in rows[0]
    assert "comments" not in rows[0]
    assert rows[0]["n_comments"] == 3


def test_extractor_define_accepts_group_index_alias(tmp_path: Path):
    """Model writes `group_index` (very common typo); should be silently renamed."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "p.html").write_text(
        "<row><n>1</n></row><row><n>2</n></row>"
    )
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<row>(.*?)</row>",
            "fields": {"n": {"regex": r"<n>(\d+)</n>", "group_index": 1}},
        },
    }, state)
    assert not r.error, r.output
    assert "matched_rows: 2" in r.output


def test_extractor_define_accepts_trim_transform_alias(tmp_path: Path):
    """`trim` is a common synonym for `strip`."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "p.html").write_text("<row>  hi  </row>" * 3)
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run({
        "name": "x",
        "spec": {
            "row_pattern": "<row>(.*?)</row>",
            "fields": {"v": {"regex": r"(.*)", "transform": "trim"}},
        },
    }, state)
    assert not r.error, r.output
    # 'hi' (whitespace stripped) appears in the output
    assert '"v": "hi"' in r.output


def test_html_find_accepts_css_selector_alias(tmp_path: Path):
    """`selector="tr.athing"` works as a shorthand for class_token='athing'+tag='tr'."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import HtmlFindTool

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "p.html").write_text(
        "<table>"
        + "".join(f'<tr class="athing">{i}</tr>' for i in range(5))
        + "</table>"
    )
    state = AgentState(
        goal="t", dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(), memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = HtmlFindTool().run(
        {"source": "p.html", "selector": "tr.athing, tr.source"},
        state,
    )
    assert not r.error, r.output
    assert "total_matches: 5" in r.output


def test_parser_autocloses_missing_braces():
    """Ollama gemma4:latest sometimes drops the final closing brace(s).
    We must auto-complete instead of giving up to TRUNCATED."""
    from gemma42.parsing import parse_tool_call

    # Real-world example from a failing run: one closing brace short.
    raw = (
        '{"thought": "x", "tool": "extractor_define", '
        '"args": {"name": "hn", "spec": {"row_pattern": "<tr>(.*?)</tr>", '
        '"fields": {"title": {"regex": "([^<]+)"}, '
        '"score": {"regex": "(\\\\d+)"}'
        # MISSING three closing braces!  fields-end + spec-end + args-end + outer-end
    )
    c = parse_tool_call(raw)
    assert c.tool == "extractor_define"
    assert "title" in c.args["spec"]["fields"]
    assert "score" in c.args["spec"]["fields"]


def test_parser_autocloses_dropped_trailing_brace_real_case():
    """The exact shape that broke in the user's run: balanced inside but
    short one final } at the very end."""
    from gemma42.parsing import parse_tool_call

    raw = (
        '{"thought": "y", "tool": "extractor_define", '
        '"args": {"name": "x", "spec": {"row_pattern": "<r>(.*?)</r>", '
        '"fields": {"a": {"regex": "(\\\\d+)", "transform": "integer"}}'
        # missing 3 closing braces (spec, args, outer)
    )
    c = parse_tool_call(raw)
    assert c.tool == "extractor_define"
    assert c.args["spec"]["fields"]["a"]["regex"] == r"(\d+)"


def test_file_ref_unwrap_for_path_arg(tmp_path: Path):
    """When model passes {"source": {"$file": "x.html"}} to html_inspect,
    the dict must be unwrapped to the path string (not file content)."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import HtmlInspectTool

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "x.html").write_text(
        "<html><body><tr><td>a</td></tr></body></html>"
    )
    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = HtmlInspectTool().run(
        {"source": {"$file": "x.html"}}, state,
    )
    # Must succeed (no KeyError), reporting the actual html content.
    assert not r.error, r.output
    assert "html_size:" in r.output
    # The size reported should match the real file, not a few chars.
    size_line = [l for l in r.output.splitlines() if l.startswith("html_size:")][0]
    n = int(size_line.split()[1])
    assert n > 30


def test_parser_accepts_python_style_single_quoted_strings():
    """Small models (Gemma 4 8B, Llama 3.1 8B) frequently emit Python-style
    single-quoted string literals inside otherwise-JSON output. We accept it."""
    from gemma42.parsing import parse_tool_call

    raw = (
        '{"thought": "x", "tool": "extractor_define", '
        '"args": {"spec": {"row_pattern": "<tr>(.*?)</tr>", '
        '"fields": {"id": {"regex": \'(\\d+)\'}, '
        '"title": {"regex": \'<a>([^<]+)</a>\'}}}}}'
    )
    c = parse_tool_call(raw)
    assert c.tool == "extractor_define"
    assert c.args["spec"]["fields"]["id"]["regex"] == r"(\d+)"
    assert c.args["spec"]["fields"]["title"]["regex"] == "<a>([^<]+)</a>"


def test_parser_repairs_invalid_regex_escapes():
    """Model writes regex strings with `\\s` in JSON; we must repair them."""
    from gemma42.parsing import parse_tool_call

    raw = (
        '```json\n'
        '{"thought": "ok", "tool": "extractor_define", '
        '"args": {"name": "listing", "spec": {"fields": '
        '{"id": {"regex": "Décision n\\u00b0\\s*([^\\s<]+)"}}}}}\n'
        '```'
    )
    c = parse_tool_call(raw)
    assert c.tool == "extractor_define"
    assert c.args["spec"]["fields"]["id"]["regex"].startswith("Décision")


def test_html_inspect_does_not_choke_on_huge_html(tmp_path: Path):
    """When given huge HTML content as `source`, html_inspect must not try
    to treat it as a filesystem path (used to crash with 'File name too long')."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import HtmlInspectTool

    big_html = "<!DOCTYPE html><html><body>" + ("<div>x</div>\n" * 5000) + "</body></html>"
    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = HtmlInspectTool().run({"source": big_html}, state)
    assert not r.error
    assert "html_size:" in r.output


def test_extractor_define_rejects_unknown_keys(tmp_path: Path):
    """Catch the row_regex/row_pattern typo case."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run(
        {
            "name": "listing",
            "spec": {
                "row_regex": "<div>(.*?)</div>",  # typo!
                "fields": {"id": {"regex": "x"}},
            },
        },
        state,
    )
    assert r.error
    assert "row_pattern" in r.output


def test_extractor_define_shows_raw_row_html_on_test(tmp_path: Path):
    """When the spec extracts rows, the output should include the raw HTML
    of row 0 so the model can diagnose bad field regexes."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    cache = tmp_path / "cache"
    cache.mkdir()
    html = (
        "<div class='row'>"
        "<a href='/x/1'>First</a>"
        "<span class='id'>A-1</span>"
        "</div>"
        "<div class='row'>"
        "<a href='/x/2'>Second</a>"
        "<span class='id'>A-2</span>"
        "</div>"
    )
    (cache / "test.html").write_text(html)
    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = ExtractorDefineTool().run(
        {
            "name": "listing",
            "spec": {
                "row_pattern": "<div class='row'>(.*?)</div>",
                "fields": {"id": {"regex": "<span class='id'>([^<]+)</span>"}},
            },
        },
        state,
    )
    assert not r.error
    assert "matched_rows: 2" in r.output
    assert "RAW HTML of row 0" in r.output
    assert "A-1" in r.output


def test_file_refs_only_resolve_for_allowed_tools(tmp_path: Path):
    """$file refs must NOT be resolved for html_inspect — it would corrupt the path arg."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.registry import default_registry

    cache = tmp_path / "cache"
    cache.mkdir()
    html = "<html><body><p>hello</p></body></html>"
    (cache / "test.html").write_text(html)

    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    reg = default_registry()
    # Pass $file ref to html_inspect — should be passed through verbatim.
    # The tool sees a dict, not a string; the resolver MUST NOT expand it.
    r = reg.dispatch(
        "html_inspect",
        {"source": {"$file": "cache/test.html"}},
        state,
    )
    # The tool will fail in some defined way (dict source is unusable) but
    # CRUCIALLY it must not raise OSError("File name too long") — i.e. the
    # dispatch layer must not have expanded the $file into 96KB of content
    # and handed it to html_inspect.
    assert "File name too long" not in r.output


def test_html_inspect_relative_path_resolves_against_workdir(tmp_path: Path):
    """Cnil bug: model passed a relative cache filename; _load was using
    process CWD, so it silently parsed the filename as 17-char HTML."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import HtmlInspectTool

    (tmp_path / "cache").mkdir()
    real_html = "<html><body>" + ("<tr><td>X</td></tr>" * 50) + "</body></html>"
    (tmp_path / "cache" / "abc123.html").write_text(real_html)

    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    # Model passes JUST the filename, not the full path.
    r = HtmlInspectTool().run({"source": "abc123.html"}, state)
    assert not r.error
    # Should report the real file size, not 13 chars (the filename length).
    assert "html_size:" in r.output
    size_line = [l for l in r.output.splitlines() if l.startswith("html_size:")][0]
    n = int(size_line.split()[1])
    assert n > 100, f"html_inspect read {n} chars; should have read the real file"


def test_html_inspect_unfindable_path_returns_clear_error(tmp_path: Path):
    """If the model passes a path-shaped string that doesn't exist anywhere,
    we should error explicitly instead of silently parsing it as inline HTML."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.html_tool import HtmlInspectTool

    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    r = HtmlInspectTool().run({"source": "nonexistent_file_12345.html"}, state)
    assert r.error
    assert "could not resolve" in r.output.lower()


def test_table_pattern_detected(tmp_path: Path):
    """CNIL-style page (one big <table> with <tr>) should be auto-detected
    and the DISCOVER_LISTING hint should offer a <tr>-based template."""
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    (tmp_path / "cache").mkdir()
    html = (
        "<html><body><table>"
        + ("<tr><td>2025-01-01</td><td>X</td></tr>" * 30)
        + "</table></body></html>"
    )
    (tmp_path / "cache" / "page.html").write_text(html)

    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([MinRowsContract(min_rows=10)]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    phase = current_phase(state, 10)
    assert phase.name == "DISCOVER_LISTING"
    # The hint must mention <tr> for this kind of page, NOT views-row.
    assert "<tr" in phase.hint
    assert "HTML <table>" in phase.hint


def test_drupal_views_pattern_detected(tmp_path: Path):
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.phases import current_phase
    from gemma42.state import AgentState

    (tmp_path / "cache").mkdir()
    html = (
        "<html><body>"
        + (
            '<div class="views-row"><div class="search-index">'
            'item</div></div>' * 10
        )
        + "</body></html>"
    )
    (tmp_path / "cache" / "page.html").write_text(html)
    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook([MinRowsContract(min_rows=10)]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    phase = current_phase(state, 10)
    assert "Drupal/Views" in phase.hint
    assert "views-row" in phase.hint


def test_scrape_paginated_single_page_when_no_placeholder(tmp_path: Path, monkeypatch):
    """If the url template has no '{page}', fetch ONCE — don't auto-paginate."""
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ScrapePaginatedTool

    # Simulate http_get by pre-populating the cache and stubbing the fetcher.
    cache = tmp_path / "cache"
    cache.mkdir()
    html = "<html><body>" + ('<div class="row">x</div>' * 5) + "</body></html>"
    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    state.memory.set(
        "extractors",
        {
            "listing": {
                "row_pattern": "<div class=\"row\">(.*?)</div>",
                "fields": {"id": {"regex": "(x)"}},
            }
        },
    )
    import gemma42.tools.extractor_tool as et

    calls = []

    def fake_http_get(url, cache_dir, **kw):
        calls.append(url)
        f = cache_dir / "stub.html"
        f.write_bytes(html.encode("utf-8"))
        return f.resolve(), html.encode("utf-8"), "text/html"

    monkeypatch.setattr(et, "_http_get", fake_http_get)
    r = ScrapePaginatedTool().run(
        {
            "url_template": "https://example.com/all",  # no {page}
            "extractor_name": "listing",
            "max_pages": 20,
            "target_count": 100,
        },
        state,
    )
    # Should have called only once, NOT 20 times.
    assert len(calls) == 1, f"expected 1 fetch, got {len(calls)}: {calls}"
    # Queue should have items.
    assert len(state.memory.get("queue", [])) > 0


def test_codebook_roundtrip(tmp_path: Path):
    from gemma42.codebook import Codebook, VariableSpec

    cb = Codebook(
        name="t",
        description="testing",
        variables=[
            VariableSpec(name="n_x", type="integer", description="count of x"),
            VariableSpec(name="is_y", type="boolean", description="y or not"),
            VariableSpec(name="cat_z", type="enum", enum_values=["a", "b"], description="z"),
            VariableSpec(name="dn_a", type="date", description="some date"),
        ],
    )
    assert cb.validate() == []
    p = tmp_path / "cb.json"
    cb.save(p)
    loaded = Codebook.load(p)
    assert loaded.name == "t"
    assert len(loaded.variables) == 4
    schema = loaded.to_json_schema()
    assert "n_x" in schema["properties"]


def test_coercion_booleans():
    from gemma42.codebook import VariableSpec
    from gemma42.coercion import coerce

    v = VariableSpec(name="b", type="boolean", description="")
    assert coerce("Yes", v) is True
    assert coerce("Oui", v) is True
    assert coerce("True", v) is True
    assert coerce("1", v) is True
    assert coerce("non", v) is False
    assert coerce("FAUX", v) is False
    assert coerce("vrai", v) is True
    assert coerce("maybe", v) is None


def test_coercion_integers():
    from gemma42.codebook import VariableSpec
    from gemma42.coercion import coerce

    v = VariableSpec(name="amount", type="integer", description="")
    assert coerce("27 millions d'euros", v) == 27_000_000
    assert coerce("1 234 567", v) == 1_234_567
    assert coerce("€ 50000", v) == 50000
    assert coerce("un million d'euros", v) == 1_000_000
    assert coerce("not a number", v) is None
    assert coerce(42, v) == 42


def test_coercion_floats_with_decimals():
    from gemma42.codebook import VariableSpec
    from gemma42.coercion import coerce

    v = VariableSpec(name="pct", type="float", description="")
    assert coerce("12,5%", v) == 12.5
    assert coerce("3.14", v) == 3.14
    assert coerce("1 234,56", v) == 1234.56


def test_coercion_dates_various_formats():
    from gemma42.codebook import VariableSpec
    from gemma42.coercion import coerce

    v = VariableSpec(name="d", type="date", description="")
    assert coerce("2026-04-16", v) == "2026-04-16"
    assert coerce("2026-04-16T12:00:00Z", v) == "2026-04-16"
    assert coerce("16/04/2026", v) == "2026-04-16"
    assert coerce("16 avril 2026", v) == "2026-04-16"
    assert coerce("April 16, 2026", v) == "2026-04-16"
    assert coerce("not a date", v) is None


def test_coercion_enums_accent_insensitive():
    from gemma42.codebook import VariableSpec
    from gemma42.coercion import coerce

    v = VariableSpec(name="cat", type="enum", description="",
                     enum_values=["NONLIEU", "SANCT", "ENG"])
    assert coerce("SANCT", v) == "SANCT"
    assert coerce("sanct", v) == "SANCT"
    assert coerce("NONLIEU", v) == "NONLIEU"
    assert coerce("Nonliëu", v) == "NONLIEU"
    assert coerce("OTHER", v) is None


def test_dataset_upsert(tmp_path: Path):
    ds = Dataset(tmp_path / "d.jsonl", unique_key="id")
    ds.append({"id": "x", "title": "first"})
    # Replace partially.
    ok, _ = ds.upsert({"id": "x", "n_words": 42, "title": None})
    assert ok
    rows = ds.rows()
    assert len(rows) == 1
    assert rows[0]["title"] == "first"   # None did not overwrite
    assert rows[0]["n_words"] == 42
    # New id → just appended.
    ok, _ = ds.upsert({"id": "y", "n_words": 5})
    assert ok
    assert len(ds) == 2


def test_codebook_contract(tmp_path: Path):
    from gemma42.contracts import CodebookContract
    from gemma42.codebook import Codebook, VariableSpec

    ds = Dataset(tmp_path / "d.jsonl")
    cb_path = tmp_path / "codebook.json"
    # Codebook missing — fails.
    c = CodebookContract(min_variables=2, min_numeric_or_boolean_ratio=0.5)
    ok, msg = c.check(ds)
    assert not ok
    # Codebook with 1 var — still fails.
    Codebook(
        name="t",
        description="",
        variables=[VariableSpec(name="n_x", type="integer", description="")],
    ).save(cb_path)
    ok, _ = c.check(ds)
    assert not ok
    # Codebook with enough numeric vars — passes.
    Codebook(
        name="t",
        description="",
        variables=[
            VariableSpec(name="n_x", type="integer", description=""),
            VariableSpec(name="is_y", type="boolean", description=""),
        ],
    ).save(cb_path)
    ok, _ = c.check(ds)
    assert ok


def test_stats_summary():
    from gemma42.codebook import Codebook, VariableSpec
    from gemma42.stats import codebook_stats

    cb = Codebook(
        name="t",
        description="",
        variables=[
            VariableSpec(name="n", type="integer", description=""),
            VariableSpec(name="b", type="boolean", description=""),
            VariableSpec(name="c", type="enum", enum_values=["x", "y"], description=""),
        ],
    )
    rows = [
        {"n": 1, "b": True, "c": "x"},
        {"n": 5, "b": True, "c": "y"},
        {"n": 9, "b": False, "c": "x"},
        {"n": None, "b": None, "c": None},
    ]
    s = codebook_stats(rows, cb)
    assert s["n_rows"] == 4
    nstats = next(x for x in s["variables"] if x["name"] == "n")
    assert nstats["coverage"] == 0.75
    assert nstats["min"] == 1
    assert nstats["max"] == 9
    cstats = next(x for x in s["variables"] if x["name"] == "c")
    assert cstats["distribution"] == {"x": 2, "y": 1}


# ── autobiography ──────────────────────────────────────────────────────────


def test_autobiography_roundtrip(tmp_path: Path):
    from gemma42.autobiography.store import Autobiography

    db = Autobiography(tmp_path / "ab.db")
    try:
        # site + recipe
        s = db.upsert_site(domain="example.com", fingerprint="abc123" * 5,
                            url_pattern="https://example.com/list")
        assert s.id is not None
        r = db.upsert_recipe(site_id=s.id, name="listing",
                              spec={"row_pattern": "<tr>"})
        assert r.id is not None
        # bump + retrieve
        db.bump_recipe(r.id, success=True)
        db.bump_recipe(r.id, success=True)
        db.bump_recipe(r.id, success=False)
        recipes = db.get_recipes(s.id)
        assert len(recipes) == 1
        assert recipes[0].n_uses == 3 and recipes[0].n_success == 2
        # fingerprint search
        sites = db.find_sites_by_fingerprint("abc123" * 5)
        assert len(sites) == 1
        # episode
        ep = db.start_episode("/tmp/wd", "test goal")
        db.finish_episode(ep.id, status="finished", n_rows=42, summary="done")
        eps = db.recent_episodes(5)
        assert eps[0].n_rows == 42
        # lessons
        l = db.add_lesson(kind="extraction", text="some lesson", confidence=0.8)
        assert l.id is not None
        hits = db.search_lessons("lesson")
        assert any(h.id == l.id for h in hits)
        # codebook
        cb = db.save_codebook(name="cb1", spec={"variables": [{"name": "x"}]},
                                domain_hint="testing")
        assert cb.id is not None
        s2 = db.search_codebooks(domain_hint="testing")
        assert any(c.id == cb.id for c in s2)
    finally:
        db.close()


# ── fingerprint ────────────────────────────────────────────────────────────


def test_fingerprint_same_template_matches():
    from gemma42.fingerprint import fingerprint_html, looks_similar

    a = (
        "<html><body><table>"
        + "".join(f"<tr><td>{i}</td><td>row {i}</td></tr>" for i in range(20))
        + "</table></body></html>"
    )
    b = (
        "<html><body><table>"
        + "".join(f"<tr><td>x</td><td>x{i}</td></tr>" for i in range(25))
        + "</table></body></html>"
    )
    c = (
        "<html><body><div class='views-row'><h2>hi</h2></div></body></html>"
    )
    fa, fb, fc = fingerprint_html(a), fingerprint_html(b), fingerprint_html(c)
    assert fa == fb or looks_similar(fa, fb, threshold=10)
    assert not looks_similar(fa, fc, threshold=3)


# ── codebook two-pass ──────────────────────────────────────────────────────


def test_codebook_two_pass_split():
    from gemma42.codebook import Codebook, VariableSpec

    cb = Codebook(
        name="t", description="",
        variables=[
            VariableSpec(name="a", type="boolean", description="", pass_=1),
            VariableSpec(name="b", type="integer", description="", pass_=2),
            VariableSpec(name="c", type="string", description=""),
        ],
    )
    p1 = cb.pass1_variables()
    p2 = cb.pass2_variables()
    assert {v.name for v in p1} == {"a", "c"}
    assert {v.name for v in p2} == {"b"}
    s1 = cb.to_json_schema_for_pass(1)
    assert "a" in s1["properties"] and "b" not in s1["properties"]


# ── constitution ───────────────────────────────────────────────────────────


def test_constitution_rules_per_row_and_cross_row():
    from gemma42.constitution import Constitution, Rule

    c = Constitution()
    c.add(Rule(name="date_in_range", kind="per_row", op="in_range",
                spec={"field": "date", "min": "2020-01-01", "max": "2026-12-31"},
                description="date in window"))
    c.add(Rule(name="implies_fine", kind="per_row", op="implies",
                spec={"when": {"field": "outcome", "eq": "SANCT"},
                      "then": {"field": "fine", "gt": 0}},
                description="if SANCT then fine>0"))
    c.add(Rule(name="unique_id", kind="cross_row", op="unique",
                spec={"fields": ["id"]}))
    rows = [
        {"id": "1", "date": "2025-01-01", "outcome": "SANCT", "fine": 1000},
        {"id": "2", "date": "2019-01-01", "outcome": "NONLIEU"},      # date out of range
        {"id": "3", "date": "2025-04-01", "outcome": "SANCT", "fine": 0},  # implies fails
        {"id": "1", "date": "2025-04-01", "outcome": "ENG"},          # dup id
    ]
    r = c.evaluate(rows)
    assert r["n_errors"] >= 3
    assert any(f["rule"] == "unique_id" for f in r["cross_row_failures"])


def test_constitution_inferred_rules():
    from gemma42.constitution import infer_rules

    rows = [{"x": 1, "cat": "a", "id": str(i), "d": f"2025-0{i % 9 + 1}-01"}
            for i in range(10)]
    for r in rows[:3]:
        r["cat"] = "a"
    for r in rows[3:]:
        r["cat"] = "b"
    proposed = infer_rules(rows)
    kinds = {(r.kind, r.op) for r in proposed}
    # numeric range on x, date range on d, enum on cat, unique on id
    assert ("per_row", "in_range") in kinds
    assert ("per_row", "enum_in") in kinds


# ── coercion already tested above ─────────────────────────────────────────


# ── provenance + diff ──────────────────────────────────────────────────────


def test_provenance_attach_and_lock_hash():
    from gemma42.provenance import LockManifest, attach_prov, make_row_prov

    row = {"id": "x", "title": "t"}
    p = make_row_prov(source_url="https://x.com/x", llm_model="m", prompt_hash="abc")
    out = attach_prov(row, p)
    assert "_prov" in out
    assert out["_prov"]["source_url"] == "https://x.com/x"

    m1 = LockManifest(goal="g", workdir="/tmp", llm_provider="p", llm_model="m")
    m2 = LockManifest(goal="g", workdir="/tmp", llm_provider="p", llm_model="m",
                       created_at=m1.created_at)
    assert m1.hash() == m2.hash()


def test_dataset_diff_added_removed_changed():
    from gemma42.provenance import diff_datasets

    a = [{"id": "1", "v": 10}, {"id": "2", "v": 20}, {"id": "3", "v": 30}]
    b = [{"id": "1", "v": 10}, {"id": "2", "v": 99}, {"id": "4", "v": 40}]
    d = diff_datasets(a, b)
    assert d["added"] == ["4"]
    assert d["removed"] == ["3"]
    assert any(c["id"] == "2" for c in d["changed"])


# ── GDT ───────────────────────────────────────────────────────────────────


def test_gdt_evaluates_against_state(tmp_path: Path):
    from gemma42.contracts import ContractBook, MinRowsContract
    from gemma42.gdt import build_tree_for_goal
    from gemma42.state import AgentState

    ds = Dataset(tmp_path / "d.jsonl", unique_key="id")
    state = AgentState(
        goal="scrape 5 items from example.com with details and build a stats dataset",
        dataset=ds,
        contracts=ContractBook([MinRowsContract(min_rows=5)]),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    tree = build_tree_for_goal(state.goal, contract_min_rows=5)
    ev = tree.evaluate(state)
    assert ev["status"] in ("pending", "in_progress")
    # add rows + extractor → corpus partially advances
    state.memory.set("extractors", {"listing": {"row_pattern": "x", "fields": {}}})
    for i in range(5):
        ds.append({"id": str(i), "title": "t"})
    ev = tree.evaluate(state)
    statuses = {c["name"]: c["status"] for c in ev["children"]}
    assert statuses["corpus"] in ("in_progress", "done")


# ── persistent kernel ──────────────────────────────────────────────────────


def test_persistent_kernel_keeps_state(tmp_path: Path):
    from gemma42.kernel import PersistentKernel

    k = PersistentKernel(tmp_path)
    try:
        r1 = k.run("x = 42\nprint('hi')")
        assert r1["status"] == "ok"
        assert "hi" in r1["stdout"]
        r2 = k.run("print(x * 2)")
        assert "84" in r2["stdout"]
    finally:
        k.close()


def test_kernel_lint_catches_requests_import_and_quote_bug():
    from gemma42.kernel import lint_snippet

    warnings = lint_snippet("import requests\nprint('hi')")
    assert any("requests" in w for w in warnings)
    bad = "import re\nr = re.findall(r'<div class=\"[^\"']*item\"', html)"
    warns = lint_snippet(bad)
    # we expect a SyntaxError-like warning
    assert any("Syntax" in w or "[^" in w for w in warns)


# ── cloud (local-cache path) ───────────────────────────────────────────────


def test_cloud_local_cache_roundtrip(tmp_path, monkeypatch):
    from gemma42 import cloud as _cl

    monkeypatch.setattr(_cl, "LOCAL_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(_cl, "DEFAULT_USER_ID_FILE", tmp_path / "uid")

    art = _cl.CloudArtifact(
        id="", kind="recipe", domain="example.com", fingerprint="abc",
        name="listing", spec={"row_pattern": "x"}, stats={"n_uses": 0},
    )
    client = _cl.CloudClient(base_url="")  # no remote
    res = client.push(art)
    assert res["status"] == "cached_local"
    found = client.search(domain="example.com")
    assert any(a.domain == "example.com" for a in found)


# ── dashboard server ───────────────────────────────────────────────────────


def test_dashboard_serves_state(tmp_path: Path):
    import json
    import urllib.request

    from gemma42.dashboard import start_dashboard

    (tmp_path / "dataset.jsonl").write_text(
        json.dumps({"id": "1", "title": "x"}) + "\n"
        + json.dumps({"id": "2", "title": "y"}) + "\n"
    )
    (tmp_path / "trace.jsonl").write_text(
        json.dumps({"event": "phase", "phase": "PROCESS"}) + "\n"
    )
    srv, url = start_dashboard(tmp_path, port=9100)
    try:
        with urllib.request.urlopen(url + "/api/state", timeout=2) as r:
            data = json.loads(r.read().decode())
        assert data["n_rows"] == 2
        assert data["phase"] == "PROCESS"
        with urllib.request.urlopen(url + "/", timeout=2) as r:
            html = r.read().decode()
        assert "gemma42" in html
    finally:
        srv.stop()


# ── codebook negative examples flow through schema ─────────────────────────


def test_variable_spec_negative_examples_in_schema():
    from gemma42.codebook import VariableSpec

    v = VariableSpec(
        name="cat_outcome", type="enum",
        description="case outcome",
        enum_values=["NONLIEU", "SANCT", "ENG"],
        positive_examples=["the Authority decided SANCT"],
        negative_examples=["dismissal during preliminary review (not SANCT)"],
        extraction_hint="pick the FINAL verdict, not intermediate steps",
    )
    schema = v.to_json_schema()
    descr = schema["description"]
    assert "EXAMPLES:" in descr and "NOT:" in descr and "HINT:" in descr


# ── self-improvement integration: recipe cache replay ─────────────────────


def test_recipe_cache_full_replay_flow(tmp_path: Path, monkeypatch):
    """v1: fingerprint a page + save a recipe. v2: fetch a similar page,
    look up the recipe, apply it — proves the recipe-cache flow is alive."""
    from gemma42.autobiography.store import Autobiography
    from gemma42.fingerprint import fingerprint_html, fingerprint_url_and_html
    from gemma42.tools.extractor_tool import apply_listing_spec

    # Page A — what v1 sees. Page B has *different content* but identical
    # structure, with the same number of rows (so frequency buckets match).
    page_a = (
        "<html><body><table>"
        + "".join(
            f'<tr class="r"><td>{i}</td><td><a href="/d/{i}">item {i}</a></td></tr>'
            for i in range(30))
        + "</table></body></html>"
    )
    page_b = (
        "<html><body><table>"
        + "".join(
            f'<tr class="r"><td>{i + 100}</td><td><a href="/d/{i + 100}">post {i + 100}</a></td></tr>'
            for i in range(30))
        + "</table></body></html>"
    )

    domain, fp_a = fingerprint_url_and_html("https://x.com/list", page_a)
    fp_b = fingerprint_html(page_b)
    # Same template → fingerprints should match exactly (or near).
    assert fp_a == fp_b or fp_a[:16] == fp_b[:16]

    # v1: save a recipe.
    db = Autobiography(tmp_path / "ab.db")
    try:
        site = db.upsert_site(domain=domain, fingerprint=fp_a, url_pattern="https://x.com/list")
        spec = {
            "row_pattern": '<tr class="r"><td>(\\d+)</td><td><a href="([^"]+)">([^<]+)</a></td></tr>',
            "fields": {
                "id":    {"regex": r"^(\d+)"},
                "url":   {"regex": r'href="([^"]+)"'},
                "title": {"regex": r">([^<]+)</a>", "transform": "strip"},
            },
        }
        recipe = db.upsert_recipe(site_id=site.id, name="listing", spec=spec, confidence=0.9)
        assert recipe.id is not None

        # v2: fetch page B, fingerprint it, look up recipe, APPLY IT.
        v2_fp = fingerprint_html(page_b)
        cached = db.find_sites_by_fingerprint(v2_fp)
        assert cached, "v2 must find an exact fingerprint match from v1"
        cached_recipes = db.get_recipes(cached[0].id)
        assert len(cached_recipes) == 1

        # Apply the cached spec directly.
        rows = apply_listing_spec(page_b, cached_recipes[0].spec)
        assert len(rows) >= 10
        # First row should be {id, url, title}
        assert "title" in rows[0] and rows[0]["title"].startswith("post")
    finally:
        db.close()


def test_unknown_transform_rejected_at_define():
    """codebook_define should refuse a spec that uses an unknown transform."""
    import tempfile
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extractor_tool import ExtractorDefineTool

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        state = AgentState(
            goal="t", dataset=Dataset(d / "ds.jsonl"),
            contracts=ContractBook(), memory=Memory(d / "m.json"),
            workdir=str(d),
        )
        r = ExtractorDefineTool().run({
            "name": "x",
            "spec": {
                "row_pattern": "<tr>(.*?)</tr>",
                "fields": {"a": {"regex": "(x)", "transform": "totally_made_up"}},
            },
        }, state)
        assert r.error
        assert "transform" in r.output.lower()


def test_multi_group_row_pattern_uses_full_match():
    """When the user's row_pattern has 2+ capture groups (e.g. <dt>(.*?)</dt><dd>(.*?)</dd>),
    we should pass the FULL match to per-field regexes, not just group 1."""
    from gemma42.tools.extractor_tool import apply_listing_spec

    html = (
        "<dt><a href='/abs/1'>1</a></dt><dd>title one author A</dd>"
        "<dt><a href='/abs/2'>2</a></dt><dd>title two author B</dd>"
    )
    spec = {
        "row_pattern": "<dt>(.*?)</dt>\\s*<dd>(.*?)</dd>",
        "fields": {
            "id":  {"regex": r"/abs/(\d+)"},
            "txt": {"regex": r"<dd>([^<]+)</dd>"},
        },
    }
    rows = apply_listing_spec(html, spec)
    assert len(rows) == 2
    assert rows[0]["id"] == "1"
    # The field regex sees the WHOLE match, so it can match across the boundary
    assert rows[0]["txt"] and "title one" in rows[0]["txt"]


# ── integration: full self-improvement walk (no LLM) ──────────────────────


def test_end_to_end_selfimprovement_pipeline(tmp_path: Path):
    """Simulates v1 → v2 with the autobiography + recipe cache. No LLM calls.

    v1: fetch a synthetic HN-like page → autobiography_recall (empty) →
        extractor_define → recipe_save → run completes.
    v2: same goal in a different workdir → autobiography_recall (HIT!) →
        fingerprint_check (EXACT match with cached recipe) → recipe adopted →
        scrape produces rows immediately.
    """
    from gemma42.autobiography.store import Autobiography
    from gemma42.fingerprint import fingerprint_html, fingerprint_url_and_html
    from gemma42.tools.extractor_tool import apply_listing_spec

    fake_hn = "<html><body><table>" + "".join(
        f'<tr class="athing submission" id="{i}"><td><a href="https://x.com/{i}">title {i}</a></td></tr>'
        f'<tr><td class="subtext"><span class="score">{100+i} points</span> | <a>{2+i} comments</a></td></tr>'
        for i in range(30)
    ) + "</table></body></html>"

    db_path = tmp_path / "global.db"
    ab = Autobiography(db_path)
    try:
        url = "https://news.ycombinator.com/"
        domain, fp = fingerprint_url_and_html(url, fake_hn)

        # v1: nothing cached yet
        sites_before = ab.find_sites_by_fingerprint(fp)
        assert sites_before == []

        # v1: agent designs an extractor and saves it
        spec = {
            "row_pattern": r'<tr class="[^"]*\bathing\b[^"]*".*?</tr>\s*<tr.*?</tr>',
            "fields": {
                "id":     {"regex": r'id="(\d+)"'},
                "title":  {"regex": r'<a href="[^"]+">([^<]+)</a>'},
                "url":    {"regex": r'<a href="([^"]+)">'},
                "score":  {"regex": r'(\d+)\s*points'},
            },
        }
        # Validate it works on v1's page
        rows = apply_listing_spec(fake_hn, spec)
        assert len(rows) == 30
        assert rows[0]["id"] == "0"
        assert rows[0]["score"] == "100"

        # Save the recipe to the autobiography
        site = ab.upsert_site(domain=domain, fingerprint=fp, url_pattern=url)
        recipe = ab.upsert_recipe(site_id=site.id, name="listing", spec=spec,
                                    confidence=0.9)
        ab.bump_recipe(recipe.id, success=True)

        # v2: different content, same template
        fake_hn_v2 = "<html><body><table>" + "".join(
            f'<tr class="athing submission" id="{500+i}"><td><a href="https://y.com/{i}">post {i}</a></td></tr>'
            f'<tr><td class="subtext"><span class="score">{200+i} points</span> | <a>{5+i} comments</a></td></tr>'
            for i in range(30)
        ) + "</table></body></html>"
        fp_v2 = fingerprint_html(fake_hn_v2)
        # Different content, same number of rows → same template → same fingerprint
        sites_v2 = ab.find_sites_by_fingerprint(fp_v2)
        assert len(sites_v2) == 1, "v2 should find v1's cached site"
        cached_recipes = ab.get_recipes(sites_v2[0].id)
        assert len(cached_recipes) == 1
        # Apply the CACHED spec directly
        rows_v2 = apply_listing_spec(fake_hn_v2, cached_recipes[0].spec)
        assert len(rows_v2) == 30
        assert rows_v2[0]["id"] == "500"
        assert rows_v2[0]["title"].startswith("post")

        # After 1 success / 0 failures, Beta(α=3, β=2) ≈ 0.6.
        # The key signal is that confidence is ≥ the 0.5 prior.
        assert cached_recipes[0].confidence >= 0.5
    finally:
        ab.close()


def test_codebook_apply_coerces_all_types(tmp_path: Path):
    """End-to-end: codebook spec → row through coercion."""
    from gemma42.codebook import Codebook, VariableSpec
    from gemma42.coercion import coerce_row

    cb = Codebook(
        name="t", description="",
        variables=[
            VariableSpec(name="n_authors", type="integer", description=""),
            VariableSpec(name="is_open", type="boolean", description=""),
            VariableSpec(name="amount_eur", type="float", description="", unit="euros"),
            VariableSpec(name="cat_outcome", type="enum", description="",
                         enum_values=["WIN", "LOSS", "DRAW"]),
            VariableSpec(name="dn_pub", type="date", description=""),
        ],
    )
    raw = {
        "n_authors":   "5 (per the paper)",
        "is_open":     "Oui",
        "amount_eur":  "27,5 millions d'euros",
        "cat_outcome": "win",   # case-insensitive match
        "dn_pub":      "16 avril 2026",
    }
    cleaned, warnings = coerce_row(raw, cb.variables)
    assert cleaned["n_authors"] == 5
    assert cleaned["is_open"] is True
    assert cleaned["amount_eur"] == 27_500_000.0
    assert cleaned["cat_outcome"] == "WIN"
    assert cleaned["dn_pub"] == "2026-04-16"


def test_constitution_inferred_rules_auto_add(tmp_path: Path):
    """rules_infer with auto_add=True actually adds rules to memory."""
    from gemma42.constitution import infer_rules
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.constitution_tool import RulesInferTool

    ds = Dataset(tmp_path / "d.jsonl")
    for i in range(12):
        ds.append({"id": str(i), "cat": "A" if i < 6 else "B", "n": i,
                    "d": f"2025-0{(i % 9) + 1}-01"})
    state = AgentState(
        goal="t", dataset=ds, contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"), workdir=str(tmp_path),
    )
    r = RulesInferTool().run({"auto_add": True}, state)
    assert not r.error
    rules = state.memory.get("rules", [])
    assert len(rules) >= 1
    names = [r["name"] for r in rules]
    assert any("range_" in n or "enum_" in n for n in names)


def test_dashboard_serves_codebook_coverage(tmp_path: Path):
    """The dashboard /api/state surfaces per-variable coverage."""
    import json
    import urllib.request

    from gemma42.dashboard import start_dashboard

    # Build a minimal workdir with a codebook + some rows
    (tmp_path / "codebook.json").write_text(json.dumps({
        "name": "t", "description": "",
        "variables": [
            {"name": "n_x", "type": "integer", "description": ""},
            {"name": "is_y", "type": "boolean", "description": ""},
        ],
    }))
    (tmp_path / "dataset.jsonl").write_text(
        json.dumps({"id": "1", "n_x": 5, "is_y": True}) + "\n"
        + json.dumps({"id": "2", "n_x": 7, "is_y": None}) + "\n"
        + json.dumps({"id": "3", "n_x": None, "is_y": False}) + "\n"
    )
    (tmp_path / "trace.jsonl").write_text(
        json.dumps({"event": "phase", "phase": "EXTRACT"}) + "\n"
    )
    srv, url = start_dashboard(tmp_path, port=9300)
    try:
        with urllib.request.urlopen(url + "/api/state", timeout=2) as resp:
            data = json.loads(resp.read().decode())
        assert data["n_rows"] == 3
        assert data["phase"] == "EXTRACT"
        cov_by_name = {c["name"]: c["coverage"] for c in data["coverage"]}
        # n_x has 2/3 = 0.67, is_y has 2/3 = 0.67
        assert abs(cov_by_name["n_x"] - 2/3) < 0.01
        assert abs(cov_by_name["is_y"] - 2/3) < 0.01
    finally:
        srv.stop()


def test_swarm_consolidator_applies_drop_merge_retype():
    """Adversary → Consolidator deterministic merge."""
    from gemma42.swarm import consolidate_codebook

    proposal = {
        "name": "test", "description": "x",
        "variables": [
            {"name": "n_apples", "type": "integer", "description": "count"},
            {"name": "n_total", "type": "integer", "description": "total"},
            {"name": "color", "type": "string", "description": "color"},
            {"name": "leaky", "type": "boolean", "description": "is_target"},
        ],
    }
    critique = {
        "drop":   [{"name": "leaky", "reason": "this is the prediction target"}],
        "merge":  [{"keep": "n_total", "into": "n_apples", "reason": "dup"}],
        "retype": [{"name": "color", "from": "string", "to": "enum",
                    "enum_values": ["red", "green", "blue"]}],
        "tighten": [],
        "approve": ["n_total"],
    }
    consolidated = consolidate_codebook(proposal, critique)
    names = [v["name"] for v in consolidated["variables"]]
    assert "leaky" not in names                  # dropped
    assert "n_apples" not in names                # merged INTO n_total
    assert "n_total" in names
    color = next(v for v in consolidated["variables"] if v["name"] == "color")
    assert color["type"] == "enum"
    assert color["enum_values"] == ["red", "green", "blue"]


def test_extract_text_dispatch(tmp_path: Path):
    from gemma42.contracts import ContractBook
    from gemma42.state import AgentState
    from gemma42.tools.extract_text_tool import ExtractTextTool

    state = AgentState(
        goal="t",
        dataset=Dataset(tmp_path / "d.jsonl"),
        contracts=ContractBook(),
        memory=Memory(tmp_path / "m.json"),
        workdir=str(tmp_path),
    )
    tool = ExtractTextTool()

    (tmp_path / "a.json").write_text('{"x": 1, "y": [2, 3]}')
    r = tool.run({"path": "a.json"}, state)
    assert not r.error and '"x": 1' in r.output

    (tmp_path / "b.html").write_text("<html><body><p>Hi <b>there</b></p><script>x=1</script></body></html>")
    r = tool.run({"path": "b.html"}, state)
    assert not r.error and "Hi" in r.output and "x=1" not in r.output

    (tmp_path / "c.csv").write_text("a,b,c\n1,2,3\n4,5,6\n")
    r = tool.run({"path": "c.csv"}, state)
    assert not r.error and "1\t2\t3" in r.output

    # zip recursion
    import zipfile
    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("inner.json", '{"k": "v"}')
    r = tool.run({"path": "d.zip"}, state)
    assert not r.error and "inner.json" in r.output and '"k": "v"' in r.output


def test_http_cache_extension():
    from gemma42.tools.http_tool import _ext_from_bytes, _ext_from_ctype, _ext_from_url

    assert _ext_from_ctype("application/json; charset=utf-8") == ".json"
    assert _ext_from_ctype("text/html") == ".html"
    assert _ext_from_ctype("application/pdf") == ".pdf"
    assert _ext_from_url("https://x.com/a/b.pdf?q=1") == ".pdf"
    assert _ext_from_url("https://x.com/a/b") == ""
    assert _ext_from_bytes(b"%PDF-1.4\n") == ".pdf"
    assert _ext_from_bytes(b"[1,2,3]") == ".json"
    assert _ext_from_bytes(b"<html><body>") == ".html"


def test_provider_factory_ollama_no_key_required():
    from gemma42.providers import list_providers, make_llm

    assert "ollama" in list_providers()
    assert "together" in list_providers()
    assert "openrouter" in list_providers()
    # Ollama doesn't require a key
    llm = make_llm("ollama", model="gemma3:27b")
    assert llm.config.base_url.endswith("11434/v1")
    assert llm.config.model == "gemma3:27b"
    # Unknown provider raises
    import pytest

    with pytest.raises(ValueError):
        make_llm("nope")


def test_default_registry_routes_schema_extraction_to_extraction_llm():
    from gemma42.tools.registry import default_registry

    class FakeLLM:
        def __init__(self, model: str):
            self.config = type("c", (), {"model": model})

    agent_llm = FakeLLM("agent")
    extraction_llm = FakeLLM("extract")
    reg = default_registry(llm=agent_llm, extraction_llm=extraction_llm)

    assert reg.get("llm_scrape").llm is agent_llm
    assert reg.get("codebook_design").llm is agent_llm
    assert reg.get("codebook_propose").llm is agent_llm
    assert reg.get("extract_items").llm is extraction_llm
    assert reg.get("extract_structured").llm is extraction_llm
    assert reg.get("codebook_test").llm is extraction_llm


def test_planner_rewrites_optional_fact_to_generic_boolean():
    from gemma42.cli import _plan_with_llm

    class FakeLLM:
        def chat(self, *_args, **_kwargs):
            return (
                '{"count": 10, "target_fields": ["title", "attachment"], '
                '"source_url": null, "source_hint": "x", '
                '"wants_codebook": true, "unique_field": null, "notes": ""}'
            )

    plan = _plan_with_llm("Collect 10 records with title and any attachment.", FakeLLM())
    assert plan["target_fields"] == ["title", "has_attachment"]


def test_memory_roundtrip(tmp_path: Path):
    m = Memory(tmp_path / "m.json")
    m.set("k", {"x": 1})
    m2 = Memory(tmp_path / "m.json")
    assert m2.get("k") == {"x": 1}
