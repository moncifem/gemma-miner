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
    assert "could not find" in r.output.lower()


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
    # Ollama doesn't require a key
    llm = make_llm("ollama", model="gemma3:27b")
    assert llm.config.base_url.endswith("11434/v1")
    assert llm.config.model == "gemma3:27b"
    # Unknown provider raises
    import pytest

    with pytest.raises(ValueError):
        make_llm("nope")


def test_memory_roundtrip(tmp_path: Path):
    m = Memory(tmp_path / "m.json")
    m.set("k", {"x": 1})
    m2 = Memory(tmp_path / "m.json")
    assert m2.get("k") == {"x": 1}
