"""Single-file local HTTP server that exposes a live view of a workdir.

  GET /                  → HTML dashboard (single page, no build step)
  GET /api/state          → JSON snapshot of the run (rows, contracts, codebook)
  GET /api/trace          → last 200 trace events
  GET /api/dataset[?n=]   → first N dataset rows

Auto-refreshes via the dashboard polling /api/state every 2 seconds.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>gemma42 · live</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 24px; font-family: ui-monospace,SFMono-Regular,Consolas,monospace;
         background:#0f1117; color:#d6d8e0; }
  h1 { color:#7dd3fc; margin:0 0 12px 0; font-size: 22px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .card { background:#161922; border:1px solid #1f2330; border-radius:8px; padding:16px;
          margin-bottom:16px; }
  .card h2 { margin:0 0 8px 0; font-size: 14px; color:#a3a9b8; letter-spacing:.08em;
             text-transform:uppercase; font-weight:600; }
  .big { font-size: 28px; color:#7dd3fc; }
  table { border-collapse:collapse; width:100%; font-size: 12px; }
  th, td { border-bottom:1px solid #1f2330; text-align:left; padding:4px 8px; }
  th { color:#a3a9b8; font-weight:500; }
  .bar { height: 10px; background:#1f2330; border-radius:3px; overflow:hidden; }
  .bar > div { height:100%; background:#7dd3fc; }
  .ok { color:#4ade80; }
  .fail { color:#f87171; }
  .dim { color:#737481; }
  pre { margin:0; font-size:11px; max-height:240px; overflow:auto; }
  .pill { display:inline-block; padding:2px 8px; border-radius:12px;
          background:#1f2330; font-size: 11px; margin-right:6px; }
  .phase { color:#facc15; font-weight:600; }
</style>
</head>
<body>
<h1>gemma42 · <span id="workdir" class="dim"></span></h1>
<div class="row">
  <div>
    <div class="card">
      <h2>status</h2>
      <div><span class="pill">phase: <span id="phase" class="phase">?</span></span>
           <span class="pill">step <span id="step">0</span></span>
           <span class="pill"><span id="rows" class="big">0</span> rows</span></div>
    </div>
    <div class="card">
      <h2>contracts</h2>
      <table id="contracts"><tbody></tbody></table>
    </div>
    <div class="card">
      <h2>codebook coverage</h2>
      <table id="coverage"><tbody></tbody></table>
    </div>
  </div>
  <div>
    <div class="card">
      <h2>goal tree</h2>
      <pre id="gdt"></pre>
    </div>
    <div class="card">
      <h2>recent tool calls</h2>
      <table id="trace"><tbody></tbody></table>
    </div>
  </div>
</div>
<script>
async function fetchJSON(p){ const r = await fetch(p); return r.json(); }
function setText(id,t){ document.getElementById(id).textContent = t; }
async function refresh() {
  try {
    const s = await fetchJSON('/api/state');
    setText('workdir', s.workdir || '');
    setText('phase', s.phase || '—');
    setText('step', s.step || 0);
    setText('rows', s.n_rows || 0);
    const ct = document.querySelector('#contracts tbody'); ct.innerHTML='';
    (s.contracts||[]).forEach(c=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${c.name}</td><td class="${c.ok?'ok':'fail'}">${c.ok?'✓':'✗'}</td><td class="dim">${c.detail||''}</td>`;
      ct.appendChild(tr);
    });
    const cov = document.querySelector('#coverage tbody'); cov.innerHTML='';
    (s.coverage||[]).forEach(v=>{
      const pct = Math.round(v.coverage*100);
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${v.name}</td><td><div class="bar"><div style="width:${pct}%"></div></div></td><td class="dim">${pct}%</td>`;
      cov.appendChild(tr);
    });
    setText('gdt', s.gdt || '');
    const tr = await fetchJSON('/api/trace');
    const tt = document.querySelector('#trace tbody'); tt.innerHTML='';
    (tr.events||[]).slice(-12).reverse().forEach(e=>{
      const row = document.createElement('tr');
      row.innerHTML = `<td>${e.turn||''}</td><td>${e.tool||e.event||''}</td><td class="${e.error?'fail':'ok'}">${e.error?'✗':'✓'}</td>`;
      tt.appendChild(row);
    });
  } catch(e) { console.log(e); }
}
refresh(); setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def _read_dataset(workdir: Path, limit: int) -> list:
    p = workdir / "dataset.jsonl"
    if not p.exists():
        return []
    out: list = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def _read_trace_events(workdir: Path, n: int = 200) -> list:
    p = workdir / "trace.jsonl"
    if not p.exists():
        return []
    out: list = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out[-n:]


def _read_codebook(workdir: Path) -> dict | None:
    p = workdir / "codebook.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _compute_coverage(workdir: Path) -> list[dict]:
    cb = _read_codebook(workdir)
    if not cb:
        return []
    rows = _read_dataset(workdir, limit=10_000)
    var_names = [v["name"] for v in cb.get("variables", [])]
    if not rows or not var_names:
        return [{"name": n, "coverage": 0.0} for n in var_names]
    out: list[dict] = []
    for n in var_names:
        c = sum(1 for r in rows if r.get(n) not in (None, ""))
        out.append({"name": n, "coverage": c / max(1, len(rows))})
    return out


def _latest_phase(events: list) -> str:
    for ev in reversed(events):
        if ev.get("event") == "phase":
            return ev.get("phase") or ""
    return ""


def _gdt_summary(workdir: Path) -> str:
    """Best-effort textual GDT — works without re-importing the agent state."""
    cb = _read_codebook(workdir)
    rows = _read_dataset(workdir, limit=10_000)
    n = len(rows)
    cov = _compute_coverage(workdir)
    populated = sum(1 for r in rows for c in cov if r.get(c["name"]) is not None)
    lines = [f"corpus: {n} rows"]
    if cb:
        var_count = len(cb.get("variables", []))
        lines.append(f"codebook: {var_count} variables")
        if n:
            mean_cov = sum(c["coverage"] for c in cov) / max(1, len(cov))
            lines.append(f"avg coverage: {mean_cov:.0%}")
    export = workdir / "export"
    if export.exists():
        pqs = list(export.glob("*.parquet"))
        lines.append(f"export: {len(pqs)} parquet file(s)")
    return "\n".join(lines)


class _Handler(BaseHTTPRequestHandler):
    workdir: Path = Path(".")

    def log_message(self, format, *args):
        return  # silence default access log

    def _json(self, obj: dict | list, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        wd = type(self).workdir
        if self.path == "/" or self.path == "/index.html":
            self._html(_INDEX_HTML)
            return
        if self.path.startswith("/api/state"):
            events = _read_trace_events(wd, 200)
            rows = _read_dataset(wd, limit=100_000)
            contracts: list = []
            for ev in reversed(events):
                if ev.get("event") == "turn" and ev.get("contracts"):
                    contracts = ev["contracts"]
                    break
            self._json({
                "workdir": str(wd),
                "phase": _latest_phase(events),
                "step": max((e.get("turn", 0) for e in events), default=0),
                "n_rows": len(rows),
                "contracts": contracts,
                "coverage": _compute_coverage(wd),
                "gdt": _gdt_summary(wd),
            })
            return
        if self.path.startswith("/api/trace"):
            self._json({"events": _read_trace_events(wd, 200)})
            return
        if self.path.startswith("/api/dataset"):
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            n = 20
            for p in qs.split("&"):
                if p.startswith("n="):
                    try:
                        n = int(p[2:])
                    except ValueError:
                        pass
            self._json(_read_dataset(wd, limit=n))
            return
        self.send_error(404)


class DashboardServer:
    def __init__(self, workdir: str | Path, *, host: str = "127.0.0.1", port: int = 7777):
        self.workdir = Path(workdir)
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        # find a free port starting at self.port
        port = self.port
        for _ in range(20):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                break
            except OSError:
                port += 1
        else:
            raise RuntimeError(f"could not bind any port near {self.port}")
        self.port = port
        cls = type("Handler", (_Handler,), {"workdir": self.workdir})
        self._server = ThreadingHTTPServer((self.host, port), cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None


def start_dashboard(workdir: str | Path, *, host: str = "127.0.0.1",
                    port: int = 7777) -> tuple[DashboardServer, str]:
    srv = DashboardServer(workdir, host=host, port=port)
    p = srv.start()
    return srv, f"http://{host}:{p}"
