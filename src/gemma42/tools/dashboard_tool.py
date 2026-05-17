"""Launch the live in-browser dashboard for the current workdir."""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


_SERVERS: dict[str, "DashboardServer"] = {}


class DashboardStartTool(Tool):
    name = "dashboard_start"
    description = (
        "Start a tiny local HTTP server on http://localhost:7777 that "
        "shows a real-time dashboard of the running workdir: row count, "
        "contracts, per-variable coverage, recent tool calls, GDT. "
        "Open the URL in any browser. The server runs until the process "
        "exits. Safe to call multiple times — only one server per workdir."
    )
    args_schema = {
        "port":   {"type": "integer", "default": 7777},
        "host":   {"type": "string",  "default": "127.0.0.1"},
        "open_browser": {"type": "boolean", "default": False},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        from gemma42.dashboard import start_dashboard

        key = str(state.workdir)
        if key in _SERVERS:
            existing = _SERVERS[key]
            return ToolResult(
                output=f"dashboard already running → http://{existing.host}:{existing.port}"
            )
        srv, url = start_dashboard(
            state.workdir,
            host=args.get("host") or "127.0.0.1",
            port=int(args.get("port") or 7777),
        )
        _SERVERS[key] = srv
        if args.get("open_browser"):
            try:
                webbrowser.open_new_tab(url)
            except Exception:  # noqa: BLE001
                pass
        return ToolResult(output=f"dashboard live → {url}")
