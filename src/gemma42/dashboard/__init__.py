"""Live in-browser dashboard for a running gemma42 workdir.

`dashboard_start(workdir)` spawns a tiny HTTP server on localhost:7777 that
streams a single-page HTML showing:
  - real-time row count + contracts
  - per-variable coverage heatmap as the codebook gets extracted
  - latest trace events
  - GDT progress
"""

from gemma42.dashboard.server import DashboardServer, start_dashboard

__all__ = ["DashboardServer", "start_dashboard"]
