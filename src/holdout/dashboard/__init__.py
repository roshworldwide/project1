"""The local-first dashboard: a read-only window onto the run store."""

from holdout.dashboard.server import make_server, serve

__all__ = ["make_server", "serve"]
