"""The local dashboard server: read-only, localhost-only, zero new deps.

Serves the bundled SPA plus a small JSON API over the local run store.
Design contract (the air-gapped moat):

- binds 127.0.0.1 only — never an external interface;
- read-only — no endpoint mutates runs, the ledger, or anything else;
- no model calls, no telemetry, 0 bytes leave the machine;
- stdlib ``http.server`` — installing holdout never drags in a web stack.

API (all JSON):

- ``GET /api/meta``                          — version, store path, run count
- ``GET /api/runs``                          — run listing with metric estimates
- ``GET /api/runs/<run_id>``                 — one full run with metrics
- ``GET /api/compare?baseline=X&candidate=Y[&alpha=A]`` — a RunComparison
- ``GET /api/ledger``                        — holdout-discipline status per eval
"""

import json
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import holdout
from holdout.leakage.ledger import HoldoutLedger
from holdout.regression.compare import compare
from holdout.store.run_store import RunStore

# Resamples for list-view estimates: cheaper than the single-run default but
# still honest; the method string travels with every estimate either way.
_LIST_RESAMPLES = 2_000

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".png": "image/png",
    ".woff2": "font/woff2",
    ".ico": "image/x-icon",
}

_MISSING_ASSETS = (
    "<!doctype html><meta charset='utf-8'><title>holdout dashboard</title>"
    "<body style='font-family:system-ui;background:#0E0E10;color:#F5F5F2;"
    "display:grid;place-items:center;height:100vh'><div>"
    "<h1>dashboard assets not bundled</h1>"
    "<p>This is a source checkout without the built SPA. Run "
    "<code>cd dashboard && npm install && npm run build</code> and retry."
    "</p></div></body>"
)


class _Api:
    """Read-only API over a run store (separated from HTTP for testability)."""

    def __init__(self, store: RunStore) -> None:
        self.store = store
        self.ledger = HoldoutLedger(store.root)
        self._metrics_cache: dict[str, dict[str, object]] = {}

    def _metrics(self, run_id: str, n_resamples: int) -> dict[str, object]:
        # Runs are content-addressed, so this cache can never go stale.
        cached = self._metrics_cache.get(run_id)
        if cached is None:
            run = self.store.load(run_id)
            cached = {k: v.to_dict() for k, v in run.metrics(n_resamples=n_resamples).items()}
            self._metrics_cache[run_id] = cached
        return cached

    def meta(self) -> dict[str, object]:
        """Version and store identity."""
        return {
            "version": holdout.__version__,
            "store": str(self.store.root),
            "n_runs": len(self.store),
        }

    def runs(self) -> dict[str, object]:
        """All stored runs, newest first, with their metric estimates."""
        out = []
        for info in self.store.runs():
            out.append(
                {
                    "run_id": info.run_id,
                    "short_run_id": info.short_run_id,
                    "eval_name": info.eval_name,
                    "target_name": info.target_name,
                    "created_at": info.created_at,
                    "n_cases": info.n_cases,
                    "n_errors": info.n_errors,
                    "seed": info.seed,
                    "metrics": self._metrics(info.run_id, _LIST_RESAMPLES),
                }
            )
        return {"runs": out}

    def run_detail(self, ref: str) -> dict[str, object]:
        """One full run, including per-case results and metrics."""
        run = self.store.load(ref)
        payload = run.to_dict()
        payload["short_run_id"] = run.short_run_id
        payload["n_errors"] = run.n_errors
        payload["metrics"] = {k: v.to_dict() for k, v in run.metrics().items()}
        return payload

    def compare(self, baseline: str, candidate: str, alpha: float) -> dict[str, object]:
        """Compute a full statistical comparison on demand (nothing stored)."""
        a = self.store.load(baseline)
        b = self.store.load(candidate)
        return compare(a, b, alpha=alpha).to_dict()

    def ledger_status(self, budget: int = 20) -> dict[str, object]:
        """Holdout-discipline status for every eval present in the store."""
        seen: dict[str, str] = {}
        for info in self.store.runs():
            run = self.store.load(info.run_id)
            seen.setdefault(run.eval_fingerprint, run.eval_name)
        return {
            "evals": [
                self.ledger.check(fp, name, budget=budget).to_dict() for fp, name in seen.items()
            ]
        }


def _assets_root() -> Path | None:
    """Locate the bundled SPA, if it was built."""
    candidate = resources.files("holdout") / "dashboard_dist"
    path = Path(str(candidate))
    return path if (path / "index.html").exists() else None


class _Handler(BaseHTTPRequestHandler):
    """Routes /api/* to the API and everything else to the bundled SPA."""

    # Injected via functools.partial in serve().
    def __init__(self, *args: object, api: _Api, assets: Path | None, **kwargs: object) -> None:
        self.api = api
        self.assets = assets
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence per-request logging (the CLI prints the URL once)."""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, object], code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        """Serve one read-only GET request."""
        url = urlparse(self.path)
        try:
            if url.path.startswith("/api/"):
                self._handle_api(url.path, parse_qs(url.query))
            else:
                self._handle_static(url.path)
        except KeyError as exc:
            self._send_json({"error": str(exc.args[0])}, code=404)
        except (ValueError, OSError) as exc:
            self._send_json({"error": str(exc)}, code=400)

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/meta":
            self._send_json(self.api.meta())
        elif path == "/api/runs":
            self._send_json(self.api.runs())
        elif path.startswith("/api/runs/"):
            self._send_json(self.api.run_detail(path.removeprefix("/api/runs/")))
        elif path == "/api/compare":
            baseline = query.get("baseline", [""])[0]
            candidate = query.get("candidate", [""])[0]
            alpha = float(query.get("alpha", ["0.05"])[0])
            if not baseline or not candidate:
                self._send_json({"error": "baseline and candidate are required"}, code=400)
                return
            self._send_json(self.api.compare(baseline, candidate, alpha))
        elif path == "/api/ledger":
            budget = int(query.get("budget", ["20"])[0])
            self._send_json(self.api.ledger_status(budget))
        else:
            self._send_json({"error": f"unknown endpoint {path}"}, code=404)

    def _handle_static(self, path: str) -> None:
        if self.assets is None:
            self._send(503, _MISSING_ASSETS.encode("utf-8"), "text/html; charset=utf-8")
            return
        name = path.lstrip("/") or "index.html"
        file = (self.assets / name).resolve()
        # Path traversal guard + SPA fallback: unknown routes get index.html.
        if not file.is_relative_to(self.assets.resolve()) or not file.is_file():
            file = self.assets / "index.html"
        mime = _MIME.get(file.suffix, "application/octet-stream")
        self._send(200, file.read_bytes(), mime)


def make_server(store: RunStore, *, port: int = 0) -> ThreadingHTTPServer:
    """Build the localhost-only server (port 0 = pick a free port)."""
    handler = partial(_Handler, api=_Api(store), assets=_assets_root())
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(store: RunStore, *, port: int = 4321, open_browser: bool = True) -> None:
    """Run the dashboard until interrupted."""
    server = make_server(store, port=port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"holdout dashboard → {url}  (read-only, local, Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
