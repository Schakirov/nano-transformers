"""
serve.py  --  the tiny web server behind http://localhost:8081
==============================================================

It does three small things:
  1. serves the repository as static files (so the site can show you the very
     source it is describing -- nanograd.py, model.py, the .cu kernels, the
     dumped PTX/SASS),
  2. streams the live training JSON in web/runs/ with no-cache headers, and
  3. lets the browser START a training run:  POST /api/train/<task>  simply
     spawns  `python3 train.py <task>`  and the dashboard polls its JSON.

No framework, ~90 lines, standard library only.   Run:  python3 serve.py
"""

import os, sys, json, subprocess, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8081"))
TASKS = ["sort", "add", "shapes"]

# track at most one live training process per task
_procs = {}
_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def log_message(self, *a):
        pass                                            # keep the console quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # This is a live-editing teaching server: never cache anything, so an
        # edit to the html/css/js (or a fresh training-run JSON) always shows
        # on reload instead of a stale copy from the browser cache.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/web/index.html"
        elif self.path == "/api/status":
            return self._json(self._status())
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/train/"):
            task = self.path.rsplit("/", 1)[-1]
            if task not in TASKS:
                return self._json({"error": "unknown task"}, 400)
            return self._json(self._start(task))
        self._json({"error": "not found"}, 404)

    # -- helpers ----------------------------------------------------------
    def _running(self, task):
        p = _procs.get(task)
        return p is not None and p.poll() is None

    def _status(self):
        st = {}
        for t in TASKS:
            run = os.path.join(ROOT, "web", "runs", f"{t}.json")
            st[t] = {"running": self._running(t), "has_run": os.path.exists(run)}
        return st

    def _start(self, task):
        with _lock:
            if self._running(task):
                return {"ok": True, "already": True}
            p = subprocess.Popen([sys.executable, "train.py", task],
                                 cwd=ROOT, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT)
            _procs[task] = p
        return {"ok": True, "started": True}


if __name__ == "__main__":
    os.chdir(ROOT)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"nano-transformers serving on http://localhost:{PORT}")
    print("open it, pick a task, and press Train.")
    httpd.serve_forever()
