"""Console UI: valid JS, nothing loaded from the internet, and every API path it calls exists on the server."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount

from tests.test_console import console

UI = Path(__file__).parents[1] / "sentinel" / "static" / "console"
FILES = sorted(p for p in UI.rglob("*") if p.suffix in (".js", ".html", ".css"))
JS = [p for p in FILES if p.suffix == ".js"]
SVG_NS = "http://www.w3.org/2000/svg"          # an XML namespace name inside a data: URI, never fetched


def test_the_ui_exists():
    assert (UI / "index.html").exists() and len(JS) >= 8


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("path", JS, ids=lambda p: p.name)
def test_js_parses(path):
    # parsed as an ES module whatever the node version (node 18 treats a bare .js file as CommonJS)
    r = subprocess.run(["node", "--input-type=module", "--check"], input=path.read_text(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_nothing_is_loaded_from_the_internet(path):
    text = path.read_text().replace(SVG_NS, "")
    assert not re.search(r"https?://", text), f"{path.name} references an external URL; the console must work offline"
    assert "fonts.googleapis" not in text and "cdn" not in text.lower()


def routes(app, prefix=""):
    out = []
    for r in app.routes:
        if isinstance(r, Mount):
            if getattr(r.app, "routes", None) is not None and r.path in ("/ops", "/lab", ""):
                out += routes(r.app, prefix + r.path)
        elif getattr(r, "path", None):
            out.append(prefix + r.path)
    return out


def api_paths():
    """'/api/...', '/ops/api/...', '/lab/api/...' literals in the UI, with ${...} as a path parameter."""
    found = set()
    for p in JS:
        for m in re.finditer(r"""["'`](/(?:ops/|lab/)?api/[^"'`?]*)""", p.read_text()):
            path = re.sub(r"\$\{[^}]*\}", "X", m.group(1)).rstrip("/")
            found.add(path)
    return sorted(found)


def test_every_api_path_the_ui_calls_exists(tmp_path):
    app, _, _, _ = console(tmp_path)
    patterns = [re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", r) + "$") for r in routes(app)]
    paths = api_paths()
    assert len(paths) >= 15
    missing = [p for p in paths if not any(rx.match(p) for rx in patterns)]
    assert not missing, f"UI calls paths the server does not have: {missing}"


def test_home_is_the_officer_console_with_the_extra_tabs(tmp_path):
    app, _, _, _ = console(tmp_path)
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Forest officer console" in r.text
        assert r.text.count("/console/officer_ext.js") == 1 and "/console/officer_ext.css" in r.text
        for asset in ("/console/officer_ext.js", "/console/officer_ext.css", "/console/app.js"):
            assert c.get(asset).status_code == 200, asset
        assert "Sentinel Console" in c.get("/console/").text             # the compact console stays available
        assert c.get("/api/config").json()["cameras"]                     # the officer page's own API, at the root
        assert c.get("/api/overview").json()["node"] == "Test node"       # console routes win over the root mount


def test_no_demo_wording_in_the_ui():
    for p in FILES:
        assert not re.search(r"\bdemo\b", p.read_text(), re.I), f"{p.name} still says demo"
