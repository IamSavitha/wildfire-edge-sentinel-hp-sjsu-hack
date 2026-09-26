"""The container's launch command matches the console's real options, and everything it copies exists."""
import json
import re
from pathlib import Path

import pytest

from sentinel.console import parser

ROOT = Path(__file__).parents[1]
if not all((ROOT / f).exists() for f in ("Dockerfile", "docker-compose.yml", ".dockerignore")):
    pytest.skip("build files not here (inside the image)", allow_module_level=True)
DOCKERFILE = (ROOT / "Dockerfile").read_text()


def cmd() -> list[str]:
    m = re.search(r"^CMD (\[.*?\])\s*$", DOCKERFILE.replace("\\\n", ""), re.M | re.S)
    return json.loads(m.group(1))


def test_cmd_runs_the_console_with_valid_options():
    args = cmd()
    assert args[:3] == ["python", "-m", "sentinel.console"]
    a = parser().parse_args(args[3:])
    assert a.host == "0.0.0.0" and a.port == 8080 and a.state_dir == "/state"
    assert a.vlm_base_url == "unix:///opt/hp/zrt/run/vllm-base7b.sock"


def test_copied_paths_exist_and_weights_stay_out_of_the_image():
    for src in re.findall(r"^COPY (\S+)", DOCKERFILE, re.M):
        assert (ROOT / src).exists(), src
    ignore = (ROOT / ".dockerignore").read_text().split()
    assert {"data/", "models/", "*.pt", ".git", ".venv", "*.env"} <= set(ignore)


def test_compose_mounts_what_the_command_expects():
    compose = (ROOT / "docker-compose.yml").read_text()
    for target in ("/app/models", "/app/data", "/assets", "/state", "/app/results", "/app/weights",
                   "/opt/sentinel/yolov8s-worldv2.pt", "/opt/hp/zrt/run"):
        assert f":{target}" in compose, target
    for path in re.findall(r'"(/(?:app|opt|assets|state)[^"]*)"', DOCKERFILE):
        if path.startswith(("/app/models/", "/opt/sentinel/")):
            assert path.rsplit("/", 1)[0] in compose or path in compose, path


def test_serve_requirements_are_pinned():
    lines = [ln for ln in (ROOT / "requirements-serve.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    loose = [ln for ln in lines if "==" not in ln and "@" not in ln and ln != "pytest"]
    assert not loose, loose
