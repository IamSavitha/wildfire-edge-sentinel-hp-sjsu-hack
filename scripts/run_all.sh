#!/usr/bin/env bash
# Starts the dispatch stub and the sentinel. Assumes the student VLM is already served on :8000.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results
python scripts/dispatch_stub.py > results/dispatch.log 2>&1 &
STUB_PID=$!
trap 'kill "$STUB_PID" 2>/dev/null || true' EXIT   # stop the dispatch stub when the sentinel exits
python -m sentinel.main
