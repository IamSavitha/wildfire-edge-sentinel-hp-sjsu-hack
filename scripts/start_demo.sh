#!/usr/bin/env bash
# Bring the full demo up on the Nano (idempotent): base7b + LoRA adapter, live metrics (8090), demo app (8095).
# Usage on the Nano:  cd ~/sentinel && ./scripts/start_demo.sh
# Then on the laptop: ssh -N -L 8095:localhost:8095 -L 8090:localhost:8090 hp11@<nano-ip>
set -euo pipefail
cd "$(dirname "$0")/.."
SOCK=/opt/hp/zrt/run/vllm-base7b.sock
ADAPTER="$PWD/adapters/context"
DETECTOR=models/joint_yolo.pt   # v1.3.0-joint (tower 0.718 / D-Fire 0.748); runs at 960 px

say() { printf '\n== %s\n' "$*"; }

say "1/4 VLM: base7b + LoRA adapter 'context'"
if curl -s -m 5 --unix-socket "$SOCK" http://localhost/v1/models 2>/dev/null | grep -q '"context"'; then
  echo "already serving base7b + context"
else
  zrt serve hf:Qwen/Qwen2.5-VL-7B-Instruct --label base7b --gpu-memory-fraction 0.45 -- \
    --max-model-len=8192 --enable-prefix-caching \
    --enable-lora --max-lora-rank=16 --lora-modules "context=$ADAPTER"
  for _ in $(seq 1 60); do
    curl -s -m 5 --unix-socket "$SOCK" http://localhost/v1/models 2>/dev/null | grep -q '"context"' && break
    sleep 10
  done
fi
curl -s -m 5 --unix-socket "$SOCK" http://localhost/v1/models | grep -q '"context"' || { echo "VLM not ready"; exit 1; }

say "2/4 Live metrics dashboard (port 8090)"
tmux has-session -t monitor 2>/dev/null || \
  tmux new-session -d -s monitor "cd $PWD && . .venv/bin/activate && python -m sentinel.monitor --port 8090 > monitor.log 2>&1"

say "3/4 Demo app (port 8095): BEFORE = YOLO-World + base7b, AFTER = $DETECTOR + LoRA"
tmux has-session -t webapp 2>/dev/null || \
  tmux new-session -d -s webapp "cd $PWD && . .venv/bin/activate && python -m sentinel.webapp --port 8095 \
    --vlm-base-url unix://$SOCK --after-weights $DETECTOR > webapp.log 2>&1"

say "4/4 Health check"
for _ in $(seq 1 30); do curl -s -m 5 localhost:8095/api/config >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 5 localhost:8095/api/config | python3 -c "
import json, sys
d = json.load(sys.stdin)
for p in d['pipelines']:
    if not p.get('optional'):
        print(f\"  {p['name']:6} detector={p['detector_weights']:28} vlm={p['vlm']:8} {'OK' if p['vlm_served'] and p['detector_available'] else 'NOT READY'}\")"
curl -s -m 5 -o /dev/null -w "  metrics dashboard: HTTP %{http_code}\n" localhost:8090/
echo
echo "Ready. On the laptop: ssh -N -L 8095:localhost:8095 -L 8090:localhost:8090 hp11@<nano-ip>"
echo "Demo app: http://localhost:8095   Live metrics: http://localhost:8090"
