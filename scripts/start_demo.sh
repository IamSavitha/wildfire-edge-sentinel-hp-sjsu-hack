#!/usr/bin/env bash
# Bring the full demo up on the Nano (idempotent): base7b + LoRA adapter, live metrics (8090), demo app (8095),
# incident map (8100; an instance already on 8100 is reused).
# Usage on the Nano:  cd ~/sentinel && ./scripts/start_demo.sh
# Then on the laptop: ssh -N -L 8095:localhost:8095 -L 8090:localhost:8090 -L 8100:localhost:8100 hp11@<nano-ip>
set -euo pipefail
cd "$(dirname "$0")/.."
SOCK=/opt/hp/zrt/run/vllm-base7b.sock
ADAPTER="$PWD/adapters/context"
DETECTOR=models/joint_yolo.pt   # v1.3.0-joint (tower 0.718 / D-Fire 0.748); runs at 960 px
# Phone alerts for the live camera (ntfy): `export NTFY_TOPIC_URL=https://ntfy.sh/<topic>` in this file,
# chmod 600. Never printed. A webapp that is already running keeps its old environment: restart it with
# `tmux kill-session -t webapp` and re-run this script.
ALERTS_ENV="$HOME/.sentinel_alerts.env"

say() { printf '\n== %s\n' "$*"; }
listening() { ss -ltnH "sport = :$1" 2>/dev/null | grep -q . || (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

say "1/5 VLM: base7b + LoRA adapter 'context'"
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

say "2/5 Live metrics dashboard (port 8090)"
tmux has-session -t monitor 2>/dev/null || \
  tmux new-session -d -s monitor "cd $PWD && . .venv/bin/activate && python -m sentinel.monitor --port 8090 > monitor.log 2>&1"

say "3/5 Demo app (port 8095): BEFORE = YOLO-World + base7b, AFTER = $DETECTOR + LoRA, live camera"
NTFY_TOPIC_URL=""
if [ -f "$ALERTS_ENV" ]; then
  # shellcheck disable=SC1090
  . "$ALERTS_ENV"
fi
if [ -n "${NTFY_TOPIC_URL:-}" ]; then echo "phone alerts: configured"; else echo "phone alerts: not configured"; fi
if tmux has-session -t webapp 2>/dev/null; then
  echo "demo app already running (reusing it; to pick up $ALERTS_ENV: tmux kill-session -t webapp, then re-run)"
else
  # the session sources the env file itself: a running tmux server would not pass our environment on
  tmux new-session -d -s webapp "cd $PWD && . .venv/bin/activate && { [ ! -f $ALERTS_ENV ] || . $ALERTS_ENV; } && \
    python -m sentinel.webapp --port 8095 --vlm-base-url unix://$SOCK --after-weights $DETECTOR > webapp.log 2>&1"
fi

say "4/5 Incident map (port 8100): tower frames through $DETECTOR, own state dir"
if listening 8100; then
  echo "incident map already running on 8100 (reusing it)"
else
  tmux has-session -t incidentmap-core 2>/dev/null || \
    tmux new-session -d -s incidentmap-core "cd $PWD && . .venv/bin/activate && python -m sentinel.incident_map \
      --port 8100 --state-dir $HOME/sentinel-assets/state-core --vlm-base-url unix://$SOCK \
      --detector-weights $PWD/$DETECTOR > incidentmap-core.log 2>&1"
  for _ in $(seq 1 20); do listening 8100 && break; sleep 3; done
fi

say "5/5 Health check"
for _ in $(seq 1 30); do curl -s -m 5 localhost:8095/api/config >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 5 localhost:8095/api/config | python3 -c "
import json, sys
d = json.load(sys.stdin)
for p in d['pipelines']:
    if not p.get('optional'):
        print(f\"  {p['name']:6} detector={p['detector_weights']:28} vlm={p['vlm']:8} {'OK' if p['vlm_served'] and p['detector_available'] else 'NOT READY'}\")
print('  live camera phone alerts (running app):', 'configured' if (d.get('phone') or {}).get('configured') else 'not configured')"
curl -s -m 5 -o /dev/null -w "  metrics dashboard: HTTP %{http_code}\n" localhost:8090/
curl -s -m 5 -o /dev/null -w "  incident map: HTTP %{http_code}\n" localhost:8100/ || true
# a reused instance keeps the flags it was started with: show which tower detector it actually runs
curl -s -m 5 localhost:8100/api/config 2>/dev/null | python3 -c "import json, sys; \
print('  incident map detector:', json.load(sys.stdin).get('detector_weights'))" 2>/dev/null || true
echo
echo "Ready. On the laptop: ssh -N -L 8095:localhost:8095 -L 8090:localhost:8090 -L 8100:localhost:8100 hp11@<nano-ip>"
echo "Demo app: http://localhost:8095   Live metrics: http://localhost:8090   Incident map: http://localhost:8100"
echo "Live camera: Demo app → Live camera tab (the camera only works on http://localhost, i.e. through the tunnel)"
