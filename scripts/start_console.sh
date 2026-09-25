#!/usr/bin/env bash
# Bring the Sentinel Console up on the Nano (idempotent): the VLM (base7b + LoRA adapters) and ONE server on
# port 8080 that holds everything: incidents, cameras, alerts, edge vs cloud, models, system.
# Needs no internet: models, map tiles and the UI are on the device; ALERTs queue until the uplink returns.
# Usage on the Nano:  cd ~/sentinel && ./scripts/start_console.sh [--lan]
#   --lan   also serve the local network (http://<nano-ip>:8080), e.g. on a tower LAN with no internet.
#           The mobile camera still needs http://localhost (browsers only allow cameras on a secure origin).
# Then on the laptop: ssh -N -L 8080:localhost:8080 hp11@<nano-ip>   → http://localhost:8080
set -euo pipefail
cd "$(dirname "$0")/.."
SOCK=/opt/hp/zrt/run/vllm-base7b.sock
DETECTOR=models/joint_yolo.pt              # v1.3.0-joint (tower 0.718 / D-Fire 0.748); runs at 960 px
PHOTO_DETECTOR=models/smoke_yolo.pt        # D-Fire model for close-up field photos
ALERTS_ENV="${ALERTS_ENV:-$HOME/.sentinel_alerts.env}"   # NTFY_TOPIC_URL=... (chmod 600, never printed);
                                           # ALERTS_ENV=/dev/null for a rehearsal that buzzes no phone
STATE="${STATE:-$HOME/sentinel-state/console}"   # incidents, frames, alert outbox (our own folder)
CLOUD_ENV="$HOME/.cloud_vlm.env"           # optional CLOUD_VLM_* for measured cloud samples
RESULTS="${RESULTS:-results}"              # evaluation results shown on the Models and Edge vs Cloud pages
HOST=127.0.0.1
[ "${1:-}" = "--lan" ] && HOST=0.0.0.0

say() { printf '\n== %s\n' "$*"; }
served() { curl -s -m 5 --unix-socket "$SOCK" http://localhost/v1/models 2>/dev/null; }

say "1/3 VLM: base7b + LoRA adapters"
LORA="context=$PWD/adapters/context"
[ -d adapters/context_v2 ] && LORA="$LORA context_v2=$PWD/adapters/context_v2"
if served | grep -q '"context"'; then
  echo "already serving: $(served | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
else
  # shellcheck disable=SC2086
  zrt serve hf:Qwen/Qwen2.5-VL-7B-Instruct --label base7b --gpu-memory-fraction 0.45 -- \
    --max-model-len=8192 --enable-prefix-caching --enable-lora --max-lora-rank=16 --lora-modules $LORA
  for _ in $(seq 1 60); do served | grep -q '"context"' && break; sleep 10; done
fi
served | grep -q '"context"' || { echo "VLM not ready"; exit 1; }

say "2/3 Sentinel Console (port 8080, host $HOST)"
if tmux has-session -t console 2>/dev/null; then
  echo "console already running (to restart: tmux kill-session -t console, then re-run)"
else
  # the session reads the env files itself: a running tmux server would not pass our environment on
  tmux new-session -d -s console "cd $PWD && . .venv/bin/activate && \
    { [ ! -f $ALERTS_ENV ] || { set -a; . $ALERTS_ENV; set +a; }; } && \
    python -m sentinel.console --host $HOST --port 8080 --vlm-base-url unix://$SOCK \
      --detector-weights $PWD/$DETECTOR --photo-detector-weights $PWD/$PHOTO_DETECTOR --results $RESULTS --state-dir $STATE > console.log 2>&1"
fi
[ -f "$CLOUD_ENV" ] && echo "cloud samples: settings found (off unless CLOUD_VLM_* are complete)"

say "3/3 Health check"
for _ in $(seq 1 40); do curl -s -m 3 localhost:8080/api/overview >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 5 localhost:8080/api/system | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('  node:', d['node'])
for h in d['health']:
    print(f\"  {h['label']:14} {h['state']:5} {h['note']}\")" || { echo "console not answering; see console.log"; exit 1; }
echo
echo "Ready. Laptop: ssh -N -L 8080:localhost:8080 hp11@<nano-ip>   then open http://localhost:8080"
[ "$HOST" = 0.0.0.0 ] && echo "Local network: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8080 (mobile camera needs localhost)"
