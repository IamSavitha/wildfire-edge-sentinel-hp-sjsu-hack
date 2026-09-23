#!/usr/bin/env bash
# One-shot setup on a fresh ZGX Nano (the node is wiped after the event).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
  echo "Install CUDA PyTorch for GB10 per the NVIDIA DGX Spark playbook first."; exit 1; }
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt -r requirements-train.txt
pip install -e .
# Warm YOLO-World while online: set_classes() loads the CLIP text encoder and downloads its
# weights on first use. Caching them now lets the BEFORE (zero-shot) detector run offline in demos.
python -c "from ultralytics import YOLO; YOLO('yolov8s-worldv2.pt').set_classes(['smoke','fire'])"
command -v zrt >/dev/null || sudo snap install --classic zrt
zrt pull Qwen/Qwen2.5-VL-7B-Instruct
echo "Next: ./scripts/download_data.sh, then see README 'Reproduce'."
