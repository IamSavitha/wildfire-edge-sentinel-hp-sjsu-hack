# Wildfire Edge Sentinel: the console (incident map, cameras, alerts, edge vs cloud) in one container for the edge
# node, an HP ZGX Nano (arm64, NVIDIA GB10). The context VLM stays in HP zrt (vLLM) on the host; the container
# reaches it through the zrt unix socket, mounted in. Weights, map tiles and recordings are mounted too, so the
# image holds only code and its dependencies. Nothing needs the internet at run time.
#
#   docker compose up -d --build                  # on the Nano: see docker-compose.yml for the mounts
#   docker build -t sentinel-console .            # image only
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t sentinel-console:cpu .
#                                                 # CPU-only build (a laptop without an NVIDIA GPU)
#   docker run --rm sentinel-console python -m pytest -q    # the test suite, inside the image

ARG PYTHON=3.12
FROM python:${PYTHON}-slim-bookworm

# CUDA 13 wheels, the same torch/torchvision the Nano runs; they carry their own CUDA libraries, so the host
# only needs the NVIDIA driver and the NVIDIA container runtime.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}"
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY sentinel ./sentinel
COPY scripts ./scripts
COPY config ./config
COPY results ./results
COPY tests ./tests
RUN pip install --no-deps -e . \
 && mkdir -p /state /assets /app/models /app/data /app/weights /opt/sentinel /opt/hp/zrt/run \
 && chmod -R a+rwX /state /app/results

# Run as the host user that owns the zrt socket (compose sets it); nothing here needs root.
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/api/overview > /dev/null || exit 1

CMD ["python", "-m", "sentinel.console", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--assets", "/assets", "--state-dir", "/state", "--data-dir", "/app/data", \
     "--detector-weights", "/app/models/joint_yolo.pt", \
     "--photo-detector-weights", "/app/models/smoke_yolo.pt", \
     "--before-weights", "/opt/sentinel/yolov8s-worldv2.pt", \
     "--vlm-base-url", "unix:///opt/hp/zrt/run/vllm-base7b.sock", \
     "--run-dir", "/opt/hp/zrt/run", \
     "--results", "/app/results"]
