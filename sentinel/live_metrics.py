"""Pure helpers for the live model-economics dashboard: Prometheus parsing, vLLM snapshots,
rates, edge-vs-cloud economics, zrt model discovery and host stats. No network I/O here."""
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Callable

Sample = tuple[str, dict, float]
Buckets = list[tuple[float, float]]

log = logging.getLogger(__name__)

SMI_CMD = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]

_ESCAPES = {"n": "\n", "\\": "\\", '"': '"'}


def _parse_labels(s: str, i: int) -> tuple[dict, int]:
    """Parse `{k="v",...}` starting just after `{`; return (labels, index after `}`)."""
    labels: dict = {}
    n = len(s)
    while True:
        while i < n and s[i] in " ,":
            i += 1
        if i >= n:
            raise ValueError("unterminated labels")
        if s[i] == "}":
            return labels, i + 1
        eq = s.index("=", i)
        key = s[i:eq].strip()
        i = eq + 1
        if i >= n or s[i] != '"':
            raise ValueError("label value must be quoted")
        i += 1
        out = []
        while True:
            if i >= n:
                raise ValueError("unterminated label value")
            c = s[i]
            if c == "\\" and i + 1 < n:
                out.append(_ESCAPES.get(s[i + 1], s[i + 1]))
                i += 2
            elif c == '"':
                i += 1
                break
            else:
                out.append(c)
                i += 1
        labels[key] = "".join(out)


def parse_prometheus(text: str) -> list[Sample]:
    """Prometheus text exposition -> [(name, labels, value)]. Skips comments, blanks, malformed
    lines and NaN values; an optional trailing timestamp is ignored."""
    samples: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            brace = line.find("{")
            space = line.find(" ")
            if brace != -1 and (space == -1 or brace < space):
                name = line[:brace]
                labels, i = _parse_labels(line, brace + 1)
                rest = line[i:].split()
            else:
                name, *rest = line.split()
                labels = {}
            value = float(rest[0])
        except (ValueError, IndexError):
            continue
        if name and not math.isnan(value):
            samples.append((name, labels, value))
    return samples


def histogram_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus-style quantile from cumulative (le, count) buckets with linear interpolation.
    A rank in the +Inf bucket returns the highest finite bound. None when empty or q not in [0,1]."""
    if not buckets or not 0 <= q <= 1:
        return None
    b = sorted(buckets, key=lambda x: x[0])
    total = b[-1][1]
    if total <= 0:
        return None
    rank = q * total
    for idx, (le, count) in enumerate(b):
        if count >= rank:
            if math.isinf(le):
                finite = [x for x, _ in b if not math.isinf(x)]
                return finite[-1] if finite else None
            start, prev = (0.0, 0.0) if idx == 0 else b[idx - 1]
            if idx == 0 and le <= 0:
                return le
            in_bucket = count - prev
            if in_bucket <= 0:
                return le
            return start + (le - start) * (rank - prev) / in_bucket
    return None


def _match(labels: dict, model_name: str | None) -> bool:
    return model_name is None or labels.get("model_name") == model_name


def _sum(samples: list[Sample], names: tuple[str, ...], model_name: str | None) -> float | None:
    """Sum of the first metric name present (tries current, then legacy vLLM names)."""
    for name in names:
        vals = [v for n, lab, v in samples if n == name and _match(lab, model_name)]
        if vals:
            return float(sum(vals))
    return None


def _mean(samples: list[Sample], names: tuple[str, ...], model_name: str | None) -> float | None:
    for name in names:
        vals = [v for n, lab, v in samples if n == name and _match(lab, model_name)]
        if vals:
            return float(sum(vals) / len(vals))
    return None


def _buckets(samples: list[Sample], name: str, model_name: str | None) -> list[tuple[float, float]]:
    acc: dict[float, float] = {}
    for n, lab, v in samples:
        if n == f"{name}_bucket" and _match(lab, model_name) and "le" in lab:
            try:
                le = float(lab["le"])
            except ValueError:
                continue
            acc[le] = acc.get(le, 0.0) + v
    return sorted(acc.items())


def model_snapshot(samples: list[Sample], model_name: str | None) -> dict:
    """Counters, gauges and latency quantiles for one served model (None where absent).
    model_name=None aggregates every sample (one vLLM backend per socket)."""
    kv = _mean(samples, ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"), model_name)
    lat = _buckets(samples, "vllm:e2e_request_latency_seconds", model_name)
    ttft = _buckets(samples, "vllm:time_to_first_token_seconds", model_name)
    return {
        "requests": _sum(samples, ("vllm:e2e_request_latency_seconds_count",
                                   "vllm:request_success_total"), model_name),
        "prompt_tokens": _sum(samples, ("vllm:prompt_tokens_total",), model_name),
        "generation_tokens": _sum(samples, ("vllm:generation_tokens_total",), model_name),
        "running": _sum(samples, ("vllm:num_requests_running",), model_name),
        "kv_cache_pct": None if kv is None else kv * 100.0,
        "latency_p50_s": histogram_quantile(lat, 0.5),
        "latency_p95_s": histogram_quantile(lat, 0.95),
        "ttft_p50_s": histogram_quantile(ttft, 0.5),
    }


HISTOGRAMS = {"latency": "vllm:e2e_request_latency_seconds", "ttft": "vllm:time_to_first_token_seconds"}
QUANTILES = (("latency_p50_s", "latency", 0.5), ("latency_p95_s", "latency", 0.95),
             ("ttft_p50_s", "ttft", 0.5))


def histogram_buckets(samples: list[Sample], model_name: str | None) -> dict[str, Buckets]:
    """Cumulative buckets of the latency and TTFT histograms, for windowed quantiles."""
    return {key: _buckets(samples, name, model_name) for key, name in HISTOGRAMS.items()}


def bucket_delta(prev: Buckets, cur: Buckets) -> Buckets | None:
    """Per-bucket increase since `prev` (a bound missing from prev counts as 0). None when any
    bucket went down, i.e. the server restarted and its histograms reset."""
    before = dict(prev)
    out = []
    for le, count in cur:
        d = count - before.get(le, 0.0)
        if d < 0:
            return None
        out.append((le, d))
    return out


def windowed_quantiles(prev: dict | None, cur: dict, last: dict | None) -> dict:
    """Latency/TTFT quantiles over requests that finished since the previous poll.
    No previous poll -> cumulative since server start ("since_start"). No new requests or a reset
    -> keep the last value ("held"). Otherwise quantiles of the bucket deltas ("window")."""
    last = last or {}
    out: dict = {}
    basis = {}
    # a held value keeps the basis it was computed on (a since-start value stays "since_start")
    held = "since_start" if last.get("latency_window") == "since_start" else "held"
    for key in HISTOGRAMS:
        cur_b = cur.get(key) or []
        if prev is None:
            basis[key], b = "since_start", cur_b
        else:
            d = bucket_delta(prev.get(key) or [], cur_b)
            new = max((c for _, c in d), default=0.0) if d is not None else 0.0
            basis[key], b = ("window", d) if new > 0 else (held, None)
        for out_key, hist, q in QUANTILES:
            if hist == key:
                out[out_key] = histogram_quantile(b, q) if b is not None else last.get(out_key)
    out["latency_window"] = basis["latency"]
    return out


TOTAL_KEYS = ("prompt_tokens", "generation_tokens", "requests")


def accumulate_totals(prev_raw: dict | None, cur_raw: dict, offsets: dict) -> tuple[dict, dict]:
    """Session totals that never decrease: when a counter drops (server restart) the value it had
    reached is added to a per-key offset. Returns (adjusted totals, new offsets)."""
    offsets = dict(offsets)
    adjusted = {}
    for key in TOTAL_KEYS:
        a = prev_raw.get(key) if prev_raw else None
        b = cur_raw.get(key)
        if a is not None and b is not None and b < a:
            offsets[key] = offsets.get(key, 0.0) + a
        adjusted[key] = None if b is None else b + offsets.get(key, 0.0)
    return adjusted, offsets


_RATE_KEYS = (("prompt_tok_per_s", "prompt_tokens"), ("gen_tok_per_s", "generation_tokens"),
              ("req_per_s", "requests"))


def rates(prev: dict | None, cur: dict, dt_s: float) -> dict:
    """Per-second counter rates. None on first sample, non-positive dt, missing values or a
    counter reset (server restart)."""
    out: dict = {}
    for out_key, key in _RATE_KEYS:
        a = prev.get(key) if prev else None
        b = cur.get(key)
        ok = a is not None and b is not None and dt_s > 0 and b >= a
        out[out_key] = (b - a) / dt_s if ok else None
    return out


def _num(d: dict | None, key: str) -> float:
    v = (d or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def effective_prices(prices: dict) -> dict:
    """Resolve in/out token prices (fall back to usd_per_mtok) and the per-frame inputs."""
    single = _num(prices, "usd_per_mtok")
    return {
        "usd_per_mtok_in": _num(prices, "usd_per_mtok_in") if "usd_per_mtok_in" in prices else single,
        "usd_per_mtok_out": _num(prices, "usd_per_mtok_out") if "usd_per_mtok_out" in prices else single,
        "usd_per_gb": _num(prices, "usd_per_gb"),
        "tokens_per_full_frame": _num(prices, "tokens_per_full_frame"),
        "bytes_per_frame": _num(prices, "bytes_per_frame"),
    }


def economics(models: dict[str, dict], pipeline_metrics: dict | None, prices: dict) -> dict:
    """Edge tokens and what the same tokens would cost on a cloud API at the given prices; plus,
    when pipeline counters exist, the cloud-every-frame vs cascade comparison."""
    p = effective_prices(prices)
    per_model: dict = {}
    tot_in = tot_out = 0.0
    for label, snap in models.items():
        t_in, t_out = _num(snap, "prompt_tokens"), _num(snap, "generation_tokens")
        tot_in, tot_out = tot_in + t_in, tot_out + t_out
        per_model[label] = {
            "prompt_tokens": snap.get("prompt_tokens"), "generation_tokens": snap.get("generation_tokens"),
            "edge_tokens": t_in + t_out,
            "cloud_equiv_usd": t_in / 1e6 * p["usd_per_mtok_in"] + t_out / 1e6 * p["usd_per_mtok_out"],
        }
    total = {"prompt_tokens": tot_in, "generation_tokens": tot_out, "edge_tokens": tot_in + tot_out,
             "cloud_equiv_usd": tot_in / 1e6 * p["usd_per_mtok_in"] + tot_out / 1e6 * p["usd_per_mtok_out"]}

    pipeline = None
    if pipeline_metrics is not None:
        frames = _num(pipeline_metrics, "frames")
        calls = _num(pipeline_metrics, "vlm_calls")
        bytes_up = _num(pipeline_metrics, "bytes_up")
        video = frames * p["bytes_per_frame"]
        pipeline = {
            "frames_seen": int(frames), "vlm_calls": int(calls),
            "vlm_tokens": int(_num(pipeline_metrics, "vlm_tokens")),
            "frames_per_vlm_call": frames / calls if calls > 0 else None,
            "cloud_per_frame_tokens": frames * p["tokens_per_full_frame"],
            "cloud_per_frame_usd": frames * p["tokens_per_full_frame"] / 1e6 * p["usd_per_mtok_in"],
            "cascade_calls_avoided": int(max(frames - calls, 0)),
            "bytes_up": int(bytes_up), "video_bytes_equiv": int(video),
            "upstream_saving_ratio": 1 - bytes_up / video if video > 0 else None,
            "video_upload_usd": video / 1e9 * p["usd_per_gb"],
            "edge_uplink_usd": bytes_up / 1e9 * p["usd_per_gb"],
        }
    return {"models": per_model, "total": total, "pipeline": pipeline}


def discover_models(run_dir) -> list[dict]:
    """One entry per zrt-served vLLM backend (`vllm-<label>.json`). A malformed metadata file
    still yields a row (label from the filename) so its socket can be polled."""
    run_dir = Path(run_dir)
    try:
        paths = sorted(run_dir.glob("vllm-*.json"))
    except OSError:
        return []
    out, seen = [], set()
    for path in paths:
        try:
            meta = json.loads(path.read_text())
            if not isinstance(meta, dict):
                meta = {}
        except (OSError, ValueError):
            meta = {}
        file_label = path.stem[len("vllm-"):]
        label = str(meta.get("label") or file_label)
        if label in seen:
            log.warning("duplicate model label %r in %s; keeping the first", label, path)
            continue
        seen.add(label)
        sock = path.with_suffix(".sock")
        out.append({
            "label": label,
            "model_uri": meta.get("model_uri"),
            "gpu_memory_fraction": meta.get("gpu_memory_fraction"),
            "started_at": meta.get("started_at"),
            "sock": str(sock) if sock.exists() else None,
        })
    return out


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=True).stdout


def system_stats(meminfo_path="/proc/meminfo", smi_cmd: list[str] = SMI_CMD,
                 run: Callable[[list[str]], str] = _run) -> dict:
    """GPU utilisation (nvidia-smi) and unified memory (MemTotal - MemAvailable). None if absent."""
    gpu = None
    try:
        gpu = float(run(smi_cmd).strip().splitlines()[0])
        if math.isnan(gpu):
            gpu = None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        gpu = None
    total = used = None
    try:
        kb = {}
        for line in Path(meminfo_path).read_text().splitlines():
            key, _, rest = line.partition(":")
            if rest.split():
                kb[key] = float(rest.split()[0])
        if "MemTotal" in kb and "MemAvailable" in kb:
            total = kb["MemTotal"] * 1024 / 1e9
            used = (kb["MemTotal"] - kb["MemAvailable"]) * 1024 / 1e9
    except (OSError, ValueError):
        pass
    return {"gpu_util_pct": gpu, "mem_used_gb": used, "mem_total_gb": total}
