"""Score a context VLM against a labeled JSONL split and save results/context_<name>.json.

Rows are {"image": <crop path>, "label": <ContextResult dict>} (scripts/teacher_label.py output,
or a hand-checked gold set in the same format).

BEFORE: --model "<base id from /v1/models>" --name before_base7b
AFTER:  --model context --name after_lora7b      (vLLM serving the LoRA adapter as "context")
CLOUD:  --cloud [--response-format json_object] [--limit 20]   (name defaults to cloud_qwen7b)

--cloud sends the same crops, prompt and model (Qwen2.5-VL-7B-Instruct) to a hosted
OpenAI-compatible provider configured ONLY by environment variables: CLOUD_VLM_BASE_URL,
CLOUD_VLM_MODEL, CLOUD_VLM_API_KEY (and optional CLOUD_VLM_RESPONSE_FORMAT). Every uncached crop
is a paid API call; answers are cached in data/cloud_cache.jsonl so re-runs are not re-billed.
Cloud latency is the real wall time of the HTTPS request from this machine (network included),
over fresh calls only; cache hits are counted separately.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from sentinel.cloud_cache import CachedVLM, ResponseCache
from sentinel.vlm_client import RESPONSE_FORMATS, ContextVLM, cloud_vlm_from_env, ensure_served

GROUP = {"wildland": "danger", "structure": "danger", "vehicle": "danger",
         "controlled_burn": "benign", "campfire": "benign", "bbq_chimney": "benign",
         "industrial_stack": "benign", "fog_dust_cloud": "lookalike", "unknown": "unknown"}


def check_output(path: Path, force: bool) -> None:
    """Exit 1 instead of silently overwriting an earlier result (e.g. a BEFORE row)."""
    if path.exists() and not force:
        print(f"{path} already exists; pass --force to overwrite it",
              file=sys.stderr)
        raise SystemExit(1)


def _pct(xs, q) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def evaluate(rows, classify, read_bytes=lambda p: Path(p).read_bytes(), clock=time.perf_counter,
             source=None) -> dict:
    """Run classify(jpeg) -> (ContextResult | None, tokens) over rows and score it against row labels.

    A parse failure (None) counts as wrong for both accuracies. `source` is the CachedVLM behind
    `classify` for a cloud run: latency is then taken over fresh calls only, cache hits and request
    errors are counted separately, and billed tokens come from provider-reported usage.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("empty split")
    exact = group = fails = 0
    latencies, tokens = [], []
    per_source: dict[str, dict[str, int]] = {}
    cloud = {"fresh_ms": [], "cached_ms": [], "errors": 0, "usage_missing": 0,
             "billed_in": 0, "billed_out": 0, "in": 0, "out": 0, "answered": 0}
    for r in rows:
        jpeg = read_bytes(r["image"])
        t0 = clock()
        ctx, tok = classify(jpeg)
        ms = (clock() - t0) * 1000
        latencies.append(ms)
        tokens.append(tok)
        if source is not None:
            _account(cloud, source, ms)
        want = r["label"]["source_type"]
        bucket = per_source.setdefault(want, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if ctx is None:
            fails += 1
            continue
        hit = ctx.source_type == want
        exact += hit
        bucket["correct"] += hit
        group += GROUP[ctx.source_type] == GROUP[want]
    n = len(rows)
    out = {
        "n": n,
        "source_type_acc": exact / n,
        "group_acc": group / n,
        "parse_fail_rate": fails / n,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "tokens_per_call": float(np.mean(tokens)),
        "per_source": per_source,
    }
    if source is not None:
        out.update(_cloud_summary(cloud, source, n))
    return out


def _account(c: dict, source, ms: float) -> None:
    usage = source.last_usage or {}
    if source.last_cached:
        if source.last_latency_ms is not None:
            c["cached_ms"].append(float(source.last_latency_ms))
    elif source.last_error:
        c["errors"] += 1
        return
    else:
        c["fresh_ms"].append(ms)
        c["billed_in"] += int(usage.get("prompt_tokens") or 0)
        c["billed_out"] += int(usage.get("completion_tokens") or 0)
        c["usage_missing"] += int(usage.get("usage_missing") or 0)
    c["answered"] += 1
    c["in"] += int(usage.get("prompt_tokens") or 0)
    c["out"] += int(usage.get("completion_tokens") or 0)


def _cloud_summary(c: dict, source, n: int) -> dict:
    answered = c["answered"]
    return {
        "n_fresh": source.n_fresh, "n_cached": source.n_cached, "n_request_errors": c["errors"],
        "request_error_rate": c["errors"] / n,
        # fresh calls only: cache hits are free and have no new latency
        "latency_ms_p50": _pct(c["fresh_ms"], 50), "latency_ms_p95": _pct(c["fresh_ms"], 95),
        # latency measured when the cached answers were originally fetched
        "cached_latency_ms_p50": _pct(c["cached_ms"], 50), "cached_latency_ms_p95": _pct(c["cached_ms"], 95),
        "billed_prompt_tokens": c["billed_in"], "billed_completion_tokens": c["billed_out"],
        "prompt_tokens_total": c["in"], "completion_tokens_total": c["out"],
        "prompt_tokens_per_call": c["in"] / answered if answered else None,
        "completion_tokens_per_call": c["out"] / answered if answered else None,
        "usage_missing_calls": c["usage_missing"],
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help='served model id, or "context" for the LoRA adapter (not with --cloud)')
    ap.add_argument("--name", help="result tag, e.g. before_base7b / after_lora7b (--cloud: cloud_qwen7b)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="data/teacher/heldout.jsonl")
    ap.add_argument("--timeout", type=float, default=60, help="per-request timeout, seconds")
    ap.add_argument("--cloud", action="store_true",
                    help="use the hosted provider from CLOUD_VLM_* env vars (cached, see above)")
    ap.add_argument("--response-format", choices=RESPONSE_FORMATS,
                    help="json_schema (default) | json_object | none; overrides CLOUD_VLM_RESPONSE_FORMAT")
    ap.add_argument("--limit", type=int, help="score only the first N rows (deterministic subsample)")
    ap.add_argument("--cache", default="data/cloud_cache.jsonl", help="--cloud response cache (JSONL)")
    ap.add_argument("--max-fresh-calls", type=int,
                    help="--cloud: stop before more than N billed calls (cached answers are free)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    a = ap.parse_args(argv)
    if a.cloud:
        a.name = a.name or "cloud_qwen7b"
    elif not (a.model and a.name):
        ap.error("--model and --name are required unless --cloud is given")
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be >= 1")
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    out = Path("results") / f"context_{a.name}.json"
    check_output(out, a.force)
    with open(a.split) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if a.limit is not None:
        rows = rows[:a.limit]

    if a.cloud:
        vlm = cloud_vlm_from_env(timeout_s=a.timeout, response_format=a.response_format)
        cached = CachedVLM(vlm, ResponseCache(a.cache), max_fresh_calls=a.max_fresh_calls)
        print(f"cloud: model {vlm.model} at {vlm.host} ({vlm.response_format}); {len(rows)} crops, "
              f"{len(cached.cache)} cached answers on file", file=sys.stderr)
        result = {"name": a.name, "model": vlm.model, "split": a.split, "cloud": True,
                  "provider_host": vlm.host, "response_format": vlm.response_format, "limit": a.limit,
                  **evaluate(rows, cached.classify, source=cached)}
    else:
        vlm = ContextVLM(a.model, a.base_url, timeout_s=a.timeout,
                         response_format=a.response_format or "json_schema")
        ensure_served(vlm.client, a.model, a.base_url)
        result = {"name": a.name, "model": a.model, "split": a.split,
                  **({"limit": a.limit} if a.limit is not None else {}), **evaluate(rows, vlm.classify)}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
