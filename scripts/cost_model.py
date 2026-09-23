"""Per-day cost/feasibility: cloud-every-frame vs local-every-frame vs our cascade.
Measured inputs come from results/bench_*.json; prices are filled from current published rates."""
import argparse
import json

DAY_S = 86_400


def compare(p: dict) -> list[dict]:
    frames = p["towers"] * p["fps"] * DAY_S
    cloud_tokens = frames * p["tokens_per_full_frame"]
    cloud_gb = frames * p["bytes_per_frame"] / 1e9
    calls = p["events_per_day"] * p["vlm_calls_per_event"]
    cascade_gpu = frames * p["detector_s"] + calls * p["local_vlm_s"]
    cascade_gb = p["alerts_per_day"] * p["bytes_per_alert"] / 1e9
    return [
        {"approach": "cloud VLM every frame", "vlm_calls": frames, "cloud_tokens": cloud_tokens,
         "upstream_gb": cloud_gb, "gpu_seconds": 0, "feasible": True,
         "usd_per_day": round(cloud_tokens / 1e6 * p["usd_per_mtok"] + cloud_gb * p["usd_per_gb"], 4)},
        {"approach": "local VLM every frame", "vlm_calls": frames, "cloud_tokens": 0,
         "upstream_gb": 0.0, "gpu_seconds": frames * p["local_vlm_s"],
         "feasible": frames * p["local_vlm_s"] < DAY_S, "usd_per_day": 0.0},
        {"approach": "cascade (ours)", "vlm_calls": calls, "cloud_tokens": 0,
         "upstream_gb": cascade_gb, "gpu_seconds": cascade_gpu, "feasible": cascade_gpu < DAY_S,
         "usd_per_day": round(cascade_gb * p["usd_per_gb"], 4)},
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="?", default="config/cost_inputs.json", help="cost inputs JSON")
    with open(ap.parse_args().inputs) as f:
        params = json.load(f)
    for row in compare(params):
        print(json.dumps(row))
