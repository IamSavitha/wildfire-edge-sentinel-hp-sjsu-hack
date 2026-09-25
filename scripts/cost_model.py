"""Per-day cost/feasibility: cloud-every-frame vs local-every-frame vs our cascade.
Measured inputs come from results/bench_*.json; prices are filled from current published rates."""
import argparse
import json

from sentinel.costmodel import DAY_S, compare  # noqa: F401 - re-exported; the console uses it too


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="?", default="config/cost_inputs.json", help="cost inputs JSON")
    with open(ap.parse_args().inputs) as f:
        params = json.load(f)
    for row in compare(params):
        print(json.dumps(row))
