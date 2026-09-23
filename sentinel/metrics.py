"""Counters and latency samples for benchmarks and the dashboard."""
from collections import Counter, defaultdict

import numpy as np


class Metrics:
    def __init__(self):
        self.counters: Counter = Counter()
        self.timings: dict[str, list[float]] = defaultdict(list)

    def inc(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def time(self, key: str, value: float) -> None:
        self.timings[key].append(value)

    def merge(self, other: "Metrics") -> None:
        self.counters.update(other.counters)
        for k, v in other.timings.items():
            self.timings[k].extend(v)

    def summary(self) -> dict:
        out: dict = dict(self.counters)
        for k, v in self.timings.items():
            out[f"{k}_p50"] = float(np.percentile(v, 50))
            out[f"{k}_p95"] = float(np.percentile(v, 95))
        return out
