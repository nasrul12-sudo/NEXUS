"""Lightweight latency & FPS metrics dengan rolling window."""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from nexus.core.config import get_config


@dataclass
class MetricWindow:
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, value: float) -> None:
        with self.lock:
            self.values.append(value)

    def stats(self) -> dict:
        with self.lock:
            if not self.values:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
            sorted_vals = sorted(self.values)
            n = len(sorted_vals)
            return {
                "count": n,
                "min": round(sorted_vals[0], 3),
                "max": round(sorted_vals[-1], 3),
                "avg": round(sum(sorted_vals) / n, 3),
                "p50": round(sorted_vals[int(n * 0.5)], 3),
                "p95": round(sorted_vals[min(int(n * 0.95), n - 1)], 3),
                "p99": round(sorted_vals[min(int(n * 0.99), n - 1)], 3),
            }


class MetricsRegistry:
    def __init__(self, window_size: int = 100):
        self._windows: dict[str, MetricWindow] = {}
        self._lock = threading.Lock()
        self._window_size = window_size

    def _get_window(self, name: str) -> MetricWindow:
        with self._lock:
            if name not in self._windows:
                w = MetricWindow(name=name)
                w.values = deque(maxlen=self._window_size)
                self._windows[name] = w
            return self._windows[name]

    def record(self, name: str, value: float) -> None:
        self._get_window(name).record(value)

    def stats(self, name: str) -> dict:
        return self._get_window(name).stats()

    def all_stats(self) -> dict[str, dict]:
        with self._lock:
            names = list(self._windows.keys())
        return {name: self.stats(name) for name in names}

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Context manager untuk measure latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(name, elapsed_ms)


# ---------- Singleton ----------

_registry: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    global _registry
    if _registry is None:
        cfg = get_config()
        _registry = MetricsRegistry(window_size=cfg.metrics.window_size)
    return _registry


def reset_metrics() -> None:
    global _registry
    _registry = None