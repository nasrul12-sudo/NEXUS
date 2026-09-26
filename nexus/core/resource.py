"""Resource Manager — GPU lease & VRAM budgeting.

Karena VRAM 4GB terbatas, kita perlu memastikan hanya satu model
GPU-heavy aktif pada satu waktu. ResourceManager menyediakan
context manager `gpu_lease()` yang:
  1. Acquire lock (via state store)
  2. Cek VRAM tersedia
  3. Evict lease prioritas rendah jika perlu
  4. Yield (user code berjalan)
  5. Release lease saat exit

Jika GPU disabled atau VRAM tidak cukup, lease tetap diberikan
tapi ditandai `cpu_fallback=True` (user code harus handle).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from nexus.core.config import get_config
from nexus.core.errors import ResourceError, ResourceTimeout
from nexus.core.logging import get_logger


log = get_logger(__name__)


@dataclass
class Lease:
    name: str
    vram_mb: int
    priority: int
    acquired_at: float
    last_used_at: float
    cpu_fallback: bool = False


@dataclass
class ResourceState:
    leases: dict[str, Lease] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


class ResourceManager:
    """GPU lease manager. Thread-safe, single-process."""

    def __init__(
        self,
        total_vram_mb: int,
        reserve_mb: int,
        priority: dict[str, int],
        eviction_idle_seconds: int = 5,
        lease_timeout_seconds: int = 30,
        gpu_enabled: bool = False,
    ):
        self.total_vram_mb = total_vram_mb
        self.reserve_mb = reserve_mb
        self.priority = priority
        self.eviction_idle_seconds = eviction_idle_seconds
        self.lease_timeout_seconds = lease_timeout_seconds
        self.gpu_enabled = gpu_enabled
        self._state = ResourceState()

    @property
    def available_vram_mb(self) -> int:
        with self._state.lock:
            used = sum(l.vram_mb for l in self._state.leases.values() if not l.cpu_fallback)
            return max(0, self.total_vram_mb - self.reserve_mb - used)

    def _priority_of(self, name: str) -> int:
        # Lower number = higher priority
        return self.priority.get(name, 99)

    def _evict_idle(self) -> int:
        """Evict leases yang idle. Returns freed VRAM (MB)."""
        freed = 0
        now = time.time()
        with self._state.lock:
            to_evict = [
                name for name, lease in self._state.leases.items()
                if (now - lease.last_used_at) > self.eviction_idle_seconds
                and not lease.cpu_fallback
            ]
            for name in to_evict:
                lease = self._state.leases.pop(name)
                freed += lease.vram_mb
                log.info(
                    "resource.evict_idle",
                    name=name,
                    vram_mb=lease.vram_mb,
                )
        return freed

    def _evict_for(self, needed_mb: int, requester_priority: int) -> int:
        """Evict lower-priority leases until needed_mb is available.

        Returns freed VRAM (MB).
        """
        freed = 0
        with self._state.lock:
            # Sort by priority (higher number = lower priority), then by last_used
            candidates = sorted(
                self._state.leases.items(),
                key=lambda kv: (-kv[1].priority, kv[1].last_used_at),
            )
            for name, lease in candidates:
                if lease.priority <= requester_priority:
                    # Can't evict same or higher priority
                    continue
                if (self.total_vram_mb - self.reserve_mb
                        - sum(l.vram_mb for l in self._state.leases.values() if not l.cpu_fallback)
                        + freed) >= needed_mb:
                    break
                self._state.leases.pop(name)
                freed += lease.vram_mb
                log.info(
                    "resource.evict_priority",
                    name=name,
                    vram_mb=lease.vram_mb,
                    priority=lease.priority,
                )
        return freed

    @contextmanager
    def gpu_lease(
        self,
        name: str,
        vram_mb: int,
        timeout_seconds: float | None = None,
        allow_cpu_fallback: bool = True,
    ) -> Iterator[Lease]:
        """Acquire a GPU lease.

        Args:
            name: lease identifier (e.g., "whisper", "ollama")
            vram_mb: perkiraan VRAM yang dibutuhkan
            timeout_seconds: max wait time. None = config default
            allow_cpu_fallback: jika True dan VRAM tidak cukup,
                                lease diberikan dengan cpu_fallback=True

        Raises:
            ResourceTimeout: jika timeout dan cpu_fallback=False
        """
        timeout = timeout_seconds or self.lease_timeout_seconds
        priority = self._priority_of(name)
        deadline = time.time() + timeout
        cpu_fallback = False

        if not self.gpu_enabled:
            # GPU disabled → always CPU fallback
            lease = Lease(
                name=name, vram_mb=0, priority=priority,
                acquired_at=time.time(), last_used_at=time.time(),
                cpu_fallback=True,
            )
            with self._state.lock:
                self._state.leases[name] = lease
            log.debug("resource.lease.cpu_only", name=name)
            try:
                yield lease
            finally:
                self._release(name)
            return

        # Try to acquire
        while True:
            with self._state.lock:
                # Check existing lease for same name
                existing = self._state.leases.get(name)
                if existing is not None:
                    existing.last_used_at = time.time()
                    yield existing
                    return

                available = self.available_vram_mb
                if available >= vram_mb:
                    lease = Lease(
                        name=name, vram_mb=vram_mb, priority=priority,
                        acquired_at=time.time(), last_used_at=time.time(),
                    )
                    self._state.leases[name] = lease
                    log.info(
                        "resource.lease.acquired",
                        name=name, vram_mb=vram_mb,
                        available_mb=available,
                    )
                    try:
                        yield lease
                    finally:
                        self._release(name)
                    return

            # Not enough VRAM — try eviction
            self._evict_idle()
            with self._state.lock:
                if self.available_vram_mb >= vram_mb:
                    continue
            self._evict_for(vram_mb, priority)

            with self._state.lock:
                if self.available_vram_mb >= vram_mb:
                    continue

            # Still not enough
            if time.time() > deadline:
                if allow_cpu_fallback:
                    cpu_fallback = True
                    lease = Lease(
                        name=name, vram_mb=0, priority=priority,
                        acquired_at=time.time(), last_used_at=time.time(),
                        cpu_fallback=True,
                    )
                    self._state.leases[name] = lease
                    log.warning(
                        "resource.lease.cpu_fallback",
                        name=name, requested_mb=vram_mb,
                    )
                    try:
                        yield lease
                    finally:
                        self._release(name)
                    return
                raise ResourceTimeout(
                    f"Could not acquire {vram_mb}MB for '{name}' within {timeout}s",
                    name=name, vram_mb=vram_mb,
                )

            time.sleep(0.05)

    def _release(self, name: str) -> None:
        with self._state.lock:
            lease = self._state.leases.pop(name, None)
        if lease:
            log.info("resource.lease.released", name=name, vram_mb=lease.vram_mb)

    def status(self) -> dict:
        with self._state.lock:
            return {
                "gpu_enabled": self.gpu_enabled,
                "total_vram_mb": self.total_vram_mb,
                "reserve_mb": self.reserve_mb,
                "available_vram_mb": self.available_vram_mb,
                "leases": [
                    {
                        "name": l.name,
                        "vram_mb": l.vram_mb,
                        "priority": l.priority,
                        "cpu_fallback": l.cpu_fallback,
                        "idle_seconds": time.time() - l.last_used_at,
                    }
                    for l in self._state.leases.values()
                ],
            }


# ---------- Singleton ----------

_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    global _manager
    if _manager is None:
        cfg = get_config()
        _manager = ResourceManager(
            total_vram_mb=cfg.resource.gpu.total_vram_mb,
            reserve_mb=cfg.resource.gpu.reserve_mb,
            priority=cfg.resource.priority,
            eviction_idle_seconds=cfg.resource.gpu.eviction_idle_seconds,
            lease_timeout_seconds=cfg.resource.gpu.lease_timeout_seconds,
            gpu_enabled=cfg.resource.gpu.enabled,
        )
    return _manager


def reset_resource_manager() -> None:
    global _manager
    _manager = None