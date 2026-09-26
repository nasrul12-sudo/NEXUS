import time
from nexus.core.resource import ResourceManager


def test_lease_basic():
    rm = ResourceManager(
        total_vram_mb=4096, reserve_mb=512,
        priority={"vision": 1, "voice": 2}, gpu_enabled=True,
    )
    with rm.gpu_lease("whisper", vram_mb=1000) as lease:
        assert lease.name == "whisper"
        assert not lease.cpu_fallback
        assert rm.available_vram_mb == 4096 - 512 - 1000
    assert rm.available_vram_mb == 4096 - 512


def test_cpu_fallback_when_disabled():
    rm = ResourceManager(
        total_vram_mb=4096, reserve_mb=512,
        priority={"vision": 1}, gpu_enabled=False,
    )
    with rm.gpu_lease("whisper", vram_mb=1000) as lease:
        assert lease.cpu_fallback


def test_priority_eviction():
    rm = ResourceManager(
        total_vram_mb=2000, reserve_mb=0,
        priority={"vision": 1, "voice": 2}, gpu_enabled=True,
        eviction_idle_seconds=0,  # immediate eviction
    )
    with rm.gpu_lease("voice", vram_mb=1500):
        pass  # released
    # Now acquire high-priority vision
    with rm.gpu_lease("vision", vram_mb=1500) as lease:
        assert not lease.cpu_fallback