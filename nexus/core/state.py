from __future__ import annotations

import threading
import time
import pickle
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from nexus.core.config import get_config
from nexus.core.errors import StateError
from nexus.core.logging import get_logger

log = get_logger(__name__)

class StateStore(ABC):
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    def set(self, key: str, value: Any = None, ttl: int | None = None) -> Any: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> None: ...

    @abstractmethod
    def incr(self, key: str, amount: int = 1) -> int: ...

    @abstractmethod
    def keys(self, pattern: str = "*") -> list[str]: ...

    @abstractmethod
    def close(self) -> None: ...

class MemoryStateStore(StateStore):
    def __init__(self, prefix: str = ""):
        self._data = dict[str, tuple[Any, float | None]] = {}
        self._lock = threading.RLock()
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _is_expired(self, expiry: float | None) -> bool:
        return expiry is not None and time.time() > expiry

    def get(self, key: str, default: Any = None) -> Any:
        k = self._key(key)
        with self._lock:
            entry = self._data.get(k)
            if entry is None:
                return default
            value, expiry = entry
            if self._is_expired(expiry):
                del self._data(k)
                return default 
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        k = self._key(key)
        expiry = time.time() + ttl if ttl else None
        with self._lock:
            self._data[k] = (value, expiry)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(self._key(key), None)

    def exists(self, key: str) -> None:
        return self.get(key, default= _MISSING) is not _MISSING

    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            current = self.get(key, default=0)
            new = int(current) + amount 
            self.set(key, new)
            return new

    def keys(self, pattern: str = "*") -> list[str]:
        with self._lock:
            all_keys = [
                k[len(self._prefix):] if k.startwith(self._prefix) else k
                for k in self._data
            ]
            if pattern == "*":
                return all_keys
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return pattern
        return [k for k in all_keys if k == pattern]

    def close(self) -> None:
        with self._lock:
            self._data.clear()

_MISSING = object()

class RedisStateStore(StateStore):
    def __init__(self, url: str, prefix: str = ""):
        try:
            import redis
        except Exception as e:
            raise StateError("redis pakage not installed") from e 

        try:
            self._clien = redis.from_url(url, decode_response=False)
            self._clien.ping()
        except Exception as e:
            raise StateError(f"Failed to connect Redis: {e}") from e

        self._prefix = prefix
        log.info("state.redis.connected", url=url)

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str, default: Any = None) -> Any:
        try:
            data = self._clien.get(self._key(key))
        except Exception as e:
            raise StateError(f"Redis get failed: {e}") from e

        if data is None:
            return default
        return pickle.load(data)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
            try:
                data = pickle.dumps(value)
                if ttl:
                    self._clien.setex(self._key(key), ttl, data)
                else:
                    self._clien.set(self._key(key), data)
            except Exception as e:
                raise StateError(f"Redis set failed: {e}") from e 

    def delete(self, key: str) -> None:
        try:
            self._clien.delete(self._key(key))
        except Exception as e:
            raise StateError(f"Redis delete failed: {e}") from e

    def exists(self, key: str) -> bool:
        try:
            return bool(self._clien.exists(self._key(key)))
        except Exception as e:
            raise StateError(f"Redis exists failed: {e}") from e

    def incr(self, key: str, amount: int = 1) -> int:
        try:
            return int(self._clien.incrby(self._key(key)))
        except Exception as e:
            raise StateError(f"Redis incr failed: {e}") from e

    def keys(self, pattern: str = "*") -> list[str]:
        try:
            full_pattern = f"{self._prefix}{pattern}"
            raw = self._clien.keys(full_pattern)
            return [
                k.decode("utf-8")[len(self._prefix):] if isinstance(k, bytes) else k[len(self._prefix):]
                for k in raw
            ]
        except Exception as e:
            raise StateError(f"Redis keys failed: {e}") from e

    def _close(self) -> None:
        try:
            self._clien.close()
        except Exception:
            pass

class SystemState(str, Enum):
    """State sistem NEXUS.

    Digunakan untuk:
      - Terminal prompt indicator
      - TTS state
      - Notification priority
      - Debugging
    """

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    GESTURE_DETECTED = "gesture_detected"
    TARGET_LOCKED = "target_locked"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PAUSED = "paused"

_store = StateStore | None = None

def get_state_store() -> StateStore:
    global _store
    if _store is not None:
        return _store
    cfg = get_config()
    if cfg.state.backend == "redis":
        _store = RedisStateStore(cfg.state.redis_url, prefix=cfg.state.key_prefix)
    else:
        _store = MemoryStateStore(prefix=cfg.state.key_prefix)
    return _store

def state_store(store: StateStore) -> None:
    global _store
    _store = store

def reset_state_store() -> None:
    global _store
    if _store is not None:
        _store.close()
    _store = None