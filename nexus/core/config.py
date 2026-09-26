"""Configuration loader dengan layered override.

Priority (lowest → highest):
  1. config/default.yaml
  2. config/{NEXUS_ENV}.yaml
  3. Environment variables (prefix NEXUS_, nested delimiter __)
  4. Runtime override (via .override_config())

Menggunakan BaseModel (bukan BaseSettings) untuk menghindari
perilaku pydantic-settings yang tidak terduga saat pass kwargs.
Env override di-handle secara manual di load_config().
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from nexus.core.errors import ConfigError


# ---------- Sub-models ----------

class NexusMeta(BaseModel):
    name: str = "NEXUS"
    mode: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"


class BusConfig(BaseModel):
    publish_endpoint: str = "ipc:///tmp/nexus-events.sock"
    hwm: int = 1000
    recv_timeout_ms: int = 100


class StateConfig(BaseModel):
    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "nexus:"
    default_ttl_seconds: int = 300


class GPUResourceConfig(BaseModel):
    enabled: bool = False
    total_vram_mb: int = 4096
    reserve_mb: int = 512
    eviction_idle_seconds: int = 5
    lease_timeout_seconds: int = 30


class ResourceConfig(BaseModel):
    gpu: GPUResourceConfig = Field(default_factory=GPUResourceConfig)
    priority: dict[str, int] = Field(
        default_factory=lambda: {"vision": 1, "voice": 2, "brain": 3, "action": 4}
    )


class MetricsConfig(BaseModel):
    enabled: bool = True
    window_size: int = 100


class ProcessConfig(BaseModel):
    shutdown_timeout_seconds: int = 5
    restart_on_crash: bool = True
    max_restart_attempts: int = 3


# ---------- Vision ----------

class CameraConfig(BaseModel):
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1


class HandTrackingConfig(BaseModel):
    model_path: str = "data/models/hand_landmarker.task"
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    running_mode: Literal["IMAGE", "VIDEO", "LIVE_STREAM"] = "VIDEO"


class VisionConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    hand_tracking: HandTrackingConfig = Field(default_factory=HandTrackingConfig)


class PoseTrackingConfig(BaseModel):
    enabled: bool = True
    model_path: str = "data/models/pose_landmarker_lite.task"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class VisionConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    hand_tracking: HandTrackingConfig = Field(default_factory=HandTrackingConfig)
    pose_tracking: PoseTrackingConfig = Field(default_factory=PoseTrackingConfig)

# ---------- Gesture ----------

class GestureConfig(BaseModel):
    enabled: bool = True
    engine: Literal["rule_based", "ml"] = "rule_based"
    smoothing_window: int = 4
    min_stable_frames: int = 2
    cooldown_ms: int = 300
    min_confidence: float = 0.5
    pinch_distance_threshold: float = 0.05
    open_palm_extension_threshold: float = 0.85
    fist_extension_threshold: float = 0.3
    point_extension_threshold: float = 0.75
    swipe_min_distance: float = 0.15
    swipe_max_duration_ms: int = 500
    swipe_history_size: int = 10
    thumb_curvature_max: float = 0.3
    thumb_others_folded_min: float = 0.4


# ---------- Output ----------

class TerminalOutputConfig(BaseModel):
    enabled: bool = True
    color: bool = True
    verbosity: Literal["quiet", "normal", "verbose"] = "normal"
    show_gesture: bool = True
    show_voice: bool = True
    show_intent: bool = True
    show_command: bool = True


class TTSOutputConfig(BaseModel):
    enabled: bool = False
    provider: str = "piper"
    voice: str = "en_US-lessac-medium"
    speak_on: list[str] = Field(
        default_factory=lambda: ["voice_response", "error", "confirmation"]
    )


class NotificationOutputConfig(BaseModel):
    enabled: bool = True
    provider: str = "notify-send"
    on: list[str] = Field(default_factory=lambda: ["error", "warning"])


class SoundOutputConfig(BaseModel):
    enabled: bool = False
    on: list[str] = Field(default_factory=lambda: ["gesture_confirmed", "wake_word"])


class OutputConfig(BaseModel):
    terminal: TerminalOutputConfig = Field(default_factory=TerminalOutputConfig)
    tts: TTSOutputConfig = Field(default_factory=TTSOutputConfig)
    notification: NotificationOutputConfig = Field(default_factory=NotificationOutputConfig)
    sound: SoundOutputConfig = Field(default_factory=SoundOutputConfig)


# ---------- Input ----------

class HotkeyInputConfig(BaseModel):
    enabled: bool = False
    bindings: dict[str, str] = Field(
        default_factory=lambda: {
            "trigger_voice": "ctrl+alt+n",
            "screenshot": "ctrl+alt+s",
            "pause": "ctrl+alt+p",
            "resume": "ctrl+alt+r",
            "quit": "ctrl+alt+q",
        }
    )


class InputConfig(BaseModel):
    hotkeys: HotkeyInputConfig = Field(default_factory=HotkeyInputConfig)


# ---------- API ----------

class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


# ---------- Root config ----------

class NexusConfig(BaseModel):
    """Root config.

    Menggunakan BaseModel (bukan BaseSettings) karena kita handle
    layered override secara manual di load_config().
    """

    model_config = {"extra": "ignore"}

    nexus: NexusMeta = Field(default_factory=NexusMeta)
    bus: BusConfig = Field(default_factory=BusConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    resource: ResourceConfig = Field(default_factory=ResourceConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    gesture: GestureConfig = Field(default_factory=GestureConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    api: APIConfig = Field(default_factory=APIConfig)


# ---------- Loader ----------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config root must be a mapping: {path}")
        return data
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML {path}: {e}") from e

def _parse_env_value(value: str) -> Any:
    """Parse env var string ke Python value.

    Supports: int, float, bool, JSON (dict/list), string fallback.
    """
    v = value.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    # Try JSON (untuk dict/list)
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        import json
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    return value


def _collect_env_overrides(prefix: str = "NEXUS_", delimiter: str = "__") -> dict:
    """Collect environment variables dengan prefix NEXUS_ dan nested delimiter __.

    Contoh:
        NEXUS_VISION__CAMERA__DEVICE=1
        → {"vision": {"camera": {"device": 1}}}

        NEXUS_NEXUS__LOG_LEVEL=DEBUG
        → {"nexus": {"log_level": "DEBUG"}}
    """
    result: dict = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        # Strip prefix
        key_stripped = key[len(prefix):]
        if not key_stripped:
            continue
        # Split by delimiter
        parts = key_stripped.split(delimiter)
        # Build nested dict
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part.lower(), {})
        node[parts[-1].lower()] = _parse_env_value(value)
    return result


def load_config(
    config_dir: str | Path | None = None,
    env: str | None = None,
    include_env_vars: bool = True,
) -> NexusConfig:
    """Load config dengan layered override.

    Args:
        config_dir: direktori config. Default: ./config atau $NEXUS_CONFIG_DIR
        env: nama environment (development/production/test).
             Default: $NEXUS_ENV atau 'development'
        include_env_vars: apakah baca env vars NEXUS_* sebagai override.
    """
    env = env or os.environ.get("NEXUS_ENV", "development")
    config_dir = Path(
        config_dir or os.environ.get("NEXUS_CONFIG_DIR", "./config")
    ).resolve()

    if not config_dir.exists():
        raise ConfigError(f"Config directory not found: {config_dir}")

    # Layer 1: default.yaml
    merged = _load_yaml(config_dir / "default.yaml")

    # Layer 2: {env}.yaml
    env_data = _load_yaml(config_dir / f"{env}.yaml")
    merged = _deep_merge(merged, env_data)

    # Layer 3: environment variables (opsional)
    if include_env_vars:
        env_overrides = _collect_env_overrides()
        merged = _deep_merge(merged, env_overrides)

    # Construct
    try:
        config = NexusConfig(**merged)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e

    return config


# ---------- Runtime singleton ----------

_config: NexusConfig | None = None
_overrides: dict[str, Any] = {}


def get_config() -> NexusConfig:
    """Get the global config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    if _overrides:
        data = _config.model_dump()
        data = _deep_merge(data, _overrides)
        return NexusConfig(**data)
    return _config


def set_config(config: NexusConfig) -> None:
    """Set global config (untuk testing)."""
    global _config
    _config = config


def override_config(**kwargs: Any) -> None:
    """Apply runtime override. Nested via double underscore."""
    global _overrides
    for key, value in kwargs.items():
        parts = key.split("__")
        node = _overrides
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def reset_config() -> None:
    """Reset global config (untuk testing)."""
    global _config, _overrides
    _config = None
    _overrides = {}