import pytest
from nexus.core.config import load_config, NexusConfig, reset_config
from nexus.core.errors import ConfigError


def test_load_default_config(tmp_path):
    (tmp_path / "default.yaml").write_text(
        "nexus:\n  name: NEXUS\n  log_level: INFO\n"
    )
    cfg = load_config(config_dir=tmp_path, env="development")
    assert cfg.nexus.name == "NEXUS"
    assert cfg.nexus.log_level == "INFO"


def test_layered_override(tmp_path):
    (tmp_path / "default.yaml").write_text(
        "nexus:\n  log_level: INFO\nbus:\n  hwm: 1000\n"
    )
    (tmp_path / "development.yaml").write_text(
        "nexus:\n  log_level: DEBUG\n"
    )
    cfg = load_config(config_dir=tmp_path, env="development")
    assert cfg.nexus.log_level == "DEBUG"
    assert cfg.bus.hwm == 1000


def test_env_var_override(tmp_path, monkeypatch):
    (tmp_path / "default.yaml").write_text("nexus:\n  log_level: INFO\n")
    monkeypatch.setenv("NEXUS_NEXUS__LOG_LEVEL", "WARNING")
    cfg = load_config(config_dir=tmp_path, env="development")
    assert cfg.nexus.log_level == "WARNING"


def test_missing_config_dir():
    with pytest.raises(ConfigError):
        load_config(config_dir="/nonexistent/path")

def test_vision_config_defaults(tmp_path):
    (tmp_path / "default.yaml").write_text("nexus:\n  name: NEXUS\n")
    cfg = load_config(config_dir=tmp_path, env="development")
    assert cfg.vision.camera.device == 0
    assert cfg.vision.camera.width == 1280
    assert cfg.vision.hand_tracking.max_hands == 2


def test_vision_config_override(tmp_path):
    (tmp_path / "default.yaml").write_text(
        "vision:\n  camera:\n    device: 1\n    width: 640\n"
    )
    cfg = load_config(config_dir=tmp_path, env="development")
    assert cfg.vision.camera.device == 1
    assert cfg.vision.camera.width == 640