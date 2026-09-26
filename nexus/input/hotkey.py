"""Global hotkey manager.

Menggunakan `pynput` untuk listen keyboard events di background thread.
Hotkey bisa di-bind ke callback. Cocok untuk trigger manual tanpa HUD.

Install:
    pip install pynput
"""
from __future__ import annotations

import threading
from typing import Callable

from nexus.core.config import get_config
from nexus.core.logging import get_logger


log = get_logger(__name__)


class HotkeyError(Exception):
    pass


class HotkeyManager:
    """Global hotkey manager.

    Usage:
        mgr = HotkeyManager()
        mgr.register("ctrl+alt+n", callback_fn)
        mgr.start()
        ...
        mgr.stop()
    """

    def __init__(self):
        self._bindings: dict[str, Callable[[], None]] = {}
        self._listener = None
        self._started = False
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        """Register hotkey.

        Args:
            combo: combination string seperti "ctrl+alt+n"
            callback: function tanpa argumen
        """
        normalized = self._normalize(combo)
        self._bindings[normalized] = callback
        log.info("hotkey.register", combo=normalized)

    def start(self) -> None:
        if self._started:
            return

        try:
            from pynput import keyboard
        except ImportError as e:
            log.warning("hotkey.pynput_not_installed", error=str(e))
            self._available = False
            return

        self._available = True

        # Build hotkey mapping
        hotkey_map = {}
        for combo, callback in self._bindings.items():
            try:
                hk = self._to_pynput_hotkey(combo, keyboard)
                hotkey_map[hk] = callback
            except Exception:
                log.exception("hotkey.parse_error", combo=combo)

        if not hotkey_map:
            log.warning("hotkey.no_valid_bindings")
            return

        def on_activate(hotkey):
            callback = hotkey_map.get(hotkey)
            if callback is None:
                return
            try:
                log.debug("hotkey.activated", hotkey=str(hotkey))
                callback()
            except Exception:
                log.exception("hotkey.callback_error")

        try:
            self._listener = keyboard.GlobalHotKeys({
                self._to_pynput_combo(combo, keyboard): (lambda cb=cb: cb())
                for combo, cb in self._bindings.items()
            })
            self._listener.start()
            self._started = True
            log.info("hotkey.started", bindings=list(self._bindings.keys()))
        except Exception as e:
            log.exception("hotkey.start_error")
            self._available = False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                log.exception("hotkey.stop_error")
            self._listener = None
        self._started = False
        log.info("hotkey.stopped")

    def _normalize(self, combo: str) -> str:
        """Normalize "Ctrl+Alt+N" → "ctrl+alt+n"."""
        return "+".join(part.strip().lower() for part in combo.split("+"))

    def _to_pynput_hotkey(self, combo: str, keyboard_module) -> str:
        """Convert "ctrl+alt+n" ke format pynput "<ctrl>+<alt>+n"."""
        parts = combo.split("+")
        result = []
        for p in parts:
            p = p.strip().lower()
            if p in ("ctrl", "control"):
                result.append("<ctrl>")
            elif p == "alt":
                result.append("<alt>")
            elif p == "shift":
                result.append("<shift>")
            elif p in ("cmd", "super", "win", "meta"):
                result.append("<cmd>")
            else:
                result.append(p)
        return "+".join(result)

    def _to_pynput_combo(self, combo: str, keyboard_module) -> str:
        return self._to_pynput_hotkey(combo, keyboard_module)


# ---------- Singleton ----------

_manager: HotkeyManager | None = None


def get_hotkey_manager() -> HotkeyManager:
    global _manager
    if _manager is None:
        _manager = HotkeyManager()
    return _manager


def reset_hotkey_manager() -> None:
    global _manager
    if _manager is not None:
        _manager.stop()
    _manager = None