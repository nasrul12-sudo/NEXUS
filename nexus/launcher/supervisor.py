"""Process Supervisor — spawn, monitor, restart NEXUS processes."""
from __future__ import annotations

import multiprocessing as mp
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from nexus.core.config import get_config
from nexus.core.errors import ProcessError
from nexus.core.logging import get_logger, setup_logging
from nexus.core.process import NexusProcess, spawn_process


log = get_logger(__name__)


@dataclass
class ProcessSpec:
    name: str
    process_class: type[NexusProcess]         # ← CLASS, bukan factory
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    restart_on_crash: bool = True
    max_restarts: int = 3
    restarts: int = 0
    proc: mp.Process | None = None
    last_start: float = 0.0


class Supervisor:
    """Supervisor untuk semua NEXUS processes."""

    def __init__(self):
        self._specs: dict[str, ProcessSpec] = {}
        self._shutdown = False
        cfg = get_config()
        self._shutdown_timeout = cfg.process.shutdown_timeout_seconds

    def register(self, spec: ProcessSpec) -> None:
        if spec.name in self._specs:
            raise ProcessError(f"Process '{spec.name}' already registered")
        self._specs[spec.name] = spec
        log.info("supervisor.register", name=spec.name)

    def start_all(self) -> None:
        for spec in self._specs.values():
            self._start(spec)

    def _start(self, spec: ProcessSpec) -> None:
        if spec.proc is not None and spec.proc.is_alive():
            log.warning("supervisor.already_running", name=spec.name)
            return
        try:
            spec.proc = spawn_process(
                spec.process_class, *spec.args, **spec.kwargs
            )
        except Exception as e:
            log.exception("supervisor.spawn_error", name=spec.name)
            raise ProcessError(
                f"Failed to spawn process '{spec.name}': {e}"
            ) from e
        spec.last_start = time.time()
        log.info(
            "supervisor.start",
            name=spec.name,
            pid=spec.proc.pid,
            restarts=spec.restarts,
        )

    def _handle_crash(self, spec: ProcessSpec) -> None:
        if self._shutdown:
            return
        if not spec.restart_on_crash:
            log.error("supervisor.no_restart", name=spec.name)
            return
        if spec.restarts >= spec.max_restarts:
            log.error(
                "supervisor.max_restarts",
                name=spec.name,
                restarts=spec.restarts,
            )
            return
        spec.restarts += 1
        backoff = min(2 ** spec.restarts, 30)
        log.warning(
            "supervisor.restart",
            name=spec.name,
            attempt=spec.restarts,
            backoff_seconds=backoff,
        )
        time.sleep(backoff)
        self._start(spec)

    def _monitor_loop(self) -> None:
        while not self._shutdown:
            for spec in self._specs.values():
                proc = spec.proc
                if proc is None:
                    continue
                if not proc.is_alive():
                    exitcode = proc.exitcode
                    log.error(
                        "supervisor.process_died",
                        name=spec.name,
                        exitcode=exitcode,
                    )
                    spec.proc = None
                    if exitcode != 0:
                        self._handle_crash(spec)
                    else:
                        log.info("supervisor.process_exited_clean", name=spec.name)
            time.sleep(0.5)

    def stop_all(self) -> None:
        self._shutdown = True
        log.info("supervisor.shutdown.start")
        for spec in self._specs.values():
            if spec.proc is not None and spec.proc.is_alive():
                log.info("supervisor.terminate", name=spec.name, pid=spec.proc.pid)
                spec.proc.terminate()
        deadline = time.time() + self._shutdown_timeout
        for spec in self._specs.values():
            if spec.proc is None:
                continue
            remaining = max(0, deadline - time.time())
            spec.proc.join(timeout=remaining)
            if spec.proc.is_alive():
                log.warning("supervisor.kill", name=spec.name)
                spec.proc.kill()
                spec.proc.join(timeout=2)
        log.info("supervisor.shutdown.done")

    def run_forever(self) -> None:
        self._install_signal_handlers()
        self.start_all()
        try:
            self._monitor_loop()
        finally:
            self.stop_all()

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            log.info("supervisor.signal", signum=signum)
            self._shutdown = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


def main() -> None:
    """Entry point: `python -m nexus.launcher.supervisor`."""
    setup_logging(process_name="supervisor")
    log.info("supervisor.starting")

    supervisor = Supervisor()

    try:
        from nexus.vision.process import VisionProcess
        supervisor.register(ProcessSpec("vision", VisionProcess))
    except ImportError as e:
        log.warning("supervisor.vision_unavailable", error=str(e))

    try:
        from nexus.voice.process import VoiceProcess
        supervisor.register(ProcessSpec("voice", VoiceProcess))
    except ImportError as e:
        log.warning("supervisor.voice_unavailable", error=str(e))

    try:
        from nexus.brain.process import BrainProcess
        supervisor.register(ProcessSpec("brain", BrainProcess))
    except ImportError as e:
        log.warning("supervisor.brain_unavailable", error=str(e))

    try:
        from nexus.action.process import ActionProcess
        supervisor.register(ProcessSpec("action", ActionProcess))
    except ImportError as e:
        log.warning("supervisor.action_unavailable", error=str(e))

    if not supervisor._specs:
        log.error("supervisor.no_processes")
        return

    supervisor.run_forever()


if __name__ == "__main__":
    main()