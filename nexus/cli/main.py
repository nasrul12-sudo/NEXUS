"""NEXUS CLI — control interface dari terminal."""
from __future__ import annotations

import argparse
import json
import sys
import time

from nexus.core.bus import EventBus
from nexus.core.config import load_config, set_config
from nexus.core.logging import setup_logging


def cmd_status(args: argparse.Namespace) -> int:
    """Show NEXUS status."""
    from nexus.core.config import get_config
    from nexus.vision.hand_tracking import MediaPipeHandTracker
    from nexus.vision.camera import OpenCVCameraProvider

    cfg = get_config()
    print("NEXUS Status")
    print("=" * 40)
    print(f"Name:       {cfg.nexus.name}")
    print(f"Mode:       {cfg.nexus.mode}")
    print(f"Log level:  {cfg.nexus.log_level}")
    print()
    print(f"Bus:        {cfg.bus.publish_endpoint}")
    print(f"State:      {cfg.state.backend}")
    print(f"GPU:        {'enabled' if cfg.resource.gpu.enabled else 'disabled'}")
    print()
    print(f"Vision camera:  device={cfg.vision.camera.device} "
          f"{cfg.vision.camera.width}x{cfg.vision.camera.height}@{cfg.vision.camera.fps}")
    print(f"Hand tracking:  max_hands={cfg.vision.hand_tracking.max_hands}")
    print(f"Gesture:        enabled={cfg.gesture.enabled} engine={cfg.gesture.engine}")
    print()
    print(f"TTS:            enabled={cfg.output.tts.enabled}")
    print(f"Notification:   enabled={cfg.output.notification.enabled}")
    print(f"Hotkeys:        enabled={cfg.input.hotkeys.enabled}")
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    """Send command to NEXUS via event bus."""
    from nexus.core.events import make_event
    from nexus.core.bus import EventBus

    bus = EventBus()
    with bus.publisher() as pub:
        evt = make_event(
            type="cli.command",
            source="system",
            payload={"text": args.text, "origin": "cli"},
        )
        pub.publish(evt)
        print(f"Sent: {args.text}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Watch event stream."""
    from nexus.core.bus import EventBus
    from nexus.core.events import EventEnvelope

    bus = EventBus()
    topics = args.topics.split(",") if args.topics else None

    print(f"Watching events (topics={topics or '*'})... Ctrl+C to stop")
    print("-" * 60)

    def on_event(evt: EventEnvelope) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(evt.ts))
        payload_short = json.dumps(evt.payload, default=str)
        if len(payload_short) > 100:
            payload_short = payload_short[:97] + "..."
        print(f"[{ts}] {evt.source:6s} {evt.type:25s} {payload_short}")

    try:
        bus.subscribe_loop(on_event, topics=topics)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Get/set config value."""
    from nexus.core.config import get_config

    cfg = get_config()
    if args.action == "get":
        if not args.key:
            # Print full config
            print(json.dumps(cfg.model_dump(), indent=2, default=str))
            return 0
        # Get nested key (dot notation)
        parts = args.key.split(".")
        node = cfg.model_dump()
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                print(f"Key not found: {args.key}", file=sys.stderr)
                return 1
        print(json.dumps(node, indent=2, default=str))
        return 0
    elif args.action == "set":
        print("Runtime config set belum diimplementasikan (butuh IPC ke process).")
        return 1
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nexus-cli",
        description="NEXUS command-line interface",
    )
    parser.add_argument("--config-dir", type=str, default=None)
    parser.add_argument("--env", type=str, default="development")
    parser.add_argument("--log-level", type=str, default="WARNING")

    sub = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = sub.add_parser("status", help="Show NEXUS status")
    p_status.set_defaults(func=cmd_status)

    # send
    p_send = sub.add_parser("send", help="Send command text")
    p_send.add_argument("text", help="Command text")
    p_send.set_defaults(func=cmd_send)

    # watch
    p_watch = sub.add_parser("watch", help="Watch event stream")
    p_watch.add_argument("--topics", type=str, default=None,
                         help="Comma-separated topic prefixes")
    p_watch.set_defaults(func=cmd_watch)

    # config
    p_config = sub.add_parser("config", help="Get/set config")
    p_config.add_argument("action", choices=["get", "set"])
    p_config.add_argument("key", nargs="?", help="Dot-notation key")
    p_config.add_argument("value", nargs="?", help="Value (untuk set)")
    p_config.set_defaults(func=cmd_config)

    args = parser.parse_args()

    setup_logging(level=args.log_level, process_name="cli")
    cfg = load_config(config_dir=args.config_dir, env=args.env)
    set_config(cfg)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())