# scripts/debug_bus.py
import time
from nexus.core.bus import Publisher, Subscriber
from nexus.core.events import make_event, EventType

endpoint = "ipc:///tmp/nexus-debug.sock"

print("=== Publisher start ===")
pub = Publisher(endpoint)
print(f"Publisher bound: {endpoint}")

print("=== Subscriber start ===")
sub = Subscriber(endpoint, topics=["test."])
print(f"Subscriber handshake_ok: {sub.handshake_ok}")

print("=== Wait for subscribers ===")
ok = pub.wait_for_subscribers(1, timeout=3.0)
print(f"wait_for_subscribers: {ok}, count={pub.subscriber_count}")

print("=== Publish ===")
evt = make_event(EventType.GESTURE_STABLE, "vision", {"gesture": "TEST"})
pub.publish(evt)
print(f"Published: {evt.type}")

print("=== Recv ===")
for i in range(20):
    received = sub.recv()
    if received:
        print(f"Received after {i*10}ms: {received.type} {received.payload}")
        break
    time.sleep(0.01)
else:
    print("FAILED: no event received")

sub.close()
pub.close()