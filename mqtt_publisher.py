import json
import re
from collections import deque
from datetime import datetime

import paho.mqtt.client as mqtt

def _bold_units(s: str) -> str:
    """
    Bold common radio unit identifiers in a best-effort way.
    This only changes formatting for the dashboard (markdown ** **), not the underlying transcription.
    """

    if not s:
        return s

    # Bold Belchertown-style units like 52A1, 52E1, 52L1, 52BS2, etc.
    s = re.sub(r"\b(52[A-Z]{1,3}\d{1,3})\b", r"**\1**", s)

    # Bold "Bravo 6" / "BRAVO 13" style calls
    s = re.sub(r"\b(Bravo\s+\d{1,2})\b", r"**\1**", s, flags=re.IGNORECASE)

    return s

class MqttPublisher:
    """
    Publishes RadioTranscriber output to MQTT.

    Publishes:
      - {topic_base}/state (JSON payload)
      - {topic_base}/availability ("online"/"offline") if enabled

    Maintains a rolling 'recent' list (newest first).
    """

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.host = cfg.get("host", "localhost")
        self.port = int(cfg.get("port", 1883))
        self.username = cfg.get("username")
        self.password = cfg.get("password")
        self.topic_base = cfg.get("topic_base", "radiotranscriber/belchertown")
        self.retain = bool(cfg.get("retain", True))
        self.recent_max_lines = int(cfg.get("recent_max_lines", 25))
        self.clip_state_chars = int(cfg.get("clip_state_chars", 255))
        self.availability = bool(cfg.get("availability", True))

        self._recent = deque(maxlen=self.recent_max_lines)
        self._client = None

        if self.enabled:
            self._connect()

    def _connect(self):
        self._client = mqtt.Client()

        if self.username:
            self._client.username_pw_set(self.username, self.password)

        # Last Will and Testament: broker will publish offline if we disappear
        if self.availability:
            self._client.will_set(
                f"{self.topic_base}/availability",
                payload="offline",
                qos=0,
                retain=True,
            )

        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()

        if self.availability:
            self._client.publish(
                f"{self.topic_base}/availability",
                payload="online",
                qos=0,
                retain=True,
            )

    def close(self):
        if not self.enabled or not self._client:
            return

        try:
            if self.availability:
                self._client.publish(
                    f"{self.topic_base}/availability",
                    payload="offline",
                    qos=0,
                    retain=True,
                )
        finally:
            self._client.loop_stop()
            self._client.disconnect()

    def publish_transcript(self, feed: str, text: str, duration_s: float, hhmmss: str):
        """
        Publish a single transcript update.

        - 'text' is clipped to clip_state_chars for safe HA state usage.
        - 'recent' is updated newest-first.
        """
        if not self.enabled or not self._client:
            return

        safe_text = text or ""

        # Clip state-safe text
        clipped = False
        if len(safe_text) > self.clip_state_chars:
            safe_text = safe_text[: self.clip_state_chars]
            clipped = True

        # Add newest line to rolling list (newest first)
        pretty_text = _bold_units(text)
        recent_line = f"<span style='color: var(--secondary-text-color);'>[{hhmmss}]</span> {pretty_text}"
        self._recent.appendleft(recent_line)

        payload = {
            "feed": feed,
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_s": float(duration_s),
            "text": safe_text,
            "text_clipped": clipped,
            "recent": list(self._recent),
        }

        self._client.publish(
            f"{self.topic_base}/state",
            payload=json.dumps(payload, ensure_ascii=False),
            qos=0,
            retain=self.retain,
        )
