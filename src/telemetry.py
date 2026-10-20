"""MQTT telemetry: the server's and its devices' state and events, published to the smart
home's broker (core.telemetry) so that a failure can be taken apart afterwards from InfluxDB.

The broker's mqtt2influx bridge writes every leaf of a JSON object payload as its own
InfluxDB 1.8 measurement: the topic plus the key path, lowercased, spaces turned into `_`,
field `value`. A number is stored as a number, a bool as 10 or 0, a text (on the bridge
stream that writes strings) as a string; null, an empty string and a list are dropped. A
measurement keeps one type forever: InfluxDB rejects a whole batch over a string written
where floats are. So every payload goes through `clean`: a text that would pass for a number
or a bool ("100", "1e3", "nan", "true") is sent inside double quotes and stays a text, a list
becomes an object keyed "0", "1", … in order, null and empty strings are left out. The bridge
skips keys starting with `_` and topics containing `_version`, `/meta/` or `/fragment/`, so no
key here has those.

Every message is JSON at QoS 1, not retained. While the broker is away the messages wait in
memory and go out when it is back; with telemetry off or no broker address nothing is published.
The bridge stamps a point with the time it got the message, so one that waited out an outage
lands at the reconnection: every payload also carries ts — the epoch seconds it was published
at — and the 30 s state is not published while the broker is away.

Topics, under the root `zakhar` (<name> is the device's hello id):

zakhar/server/event/started — once, when the process starts:
    version — the server's version (src/version.py).
zakhar/server/state — every 30 s:
    version; uptime — seconds since the process started; devices — how many enabled
    devices the config has; devices_online — how many of them are connected.
zakhar/devices/<name>/state — every 30 s, for every enabled configured device:
    online (10/0); label — its panel label, when set. While online also: uptime — seconds
    since the device booted; fw — its firmware string; model — the wake model it reported
    loaded; volume — its speaker volume, percent; ambient — the room's noise, dBFS, the
    median of the connection's last `ambient` values; boost — the `volume_boost` sent, percent.
zakhar/devices/<name>/event/<kind> — a device event, published the moment it happens:
    connected — a hello was admitted: fw; uptime — the hello's, seconds; reset — why the
        device last booted (the hello's `reset`, docs/device-protocol.md; absent from older
        firmware); speaker (10/0); replaced — an older connection of the device was closed;
        rebooted — this process saw the device before and its boot time moved later by more
        than 10 s.
    disconnected — the live connection closed: code — the WebSocket close code; error —
        the socket's exception, when there was one; connected_s — how long it was connected.
    wake — session; word; over_answer — it came over a running answer or an open session.
    stop — the stop word: session (absent outside a session); in_session (10/0).
    volume — value: the speaker volume the device reports, percent.
    ambient — value: the noise the device measured, dBFS; zeros (10/0) — the microphone
        read only zeros, which is no level (value absent).
    played — a play stream ended on the device: dry, gap_ms (times it ran dry, the ms of
        silence played meanwhile), or dropped (10).
zakhar/devices/<name>/stats — the device's `stats` message as received, without `type`
    (docs/device-protocol.md, Settings and diagnostics).
zakhar/devices/<device>/run/<result> — every finished run (a voice run, a text request, an
    announcement, an assistant task's reply), <device> its device or `server` for a run of
    none; <result> ok, tool, error, rejected or empty:
    run_id — its row in the run log (absent when the log is off or the insert failed);
    reason; chain — the recognition chain that served it; model; tokens; stt_text — what was
    heard; llm_text — the reply; stress_text — the reply with stress marks; filler_text;
    error_stage, error_text — the stage that failed and its error; wakeword_score; audio_ms;
    t_vad, t_stt, t_llm, t_stress, t_tts, t_filler, t_wakeword, t_total — every stage timing
    the run record has, ms; play_dry, play_gap_ms — the reply's playback as the device
    reported it;
    attempts/<kind>_<chain or provider> (`<kind>` for an entry with neither, a `/` in the
        name turned into `_`) — one per entry of the record's `attempts`, with all the
        entry's fields (kind and its chain or provider too): a recognition chain
        (`chain_<chain>`: stt, handler, text, outcome, error, detail, t_stt, t_handler;
        src/chains.py), the follow-up check (`followup_<provider>_<model>`: text, outcome,
        probability, t) and each TTS variant tried (`tts_<provider>`: outcome, error, t;
        src/pipeline.py); a key met again gets `_2`, `_3`, …;
    tools/<n> — every tool call of the run in order, n from 0: round, name, args (the
        arguments as JSON text), result.
"""

import asyncio
import contextlib
import json
import statistics
import time

import paho.mqtt.client as mqtt
from loguru import logger
from paho.mqtt.enums import CallbackAPIVersion

from src.version import __version__

ROOT = "zakhar"
STATE_INTERVAL_S = 30

# The run record's fields a run's payload carries as they are (besides its t_* timings).
_RUN_FIELDS = ("reason", "chain", "model", "tokens", "stt_text", "llm_text", "stress_text",
               "filler_text", "error_stage", "error_text", "wakeword_score", "audio_ms",
               "play_dry", "play_gap_ms")


def _passes_for_number(text: str) -> bool:
    """Whether the bridge would store `text` as a number: float() reads it or it is a bool."""
    if text.lower() in ("true", "false"):
        return True
    try:
        float(text)
    except ValueError:
        return False
    return True


def clean(value):
    """`value` as the bridge must get it to keep each leaf's type: None and "" left out,
    a list turned into a dict keyed "0", "1", …, dicts cleaned through, and a string the
    bridge would take for a number or a bool wrapped in double quotes."""
    if isinstance(value, list):
        value = {str(i): item for i, item in enumerate(value)}
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            item = clean(item)
            if item is not None:
                out[key] = item
        return out
    if isinstance(value, str):
        if value == "":
            return None
        if _passes_for_number(value):
            return f'"{value}"'
    return value


def _attempts(attempts: list) -> dict:
    """The record's attempts keyed `<kind>_<chain or provider>` (`<kind>` with neither), a
    key met again suffixed `_2`, `_3`, …"""
    out = {}
    for attempt in attempts:
        name = attempt.get("chain") or attempt.get("provider")
        # A "provider/model" would split the bridge's key path into an extra level per model.
        base = f"{attempt.get('kind')}_{name.replace('/', '_')}" if name else str(attempt.get("kind"))
        key, n = base, 2
        while key in out:
            key, n = f"{base}_{n}", n + 1
        out[key] = attempt
    return out


def _tools(rounds: list) -> list:
    """Every tool call of the record's rounds (src/llm.py), in order."""
    return [{"round": r.get("round"), "name": call.get("name"),
             "args": json.dumps(call.get("args"), ensure_ascii=False),
             "result": str(call.get("result"))}
            for r in rounds for call in r.get("calls") or []]


def run_payload(record: dict, run_id: int | None = None) -> dict:
    """The telemetry payload of a finished run's record (the module docstring lists it)."""
    payload = {"run_id": run_id, **{f: record.get(f) for f in _RUN_FIELDS}}
    payload.update({k: v for k, v in record.items() if k.startswith("t_")})
    payload["attempts"] = _attempts(record.get("attempts") or [])
    payload["tools"] = _tools(record.get("rounds") or [])
    return payload


def publish(rt, topic: str, payload: dict) -> None:
    """Publish `payload` to `zakhar/<topic>` through the runtime's Telemetry; nothing while
    the runtime has none (one built without the app)."""
    if rt.telemetry is not None:
        rt.telemetry.publish(topic, payload)


def publish_run(rt, record: dict, run_id: int | None = None) -> None:
    """Publish a finished run (its run log row `run_id`, if it has one); a failure is
    logged, never raised into the run."""
    try:
        device = record.get("device") or "server"
        publish(rt, f"devices/{device}/run/{record.get('result')}", run_payload(record, run_id))
    except Exception as e:
        logger.error(f"telemetry: run not published: {e}")


def device_state(device) -> dict:
    """A device's 30 s state (the module docstring lists it)."""
    conn = device.conn
    state = {"online": conn is not None, "label": device.cfg.label}
    if conn is not None:
        state.update(
            uptime=int(time.time() - conn.booted_at), fw=conn.fw, model=conn.model,
            volume=conn.volume, boost=conn.boost,
            ambient=statistics.median(conn.ambient) if conn.ambient else None,
        )
    return state


class Telemetry:
    """The MQTT publisher. paho's network thread connects, reconnects by itself while the
    broker is away and keeps the QoS 1 messages published meanwhile; start() and stop()
    follow core.telemetry (restarted by the Reconfigurator on a change)."""

    def __init__(self, rt, started_at: float):
        self.rt = rt
        self.started_at = started_at
        self._client: mqtt.Client | None = None
        self._state_task: asyncio.Task | None = None
        self._where = ""
        # The broker is away and that was logged: warned once per outage.
        self._down = False

    async def start(self) -> None:
        """Connect to the configured broker and start the 30 s state; nothing with telemetry
        off or no broker address."""
        cfg = self.rt.core.telemetry
        if not cfg.enabled or not cfg.host:
            return
        client = mqtt.Client(CallbackAPIVersion.VERSION2)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_connect_fail = self._on_connect_fail
        self._where = f"{cfg.host}:{cfg.port}"
        self._down = False
        # Resolves and connects in the network thread: the event loop never waits for it.
        client.connect_async(cfg.host, cfg.port)
        client.loop_start()
        self._client = client
        self._state_task = asyncio.create_task(self._state_loop())
        logger.info(f"telemetry: publishing to the MQTT broker {self._where}")

    async def stop(self) -> None:
        """Stop the state and close the broker connection; the messages not yet sent are lost."""
        task, self._state_task = self._state_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        client, self._client = self._client, None
        if client is not None:
            client.disconnect()
            # Joins the network thread, which may sit in a connect attempt for up to 5 s.
            await asyncio.to_thread(client.loop_stop)

    def publish(self, topic: str, payload: dict) -> None:
        """Publish the cleaned `payload` as JSON to `zakhar/<topic>` at QoS 1; a no-op while
        stopped. A failure is logged, never raised."""
        client = self._client
        if client is None:
            return
        try:
            payload = clean({**payload, "ts": round(time.time(), 3)})
            client.publish(f"{ROOT}/{topic}", json.dumps(payload, ensure_ascii=False), qos=1)
        except Exception as e:
            logger.error(f"telemetry: {topic} not published: {e}")

    def publish_state(self) -> None:
        """The server's state and every enabled configured device's; nothing while the broker
        is away, where it would only land late, all at the reconnection."""
        if self._client is None or not self._client.is_connected():
            return
        devices = self.rt.manager.clients
        self.publish("server/state", {
            "version": __version__, "uptime": int(time.time() - self.started_at),
            "devices": len(devices), "devices_online": sum(1 for d in devices if d.online),
        })
        for device in devices:
            self.publish(f"devices/{device.name}/state", device_state(device))

    async def _state_loop(self) -> None:
        while True:
            try:
                self.publish_state()
            except Exception as e:
                logger.error(f"telemetry: state not published: {e}")
            await asyncio.sleep(STATE_INTERVAL_S)

    # paho callbacks, called in its network thread.

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            self._lost(f"refused: {reason_code}")
            return
        self._down = False
        logger.info(f"telemetry: connected to the MQTT broker {self._where}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        # A disconnect stop() asked for comes with a success code.
        if reason_code.is_failure:
            self._lost(str(reason_code))

    def _on_connect_fail(self, client, userdata) -> None:
        self._lost("cannot connect")

    def _lost(self, why: str) -> None:
        """Warn of a broker away once per outage: paho retries it every 1 s to 2 min."""
        if not self._down:
            self._down = True
            logger.warning(f"telemetry: no MQTT broker at {self._where} ({why}); "
                           f"the messages wait in memory until it is back")
