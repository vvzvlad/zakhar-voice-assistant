import React, { useEffect, useState, useRef } from "react";
import { Field, StatusPill, Modal, Stepper, Select, Slider, Toggle, Seg, ScaleSeg } from "./primitives.jsx";
import DeviceLog from "./DeviceLog.jsx";
import { useAppData } from "../appData.jsx";
import {
  startCapture, getCaptureStatus, downloadCaptureResult, getDeviceControls, setDeviceControl,
  sayOnDevice, getWakeProbability, setWakeProbabilityStream, getWakeModels, uploadWakeModel,
} from "../api.js";
import { deviceVersion, fwVersion } from "../format.js";

const DEVICE_TABS = ["General", "Wake word", "Sound", "Stop word", "Log"];

// Add mode (`onSave({name, enabled})`) asks for a device id. Edit mode (`device`, the saved id)
// shows the device's status, its id — what the firmware sends in hello — and its versions above
// five tabs: General (the display label, the enabled switch and any live control no other tab
// places), Wake word (the wake word model, the live wake probability, the cutoffs, sample
// capture), Sound (the speaker volume, a test phrase, the auto volume switch and, while it is on,
// its settings and readings, the reply compression switch, the playback reserve), Stop word (the
// echo cancellation) and Log (the device log switch and the log the server keeps). `onSave` gets
// only the changed ones of {label, enabled, wake_model, log, auto_volume, compress_replies}
// (wake_model "" = the release default; an uploaded .tflite becomes the selected model), whichever
// tabs they were edited on; a tab holding such a change carries a dot. The live controls are
// written at once, each marked "live", polled for the whole life of the dialog. All tabs stay
// mounted and only the open one shows, so a running capture or a typed phrase outlives a tab
// switch; the log alone mounts with its tab, so it is polled only while shown. `online` says
// whether the device is reachable, `logDays` how many days of device log the server keeps,
// `onDelete` deletes the device (the footer's Delete device).
export default function DeviceModal({ onSave, onClose, onDelete, title, device, label: savedLabel = "", enabled: savedEnabled = true, wakeModel: savedWakeModel = "", log: savedLog = false, autoVolume: savedAutoVolume = false, compressReplies: savedCompressReplies = false, logDays, online }) {
  const [name, setName] = useState("");
  const [label, setLabel] = useState(savedLabel);
  const [enabled, setEnabled] = useState(savedEnabled);
  const [wakeModel, setWakeModel] = useState(savedWakeModel);
  const [log, setLog] = useState(savedLog);
  const [autoVolume, setAutoVolume] = useState(savedAutoVolume);
  const [compressReplies, setCompressReplies] = useState(savedCompressReplies);
  const [wakeModels, setWakeModels] = useState(null); // {options, release} from GET /api/wake-models
  const [uploadErr, setUploadErr] = useState(null);
  const [tab, setTab] = useState("General");
  const ctl = useDeviceControls(device);
  const fileRef = useRef(null);
  const tabsRef = useRef(null);
  // A new tab opens at its first row. Once the sticky tabs bar has caught on, the scroll past
  // that point shows as a gap between the bar and the header it normally touches; drop just that.
  function pickTab(t) {
    const bar = tabsRef.current;
    if (bar) bar.parentElement.scrollTop -= bar.getBoundingClientRect().top - bar.previousElementSibling.getBoundingClientRect().bottom;
    setTab(t);
  }
  useEffect(() => {
    if (!device) return undefined;
    let alive = true;
    getWakeModels()
      .then((r) => { if (alive) setWakeModels(r); })
      .catch(() => { /* the model field stays hidden */ });
    return () => { alive = false; };
  }, [device]);
  // Upload the picked .tflite, re-read the list and select the new model (Save sends it).
  async function upload(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // picking the same file again fires another change
    if (!file) return;
    setUploadErr(null);
    try {
      const { name: uploaded } = await uploadWakeModel(file);
      setWakeModels(await getWakeModels());
      setWakeModel(uploaded);
    } catch (err) {
      setUploadErr(err.message);
    }
  }
  if (device) {
    // The tabs holding a field Save would send.
    const marked = DEVICE_TABS.filter((t) => ({
      General: label.trim() !== savedLabel || enabled !== savedEnabled,
      "Wake word": wakeModel !== savedWakeModel,
      Sound: autoVolume !== savedAutoVolume || compressReplies !== savedCompressReplies,
      Log: log !== savedLog,
    })[t]);
    const snapControls = ctl.snap?.controls || [];
    const controlField = (c) => <ControlField key={c.id} control={c} value={ctl.valueOf(c)}
      onChange={(v) => ctl.write(c, v)} />;
    // The live controls of `ids`, in that order, each once the snapshot lists it.
    const controls = (...ids) => ids.map((id) => snapControls.find((c) => c.id === id)).filter(Boolean)
      .map(controlField);
    const loading = !ctl.snap && !ctl.err && <div className="z-f"><div className="z-fh">Loading…</div></div>;
    return <Modal top wide title={title} onClose={onClose}
      footer={<><button className="z-btn d" style={{ marginRight: "auto" }} onClick={onDelete}>Delete device</button>
        <button className="z-btn g" onClick={onClose}>Close</button>
        <button className="z-btn p" disabled={marked.length === 0} onClick={() => onSave({
          // Only the edited fields: the rest stays as in the server's fresh list (saveList).
          ...(label.trim() !== savedLabel && { label: label.trim() }),
          ...(enabled !== savedEnabled && { enabled }),
          ...(wakeModel !== savedWakeModel && { wake_model: wakeModel }),
          ...(log !== savedLog && { log }),
          ...(autoVolume !== savedAutoVolume && { auto_volume: autoVolume }),
          ...(compressReplies !== savedCompressReplies && { compress_replies: compressReplies }),
        })}>Save</button></>}>
      <div className="z-dev-stats">
        <Stat label="Status"><StatusPill status={online ? "online" : "offline"} /></Stat>
        <Stat label="Device id" title="Sent by the firmware when it connects — keys the device's settings and dialog context.">{device}</Stat>
        <Stat label="Firmware" title={deviceVersion(ctl.snap, "config_version")}>{fwVersion(deviceVersion(ctl.snap, "config_version")) ?? "—"}</Stat>
        <Stat label="Wake word model" title={deviceVersion(ctl.snap, "model_version")}>{deviceVersion(ctl.snap, "model_version") ?? "—"}</Stat>
      </div>
      <div className="z-tabs" ref={tabsRef}><Seg full options={DEVICE_TABS} value={tab} onChange={pickTab} marked={marked} /></div>
      {ctl.err && <div className="z-fh" style={{ color: "#b91c1c", marginTop: 8 }}>{ctl.err.message || String(ctl.err)}</div>}
      <div hidden={tab !== "General"}>
        <Field row label="Name" hint="Shown in the panel instead of the device id; empty shows the id."><div className="z-rowctl"><div className="z-inp"><input value={label} placeholder={device} onChange={(e) => setLabel(e.target.value)} /></div></div></Field>
        {/* Off closes the device's connection and refuses it from then on, on admits it again;
            a missing flag (old configs) is on. */}
        <Field row label="Enabled"><Toggle on={enabled} onChange={setEnabled} /></Field>
        {snapControls.filter((c) => !PLACED_CONTROLS.has(c.id)).map(controlField)}
      </div>
      <div hidden={tab !== "Wake word"}>
        {wakeModels && <>
          <Field row label="Wake word model" hint="Sent to the device on save and at every connection.">
            <div className="z-rowctl">
              <Select value={wakeModel} onChange={setWakeModel} options={[
                { value: "", label: `Release default (${wakeModels.release})` },
                ...wakeModels.options.map((n) => ({ value: n, label: n })),
              ]} />
            </div>
          </Field>
          {/* A row of its own, so the button does not read as a step after picking a model. The
              file is kept in the server's data volume; a new file of the same name replaces it
              and goes at once to every device using it. */}
          <Field row label="Upload model" hint="Picks it; the same file name replaces it on every device.">
            <div className="z-rowctl" style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
              <button className="z-btn g" onClick={() => fileRef.current.click()}>Upload</button>
              <input ref={fileRef} type="file" accept=".tflite" hidden aria-label="Wake word model file" onChange={upload} />
              {uploadErr && <div className="z-fh" style={{ color: "#b91c1c" }}>{uploadErr}</div>}
            </div>
          </Field>
        </>}
        <WakeProbabilityMeter device={device} online={online} />
        {loading}
        {controls("wake_probability_cutoff", "vad_cutoff")}
        <CaptureControl device={device} online={online} />
      </div>
      <div hidden={tab !== "Sound"}>
        {loading}
        {controls("speaker_volume")}
        <SayControl device={device} online={online} />
        {/* Every 15 s of quiet the device measures the room's noise on its raw microphone; the
            reply's volume is the dial's plus (noise − threshold) × gain. No reflash needed. */}
        <Field row label="Auto volume" hint="Louder replies in a noisy room, quieter in a quiet one; applies on save.">
          <Toggle on={autoVolume} onChange={setAutoVolume} />
        </Field>
        {autoVolume && <>
          {controls("auto_volume_threshold", "auto_volume_gain")}
          {ctl.snap?.readings.length > 0 && <Field row label="Auto volume now" hint="The room noise the device measured and the volume boost sent for it.">
            <div className="z-rowctl" style={{ display: "flex", gap: 24 }}>
              {ctl.snap.readings.map((r) => <Stat key={r.id} label={r.name}>
                {r.value == null ? "—" : withUnit(Number.isInteger(r.value) ? r.value : r.value.toFixed(1), r.unit)}
              </Stat>)}
            </div>
          </Field>}
        </>}
        {/* Quiet parts up to 20 dB louder, peaks held at −11.5 dBFS: louder for a TTS whose
            peaks sit below full scale (about +2–3 dB with Fish Audio); one that already peaks at
            full scale, as Piper does, plays up to ~3 dB quieter. No reflash needed. */}
        <Field row label="Compress replies" hint="Makes the quiet parts of replies louder; applies on save.">
          <Toggle on={compressReplies} onChange={setCompressReplies} />
        </Field>
        {controls("play_prime_ms")}
      </div>
      <div hidden={tab !== "Stop word"}>
        {loading}
        {controls("echo_cancel", "echo_cancel_mode", "echo_cancel_filter")}
      </div>
      <div hidden={tab !== "Log"} className="z-logpane">
        {/* The server keeps the log in data/device_logs/<device id>/; off, the device sends
            nothing. No reflash needed. */}
        <Field row label="Device log" hint={`Keeps the device's log on the server for ${logDays} days.`}>
          <Toggle on={log} onChange={setLog} />
        </Field>
        {tab === "Log" && <DeviceLog device={device} />}
      </div>
    </Modal>;
  }
  return <Modal top title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      <button className="z-btn p" disabled={!name} onClick={() => onSave({ name, enabled: true })}>Add</button></>}>
    <Field label="Device id" hint="The id its firmware sends when it connects (its Wi-Fi MAC) — also keys the dialog context."><div className="z-inp"><input value={name} placeholder="e.g. fc012cd1e5ec" onChange={(e) => setName(e.target.value)} /></div></Field>
  </Modal>;
}

// "Record X seconds" control shown inside the Edit device dialog. Each recording
// runs as a SERVER-SIDE background task (start -> poll countdown -> download WAV).
// With a count > 1 the browser loops that full cycle back-to-back, downloading one
// WAV per take — so you collect many wake-word samples without re-clicking. A Stop
// button ends the batch after the current take. Disabled (tooltip) while offline.
function CaptureControl({ device, online }) {
  const [seconds, setSeconds] = useState(30);
  const [count, setCount] = useState(1);      // how many takes to record back-to-back
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState("");     // progress / status line
  const [err, setErr] = useState(null);
  const [stopping, setStopping] = useState(false); // Stop pressed, waiting for current take to finish
  const cancelRef = useRef(false);            // Stop button -> end batch after current take

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // One full capture: start the server-side job, poll until terminal, download the WAV.
  // Returns true if a WAV was downloaded, false if the job ended without a result.
  async function captureOnce(idx, total) {
    await startCapture(device, seconds);
    for (;;) {
      const s = await getCaptureStatus(device);
      if (s.state === "starting" || s.state === "recording") {
        setPhase(`Recording ${idx}/${total}… ${s.remaining > 0 ? s.remaining + "s left" : "processing"}`);
        await sleep(1000);
        continue;
      }
      if (s.state === "done") { await downloadCaptureResult(device); return true; }
      if (s.state === "error") { throw new Error(s.error || "capture failed"); }
      return false;  // idle / unknown -> no result for this take
    }
  }

  // Run `count` takes sequentially. Stops early on the Stop button or a hard error.
  const run = async () => {
    setErr(null);
    cancelRef.current = false;
    setRunning(true);
    setStopping(false);
    let ok = 0;
    try {
      for (let i = 1; i <= count; i++) {
        if (cancelRef.current) break;
        setPhase(`Recording ${i}/${count}…`);
        if (await captureOnce(i, count)) ok += 1;
      }
      setPhase(cancelRef.current ? `Stopped: ${ok}/${count}` : `Done: ${ok}/${count}`);
    } catch (e) {
      setErr(e.message || "failed");
      setPhase(`Stopped at ${ok}/${count}`);
    } finally {
      setRunning(false);
      setStopping(false);
    }
  };

  return <Field row label="Capture sample" hint="Downloads mic audio as WAV files, for wake word training.">
    <div className="z-rowctl">
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        {running
          ? <>
              <span className="z-fh">{(phase || "…") + (stopping ? " — will stop after this take" : "")}</span>
              <button
                className={stopping ? "z-btn warn" : "z-btn d"}
                disabled={stopping}
                onClick={() => { cancelRef.current = true; setStopping(true); }}
              >{stopping ? "Stopping…" : "Stop"}</button>
            </>
          : <>
              <Stepper sm value={seconds} min={1} max={300} onChange={setSeconds} unit="s" />
              <Stepper sm value={count} min={1} max={1000} onChange={setCount} unit="takes" />
              <button className="z-btn p" style={{ marginLeft: "auto" }} disabled={!online} title={online ? "" : "Device offline"}
                onClick={run}>Record</button>
            </>}
      </div>
      {!running && phase && <div className="z-fh">{phase}</div>}
      {err && <div className="z-fh" style={{ color: "#b91c1c" }}>{err}</div>}
    </div>
  </Field>;
}

// Short per-control hint shown under the control's label, keyed by control id.
//   echo_cancel: off, the stop word hears the wake word's input; 1..3 add the residual
//     echo suppression, which only the FD modes have.
//   play_prime_ms: the reserve is held again after the stream runs dry.
//   auto_volume_threshold: above it the reply is louder than the dial, below it quieter.
const CONTROL_HINTS = {
  wake_probability_cutoff: "Higher means fewer false wakes but more missed ones.",
  vad_cutoff: "Lower drops fewer wakes but lets a little more non-speech through.",
  echo_cancel: "Off, or echo suppression from mild to strongest (FD modes only).",
  echo_cancel_mode: "Low is cheaper, high cancels better; FD also suppresses echo.",
  echo_cancel_filter: "From 1 to 8; a change briefly restarts echo cancellation.",
  play_prime_ms: "Buffer before playback: fewer stutters, but a later reply.",
  auto_volume_threshold: "Room noise at which the reply plays at the dial's volume.",
  auto_volume_gain: "Volume % per dB of noise away from the threshold; 2.5 %/dB moves the reply 1 dB per dB.",
};

// The live controls DeviceModal places on its Wake word, Sound and Stop word tabs; any other
// control the server lists shows on General.
const PLACED_CONTROLS = new Set([
  "wake_probability_cutoff", "vad_cutoff",
  "speaker_volume", "auto_volume_threshold", "auto_volume_gain", "play_prime_ms",
  "echo_cancel", "echo_cancel_mode", "echo_cancel_filter",
]);

// Labels of the controls on the Stop word tab, whose server names repeat the tab's subject;
// every other control shows the server's name.
const CONTROL_LABELS = {
  echo_cancel: "Echo cancellation",
  echo_cancel_mode: "Mode",
  echo_cancel_filter: "Filter length",
};

// Controls shown as a segmented scale: the button labels, the stored value is the button's index.
const CONTROL_SCALES = {
  echo_cancel: ["Off", "1", "2", "3"],
  echo_cancel_mode: ["SR low", "SR high", "FD low", "FD high"],
};

// A value with its unit: "%" sticks to the number, any other unit follows a space.
const withUnit = (text, unit) => (!unit || unit === "%" ? `${text}${unit || ""}` : `${text} ${unit}`);

// A read-only figure: a small uppercase label over its mono value; `title` is the value's tooltip.
function Stat({ label, title, children }) {
  return <div className="z-stat">
    <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>{label}</div>
    <div className="mono" title={title} style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{children}</div>
  </div>;
}

// One live control: a ScaleSeg for the ids in CONTROL_SCALES, else a Slider over the range and
// step the server gives it; its label carries a "live" mark: it is written at once, not by Save.
function ControlField({ control: c, value, onChange }) {
  const scale = CONTROL_SCALES[c.id];
  const caption = <>{CONTROL_LABELS[c.id] || c.name}<span className="z-live" title="Applies at once, without Save">live</span></>;
  return <Field row label={caption} hint={CONTROL_HINTS[c.id]}>
    <div className="z-rowctl">
      {scale
        ? <ScaleSeg options={scale.map((label, i) => ({ value: i, label }))} value={value} onChange={onChange} />
        : <Slider min={c.min} max={c.max} step={c.step || 1} value={value} onChange={onChange}
            fmt={(val) => withUnit(c.step < 1 ? Number(val).toFixed(1) : Math.round(val), c.unit)} />}
    </div>
  </Field>;
}

// The per-device live state of the Edit device dialog, used once by DeviceModal so it keeps
// polling while the tabs switch: `snap` is the last snapshot {controls, readings, versions}, null
// before the first. The controls are the wake-word probability cutoff, the VAD cutoff, the stop
// word's echo cancellation, the playback reserve, the auto volume's threshold and gain (all from
// the server config) and the speaker volume (read from the connected device, listed only while a
// device with a speaker is connected); the readings are the auto volume's (the room's noise, the
// boost), each null before its first value; the versions are the connected firmware's. `write`
// sends a control's new value debounced, `valueOf` is the value to show, `err` the last failed
// load or write. A background poll keeps the snapshot in sync with external changes (e.g. the
// physical volume dial) without stomping a control the user is actively dragging.
function useDeviceControls(device) {
  const { refreshConfig } = useAppData();
  const [snap, setSnap] = useState(null);
  const [err, setErr] = useState(null);
  const [local, setLocal] = useState({});        // id -> in-flight value (responsive drag)
  const pending = useRef({});                     // id -> bool (freeze poll-sync while writing)
  const timers = useRef({});                      // id -> debounce timeout handle

  // Apply a fresh snapshot, skipping controls the user is actively writing.
  const applySnapshot = (s) => {
    setSnap({ controls: s.controls || [], readings: s.readings || [], versions: s.versions || [] });
    setLocal((prev) => {
      const next = { ...prev };
      for (const c of s.controls || []) {
        if (!pending.current[c.id] && c.value != null) next[c.id] = c.value;
      }
      return next;
    });
  };

  // Initial load + background poll while the modal is open.
  useEffect(() => {
    if (!device) return undefined;
    let alive = true;
    const load = () => getDeviceControls(device)
      .then((snap) => { if (alive) { applySnapshot(snap); setErr(null); } })
      .catch((e) => { if (alive) setErr(e); });
    load();
    const iv = setInterval(load, 2500);
    return () => { alive = false; clearInterval(iv); };
  }, [device]);

  // Clear any pending debounce timers on unmount (modal close).
  useEffect(() => () => { for (const t of Object.values(timers.current)) clearTimeout(t); }, []);

  const write = (c, v) => {
    setLocal((prev) => ({ ...prev, [c.id]: v }));
    pending.current[c.id] = true;
    if (timers.current[c.id]) clearTimeout(timers.current[c.id]);
    timers.current[c.id] = setTimeout(() => {
      // `device` is a stable prop for the modal's lifetime, so no stale-closure risk.
      // A cutoff, an echo cancellation setting, the playback reserve or an auto volume setting is
      // stored in the server config: refresh the config snapshot so a later save of the device
      // list carries the new value.
      setDeviceControl(device, c.id, v)
        .then((snap) => { applySnapshot(snap); setErr(null); return refreshConfig(); })
        .catch((e) => setErr(e))
        .finally(() => { pending.current[c.id] = false; });
    }, 250);
  };

  return { snap, err, write, valueOf: (c) => local[c.id] ?? c.value ?? c.min ?? 0 };
}

// "Say on device" control shown on the Sound tab of the Edit device dialog, under the speaker
// volume: speaks the typed text on the device through the TTS, so the volume can be checked by
// ear. The request resolves once the speech has played. Disabled (tooltip) while offline.
function SayControl({ device, online }) {
  const [text, setText] = useState("Проверка громкости.");
  const [saying, setSaying] = useState(false);
  const [err, setErr] = useState(null);

  const say = async () => {
    setErr(null);
    setSaying(true);
    try {
      await sayOnDevice(device, text);
    } catch (e) {
      setErr(e.message || "failed");
    } finally {
      setSaying(false);
    }
  };

  return <Field row label="Say on device" hint="Speaks the text on the device to check the volume.">
    <div className="z-rowctl">
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <div className="z-inp" style={{ flex: 1 }}><input value={text} onChange={(e) => setText(e.target.value)} /></div>
        <button className="z-btn p" disabled={!online || !text.trim() || saying} title={online ? "" : "Device offline"}
          onClick={say}>{saying ? "Saying…" : "Say"}</button>
      </div>
      {err && <div className="z-fh" style={{ color: "#b91c1c" }}>{err}</div>}
    </div>
  </Field>;
}

// Live, read-only Wake Probability meter shown on the Wake word tab of the Edit device dialog
// (only for an EXISTING device). The stream is gated ON THE DEVICE: opening the modal enables it
// (POST enabled=true), closing it disables it (POST enabled=false), so the firmware only streams
// while the panel is open. While open we poll once a second and render the
// on-device PEAK wake-word probability over the last second as a percent + a thin bar.
// The enable/disable are idempotent and tolerate React StrictMode's double-invoke (the
// server forwards `wake_peak_stream {on}` to the device; failures are swallowed since
// the device may be offline). `online` seeds reachability and is refreshed from each
// snapshot.
function WakeProbabilityMeter({ device, online: onlineProp }) {
  const [online, setOnline] = useState(onlineProp);
  const [value, setValue] = useState(null);         // latest percent (number) or null
  const [err, setErr] = useState(null);

  // Single effect owns the whole lifecycle: enable on open, poll while open, disable on
  // close. `device` is stable for the modal's lifetime, so the cleanup disables the same
  // device it enabled. Both enable and disable are best-effort (offline -> ignore).
  useEffect(() => {
    if (!device) return undefined;
    let alive = true;
    // Best-effort enable; ignore errors (the device may be offline).
    setWakeProbabilityStream(device, true).catch(() => {});
    const poll = () => getWakeProbability(device)
      .then((snap) => {
        if (!alive) return;
        setOnline(snap.online);
        setValue(snap.value ?? null);
        setErr(null);
      })
      .catch((e) => { if (alive) setErr(e); });
    poll();
    const iv = setInterval(poll, 1000);
    return () => {
      alive = false;
      clearInterval(iv);
      // Best-effort disable so the device stops streaming once nobody is watching.
      setWakeProbabilityStream(device, false).catch(() => {});
    };
  }, [device]);

  let body;
  if (!online) {
    body = <div className="z-fh">Device offline.</div>;
  } else {
    const pct = value == null ? null : Math.max(0, Math.min(100, value));
    // The slider's track and value markup without a thumb, so it lines up with the sliders below.
    body = <div className="z-slider">
      <div className="z-trk" style={{ cursor: "default" }}><i style={{ width: `${pct == null ? 0 : pct}%`, transition: "width .2s" }} /></div>
      <span className="z-sval">{pct == null ? "…" : `${Math.round(pct)}%`}</span>
    </div>;
  }

  return <Field row label="Wake probability" hint="The device's wake word probability, peak over the last second.">
    <div className="z-rowctl">
      {body}
      {err && <div className="z-fh" style={{ color: "#b91c1c", marginTop: 6 }}>
        {(err && (err.message || String(err))) || "Failed to read wake probability."}
      </div>}
    </div>
  </Field>;
}
