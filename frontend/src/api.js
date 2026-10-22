// Thin fetch wrapper around the backend HTTP API (aiohttp at :8201, /api routes).
// Base URL is empty in production (panel serves the build → same origin); in dev
// VITE_API_BASE can point elsewhere, but the Vite proxy already forwards /api.

const BASE = import.meta.env.VITE_API_BASE ?? "";

// Error that carries the parsed JSON body of a failed response so forms can show
// validation messages. `status` is the HTTP code; `detail` is pydantic's errors[].
export class ApiError extends Error {
  constructor(message, { status, detail, body } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

async function request(path, { method = "GET", body } = {}) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) {
    // Sent as is: the browser sets the multipart Content-Type with its boundary.
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let resp;
  try {
    resp = await fetch(BASE + path, opts);
  } catch (e) {
    throw new ApiError("Failed to reach the server: " + e.message, { status: 0 });
  }
  // 202/204 and other 2xx with no JSON body are fine.
  let data = null;
  const text = await resp.text();
  if (text) {
    try { data = JSON.parse(text); } catch { data = text; }
  }
  if (!resp.ok) {
    const msg =
      (data && (data.error || data.detail)) ||
      `HTTP ${resp.status}`;
    throw new ApiError(
      typeof msg === "string" ? msg : `HTTP ${resp.status}`,
      { status: resp.status, detail: data && data.detail, body: data }
    );
  }
  return data;
}

export const getCatalog = () => request("/api/catalog");
export const getConfig = () => request("/api/config");
export const patchConfig = (patch) => request("/api/config", { method: "PATCH", body: patch });
// `q` is an optional server-side search string for fields with remote catalog
// search ("search": "remote"); omitted/empty -> the provider's baseline list.
export const getOptions = (category, plugin, field, q) =>
  request(`/api/options?category=${encodeURIComponent(category)}&plugin=${encodeURIComponent(plugin)}&field=${encodeURIComponent(field)}${q ? `&q=${encodeURIComponent(q)}` : ""}`);
// Bundled end-of-phrase chime files for the ack sound_path selector.
export const getChimes = () => request("/api/chimes");
// Wake-word models (the release's and the uploaded ones), for a device's wake_model
// selector: {options: [names], release: the name an empty wake_model falls back to}.
export const getWakeModels = () => request("/api/wake-models");
// Upload a .tflite wake-word model into the server's data volume. Returns
// {name: "uploaded/<stem>"}, the name getWakeModels now lists.
export const uploadWakeModel = (file) => {
  const form = new FormData();
  form.append("file", file);
  return request("/api/wake-models", { method: "POST", body: form });
};
// Play the selected end-of-phrase chime on the speaker(s) for an operator preview.
// soundPath "" = the synthesized default; device omitted = play on every online speaker.
// Returns {played: [names], offline: [names]}.
export const playChime = (soundPath, device) =>
  request("/api/chimes/play", {
    method: "POST",
    body: { sound_path: soundPath ?? "", ...(device ? { device } : {}) },
  });
// Back-compat prompt endpoints: GET returns the ACTIVE profile {id,name,text},
// PUT updates the active profile's text.
export const getPrompt = () => request("/api/prompt");
export const putPrompt = (text) => request("/api/prompt", { method: "PUT", body: { text } });
// Named system-prompt profiles: list (summaries + active_id), full read, CRUD and
// quick activation. createPromptProfile with `text` omitted copies the active
// profile's text server-side (duplicate-friendly default).
export const getPromptProfiles = () => request("/api/prompt/profiles");
export const getPromptProfile = (id) => request(`/api/prompt/profiles/${encodeURIComponent(id)}`);
export const createPromptProfile = (body) =>
  request("/api/prompt/profiles", { method: "POST", body });
export const updatePromptProfile = (id, body) =>
  request(`/api/prompt/profiles/${encodeURIComponent(id)}`, { method: "PUT", body });
export const deletePromptProfile = (id) =>
  request(`/api/prompt/profiles/${encodeURIComponent(id)}`, { method: "DELETE" });
// Activate a profile. opts.applyVoice also applies that profile's preferred-voice
// marker (if any) to the TTS config server-side.
export const activatePromptProfile = (id, { applyVoice = false } = {}) =>
  request(`/api/prompt/profiles/${encodeURIComponent(id)}/activate`,
    { method: "POST", body: { apply_voice: applyVoice } });
// The facts the assistant saved with its memory tool: GET returns {facts: [...]}, PUT
// replaces them and returns {facts} as stored (normalized, empty lines and repeats dropped).
export const getMemory = () => request("/api/memory");
export const putMemory = (facts) => request("/api/memory", { method: "PUT", body: { facts } });
export const getSystem = () => request("/api/system");
export const getDevices = () => request("/api/devices");

// Live device controls (wake-word probability cutoff + speaker volume) exposed by
// the firmware as 0..100 number entities. GET returns {device, online, controls:[{id,name,value,min,max,step,unit}]}.
export const getDeviceControls = (device) =>
  request(`/api/device/controls?device=${encodeURIComponent(device)}`);
// Set one control; returns the refreshed snapshot.
export const setDeviceControl = (device, control, value) =>
  request("/api/device/controls", { method: "POST", body: { device, control, value } });
// Speak the text on the speaker through the TTS; resolves once it has played.
export const sayOnDevice = (device, text) =>
  request("/api/device/say", { method: "POST", body: { device, text } });

// Live Wake Probability (on-device per-second PEAK, %). GET returns
// {device, online, available, value}. The stream is gated on the device: POST
// enabled=true when the device modal opens, enabled=false when it closes, so the
// firmware only streams the probability while someone is watching.
export const getWakeProbability = (device) =>
  request(`/api/device/wake-probability?device=${encodeURIComponent(device)}`);
export const setWakeProbabilityStream = (device, enabled) =>
  request("/api/device/wake-probability", { method: "POST", body: { device, enabled } });

// Capture: the device records on its own and the server writes the audio to disk as it
// arrives, so closing the browser does not cancel it. start kicks it off
// (202 -> status "starting"), getCaptureStatus polls for the live countdown, and
// downloadCaptureResult fetches the finished WAV.
export const startCapture = (device, seconds) =>
  request("/api/capture", { method: "POST", body: { device, seconds } });
export const getCaptureStatus = (device) =>
  request(`/api/capture?device=${encodeURIComponent(device)}`);

// Download the device's newest finished capture (it stays on the server until deleted)
// and trigger a browser download. The backend returns the WAV as an attachment; we
// fetch it as a blob and click a temporary <a download>. On a non-OK response the
// JSON/text error body is read and thrown as an ApiError, like request() does.
export async function downloadCaptureResult(device) {
  let resp;
  try {
    resp = await fetch(BASE + "/api/capture/result?device=" + encodeURIComponent(device));
  } catch (e) {
    throw new ApiError("Failed to reach the server: " + e.message, { status: 0 });
  }
  if (!resp.ok) {
    const text = await resp.text();
    let data = null;
    if (text) { try { data = JSON.parse(text); } catch { data = text; } }
    const msg = (data && (data.error || data.detail)) || `HTTP ${resp.status}`;
    throw new ApiError(
      typeof msg === "string" ? msg : `HTTP ${resp.status}`,
      { status: resp.status, detail: data && data.detail, body: data }
    );
  }
  const blob = await resp.blob();
  // Prefer the server-provided filename from Content-Disposition; else a default.
  const disp = resp.headers.get("Content-Disposition") || "";
  const m = /filename="?([^"]+)"?/.exec(disp);
  const filename = (m && m[1]) || `zakhar_${device}.wav`;
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Device logs: the speakers' own logs the server keeps on disk, one file per device and day.
// getDeviceLogs returns {devices: {id: [YYYY-MM-DD, newest first]}}; getDeviceLog returns a
// day's text from byte `offset` up to its last whole line: {text, offset: where to read on from}.
export const getDeviceLogs = () => request("/api/device-logs");
export const getDeviceLog = (device, day, offset) =>
  request(`/api/device-logs/${encodeURIComponent(device)}/${encodeURIComponent(day)}?offset=${offset}`);
// Start a voice test: POST the CURRENT (possibly unsaved) TTS draft to
// /api/tts/test and return the raw Response (a WAV body) for the caller to play.
// The backend always sends any error as a JSON body with a non-2xx status BEFORE
// the audio starts, so error shaping here mirrors request(). `signal` (optional) lets the caller abort the in-flight request.
export const streamTtsVoice = async (provider, settings, text, signal) => {
  let resp;
  try {
    resp = await fetch(BASE + "/api/tts/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, settings, text }),
      signal,
    });
  } catch (e) {
    // A caller-driven abort (Stop / unmount) is not a server error: re-throw it
    // unchanged so the component can recognise e.name === "AbortError" and stay
    // silent instead of flashing a red "failed to reach the server" line.
    if (e && e.name === "AbortError") throw e;
    throw new ApiError("Failed to reach the server: " + e.message, { status: 0 });
  }
  if (!resp.ok) {
    const body = await resp.text();
    let data = null;
    if (body) { try { data = JSON.parse(body); } catch { data = body; } }
    const msg = (data && (data.error || data.detail)) || `HTTP ${resp.status}`;
    throw new ApiError(
      typeof msg === "string" ? msg : `HTTP ${resp.status}`,
      { status: resp.status, detail: data && data.detail, body: data }
    );
  }
  return resp;
};

// Live tool sources (external MCP + built-ins) with their advertised tools.
export const getTools = () => request("/api/tools");

// --- observability (run log + metrics) -------------------------------------
// Builds /api/runs?device=&chain=&result=&search=&limit=&offset= from the given params,
// dropping empty/undefined ones so the backend sees a clean query string. `offset`
// is the number of matching rows to skip for offset-based numbered pagination.
export const getRuns = (params = {}) => {
  const q = new URLSearchParams();
  for (const k of ["device", "chain", "result", "search", "limit", "offset"]) {
    const v = params[k];
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const qs = q.toString();
  return request("/api/runs" + (qs ? "?" + qs : ""));
};
export const getRun = (id) => request(`/api/runs/${encodeURIComponent(id)}`);

// The run's JSON as a full address on the server this panel talks to: the Request Log's
// copy button hands it to agents, who fetch it with curl.
export const runLink = (id) => new URL(`${BASE}/api/runs/${encodeURIComponent(id)}`, window.location.href).href;

// Absolute URL of a run's stored utterance WAV (for an <audio> src). BASE is "" in
// prod (same origin) or an absolute base in dev. Optional channel ("stt" | "raw")
// asks the backend to split that channel out of a stereo recording as mono.
export const runAudioUrl = (id, channel) =>
  `${BASE}/api/runs/${encodeURIComponent(id)}/audio${channel ? `?channel=${encodeURIComponent(channel)}` : ""}`;

// Fetch a run's stored utterance WAV as a blob and trigger a browser download.
export async function downloadRunAudio(id) {
  let resp;
  try {
    resp = await fetch(runAudioUrl(id));
  } catch (e) {
    throw new ApiError("Failed to reach the server: " + e.message, { status: 0 });
  }
  if (!resp.ok) throw new ApiError(`HTTP ${resp.status}`, { status: resp.status });
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = `zakhar_run_${id}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Absolute URL of a run's stored generated TTS reply audio (for an <audio> src).
export const runTtsAudioUrl = (id) =>
  `${BASE}/api/runs/${encodeURIComponent(id)}/tts-audio`;

// Fetch a run's stored TTS reply audio as a blob and trigger a browser download.
// The native format varies (mp3/wav/flac), so derive the extension from the blob mime.
export async function downloadRunTtsAudio(id) {
  let resp;
  try {
    resp = await fetch(runTtsAudioUrl(id));
  } catch (e) {
    throw new ApiError("Failed to reach the server: " + e.message, { status: 0 });
  }
  if (!resp.ok) throw new ApiError(`HTTP ${resp.status}`, { status: resp.status });
  const blob = await resp.blob();
  const ext = (blob.type && blob.type.split("/")[1]) || "mp3";
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = `zakhar_run_${id}_tts.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}
export const getMetrics = () => request("/api/metrics");

// --- live panel stream (WebSocket) -----------------------------------------
// Opens a WebSocket to /api/runs/stream and surfaces every parsed message plus
// the connection state. The backend pushes two frame types: {type:"run",run}
// for finalized runs and {type:"system",...} heartbeats (~1/s) used for
// liveness + a live uptime. Auto-reconnects with capped exponential backoff.
// onStatus(connected:boolean) fires on open (true) and on close/error (false).
// Returns a stop() that closes the socket and cancels any pending reconnect.
export function openPanelStream({ onMessage, onStatus } = {}) {
  let ws = null;
  let stopped = false;
  let attempt = 0;
  let timer = null;

  const wsUrl = () => {
    const u = new URL("/api/runs/stream", BASE || window.location.origin);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.toString();
  };

  const schedule = () => {
    if (stopped) return;
    const delay = Math.min(500 * 2 ** attempt, 30000);
    attempt = Math.min(attempt + 1, 6);
    timer = setTimeout(connect, delay);
  };

  const connect = () => {
    if (stopped) return;
    try {
      ws = new WebSocket(wsUrl());
    } catch {
      schedule();
      return;
    }
    ws.onopen = () => { attempt = 0; if (onStatus) onStatus(true); };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (onMessage) onMessage(msg);
    };
    ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
    ws.onclose = () => { if (onStatus) onStatus(false); if (!stopped) schedule(); };
  };

  connect();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    if (ws) { try { ws.close(); } catch { /* ignore */ } }
  };
}

// Back-compat helper: stream only finalized runs. onRun(row) fires per run frame.
export function openRunsStream(onRun) {
  return openPanelStream({
    onMessage: (msg) => { if (msg && msg.type === "run" && msg.run) onRun(msg.run); },
  });
}
