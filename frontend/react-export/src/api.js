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
  if (body !== undefined) {
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
export const getOptions = (category, plugin, field) =>
  request(`/api/options?category=${encodeURIComponent(category)}&plugin=${encodeURIComponent(plugin)}&field=${encodeURIComponent(field)}`);
export const getPrompt = () => request("/api/prompt");
export const putPrompt = (text) => request("/api/prompt", { method: "PUT", body: { text } });
export const getSystem = () => request("/api/system");
export const getDevices = () => request("/api/devices");

// Background capture: the recording runs as a server-side task decoupled from the
// browser request, so closing the browser no longer cancels it (which used to
// reboot the device). start kicks it off (202 -> initial status), getCaptureStatus
// polls for the live countdown, and downloadCaptureResult fetches the finished WAV.
export const startCapture = (device, seconds) =>
  request("/api/capture", { method: "POST", body: { device, seconds } });
export const getCaptureStatus = (device) =>
  request(`/api/capture?device=${encodeURIComponent(device)}`);

// Download the already-recorded WAV held server-side (one-shot; consumed on read)
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
// Live tool sources (external MCP + built-ins) with their advertised tools.
export const getTools = () => request("/api/tools");

// --- observability (run log + metrics) -------------------------------------
// Builds /api/runs?device=&result=&search=&limit= from the given params,
// dropping empty/undefined ones so the backend sees a clean query string.
export const getRuns = (params = {}) => {
  const q = new URLSearchParams();
  for (const k of ["device", "result", "search", "limit"]) {
    const v = params[k];
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const qs = q.toString();
  return request("/api/runs" + (qs ? "?" + qs : ""));
};
export const getRun = (id) => request(`/api/runs/${encodeURIComponent(id)}`);

// Absolute URL of a run's stored utterance WAV (for an <audio> src). BASE is "" in
// prod (same origin) or an absolute base in dev.
export const runAudioUrl = (id) => `${BASE}/api/runs/${encodeURIComponent(id)}/audio`;

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
export const getMetrics = () => request("/api/metrics");

// --- live run stream (WebSocket) -------------------------------------------
// Opens a WebSocket to /api/runs/stream and invokes onRun(row) for every run the
// backend pushes (row is the same snake_case shape as a /api/runs list item).
// Auto-reconnects with capped exponential backoff. Returns a stop() that closes
// the socket and cancels any pending reconnect.
export function openRunsStream(onRun) {
  let ws = null;
  let stopped = false;
  let attempt = 0;
  let timer = null;

  const wsUrl = () => {
    // BASE is "" in prod (same origin) or an absolute http(s) base in dev.
    const u = new URL("/api/runs/stream", BASE || window.location.origin);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.toString();
  };

  const schedule = () => {
    if (stopped) return;
    // Backoff on the CURRENT attempt count (fast first retry ~0.5s), then bump it
    // for the next one. Capped at 30s; onopen resets attempt to 0 on success.
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
    ws.onopen = () => { attempt = 0; };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg && msg.type === "run" && msg.run) onRun(msg.run);
    };
    ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
    ws.onclose = () => { if (!stopped) schedule(); };
  };

  connect();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    if (ws) { try { ws.close(); } catch { /* ignore */ } }
  };
}
