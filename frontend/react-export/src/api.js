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
    throw new ApiError("Не удалось связаться с сервером: " + e.message, { status: 0 });
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
export const postRestart = () => request("/api/restart", { method: "POST" });
export const getDevices = () => request("/api/devices");
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
export const getMetrics = () => request("/api/metrics");
