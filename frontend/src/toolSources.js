// The tool sources the Tools page lists, shared with the menu's counter: the external
// MCP servers (core.mcp_servers) and the built-in sources below.

// The built-in sources; `id` is both the core.* section and the source id in
// /api/tools, `fallback` the values shown before the section exists.
export const BUILTIN_SOURCES = [
  { id: "openweathermap", name: "OpenWeatherMap",
    fallback: { enabled: true, api_key: "", city: "Moscow" }, configured: (v) => !!(v && v.api_key) },
  { id: "calendar", name: "Calendar", sub: "CalDAV",
    fallback: { enabled: true, url: "", username: "", password: "", calendar: "" }, configured: (v) => !!(v && v.url && v.username) },
  { id: "reminders", name: "Reminders",
    fallback: { enabled: true }, configured: (v) => !!(v && v.enabled) },
  { id: "assistant_tasks", name: "Assistant tasks", sub: "ZeroClaw webhook",
    fallback: { enabled: true, url: "", secret: "", task_suffix: "", prompt: "" }, configured: (v) => !!(v && v.url && v.secret) },
];

// How many of the listed sources are switched on in `core`; a missing flag is on.
export const enabledToolSources = (core = {}) =>
  (core.mcp_servers || []).filter((s) => s.enabled ?? true).length
  + BUILTIN_SOURCES.filter((s) => (core[s.id] || s.fallback).enabled ?? true).length;
