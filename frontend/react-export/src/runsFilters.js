// Client-side mirror of the backend /api/runs filters, extracted from
// pages/log.jsx (R-FE-1, behaviour-preserving) so it can be unit-tested.
// Used to decide whether a live-pushed run matches the current UI filters.
// device: exact; result: all|errors|ok; search: case-insensitive substring
// over recognized/response text.
//
// Two known, intentional points to keep in sync with the backend (src/runs_store.py list()):
// 1. The UI's `result` filter domain is only "all" | "errors" | "ok" (the chip cycles
//    those three). The backend list() also supports an exact-match branch for any other
//    value, but the UI never sends one, so this client mirror covers exactly the UI's domain.
// 2. Search here is case-insensitive via toLowerCase(), which lowercases Cyrillic too;
//    the backend uses SQLite LIKE, which is case-insensitive only for ASCII (Cyrillic is
//    case-sensitive). For an uppercase-Cyrillic search term a live-pushed row may therefore
//    appear that a fresh server fetch would not return; the next manual reload/filter-change
//    re-runs the server query and reconciles. This is a known, accepted minor divergence.
export function matchesFilters(row, { result, search, device }) {
  if (device.trim() && row.device !== device.trim()) return false;
  if (result === "errors" && row.result !== "error") return false;
  if (result === "ok" && !(row.result === "ok" || row.result === "tool")) return false;
  const s = search.trim().toLowerCase();
  if (s) {
    const hay = `${row.stt_text || ""}\n${row.llm_text || ""}`.toLowerCase();
    if (!hay.includes(s)) return false;
  }
  return true;
}
