// Adapts backend run rows (snake_case, epoch ts, flat timings) into the UI shape
// the Log/Dashboard components expect, plus the presentation consts they read.
//
// Backend row (list)  : { id, ts, device, result, reason, stt_text, llm_text,
//                         tokens, t_vad, t_stt, t_llm, t_stress, t_tts, t_total }
// Backend row (detail): the above PLUS model, rounds[], audio_*, error_*.

import { total } from "./components/primitives.jsx";

// result -> { label, tone }; tone drives the pill/dot color (good|muted|bad).
export const RESULT_META = {
  ok:    { label: "OK",        tone: "good" },
  tool:  { label: "OK · tool", tone: "good" },
  empty: { label: "Empty",     tone: "muted" },
  error: { label: "Error",     tone: "bad" },
};

// Per-stage accent colors for the waterfall / gantt segments.
export const STAGE_COLOR = {
  vad:      "#64748b",
  stt:      "#0891b2",
  llm:      "#4f46e5",
  stress:   "#9333ea",
  tts:      "#0d9488",
};

// ms -> "1.23s". Null/undefined timings render as an em dash.
export function fmtSec(ms) {
  if (ms == null || isNaN(ms)) return "—";
  return (ms / 1000).toFixed(2) + "s";
}

// Authoritative total run time: prefer the backend t_total, fall back to summing r.t.
export const totalMs = (r) => (r.t_total != null ? r.t_total : total(r.t));

// epoch seconds (float) -> "HH:MM:SS" in local time.
function fmtTime(ts) {
  if (ts == null || isNaN(ts)) return "—";
  const d = new Date(ts * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

// Map a backend row to the UI run object. `t` groups per-stage timings (ms);
// 0/null are coerced to 0 so the waterfall's `> 0` filter drops empty stages
// (stress shows only when the stage actually ran, i.e. t_stress > 0).
// `audio`/`error` are null unless the detail payload carries them.
export function mapRun(row) {
  if (!row) return row;
  // Live (in-progress) rows have no DB id yet; they are upserted by device, so
  // they are keyed "live:<device>". Finalized rows are keyed by their real id.
  const live = !!row.live;
  const key = live ? "live:" + (row.device ?? "") : row.id;
  return {
    ...row,
    key,
    live,
    time: fmtTime(row.ts),
    stt: row.stt_text,
    llm: row.llm_text,
    stress: row.stress_text,
    t: {
      vad: row.t_vad || 0,
      stt: row.t_stt || 0,
      llm: row.t_llm || 0,
      stress: row.t_stress || 0,
      tts: row.t_tts || 0,
    },
    audio: row.audio_bytes
      ? { ms: row.audio_ms, bytes: row.audio_bytes, fmt: row.audio_fmt }
      : null,
    error: row.error_stage
      ? { stage: row.error_stage, text: row.error_text }
      : null,
  };
}

// Status pill meta for a run row: an in-progress (live) row shows "Running";
// otherwise it maps result -> RESULT_META (falling back to a muted label).
export function statusMeta(r) {
  if (r && r.live) return { label: "Running", tone: "muted" };
  return RESULT_META[r && r.result] || { label: r && r.result, tone: "muted" };
}

// Apply a streamed run (live partial or finalized) to the current list.
// Rows are keyed by `mapped.key` (real id when finalized, "live:<device>" while
// live). A finalized run ALWAYS removes the live row it supersedes (same device),
// even when the finalized row itself is filtered out, so a live row is never
// stranded; it is (re)inserted only when `match` is true. A live partial is
// inserted only when `match` is true. Newest first, capped at `cap`.
export function applyStreamedRun(prev, mapped, match, cap = 100) {
  const drop = new Set([mapped.key]);
  if (!mapped.live && mapped.device != null) drop.add("live:" + mapped.device);
  const rest = prev.filter((r) => !drop.has(r.key));
  if (mapped.live) return match ? [mapped, ...rest].slice(0, cap) : prev;
  return match ? [mapped, ...rest].slice(0, cap) : rest;
}

// Compact page list for numbered pagination: always show first + last + a window
// around the current page, collapsing gaps to "…". Returns a mix of page numbers
// and "…" strings (the ellipsis is non-clickable in the UI).
export function pageWindow(current, totalPages) {
  if (totalPages <= 7) return Array.from({ length: Math.max(totalPages, 0) }, (_, i) => i + 1);
  const wanted = new Set([1, totalPages, current, current - 1, current + 1]);
  const pages = [...wanted].filter((p) => p >= 1 && p <= totalPages).sort((a, b) => a - b);
  const out = [];
  let prev = 0;
  for (const p of pages) {
    if (p - prev > 1) out.push("…");
    out.push(p);
    prev = p;
  }
  return out;
}
