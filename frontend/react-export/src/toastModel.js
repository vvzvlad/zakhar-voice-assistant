// Pure helpers for run toast notifications (kept free of React for testing).
import { RESULT_META } from "./runsModel.js";

// Pages that already render live runs — no toasts while the user is there.
export const QUIET_PAGES = new Set(["dashboard", "log"]);

// A finalized run frame is the single notification trigger: live in-progress
// snapshots (no id yet) would fire several times per run.
export function shouldNotify(run, activeId) {
  if (!run || run.live || run.id == null) return false;
  return !QUIET_PAGES.has(activeId);
}

// Build the toast view-model for a finalized run row.
export function toastFromRun(run) {
  const meta = RESULT_META[run.result] || { label: run.result || "—", tone: "muted" };
  return {
    id: run.id,
    device: run.device || "—",
    label: meta.label,
    tone: meta.tone,
    text: run.stt_text || "(silence)",
    totalMs: run.t_total ?? null,
  };
}

// Prepend a toast, dropping any duplicate id; cap the visible stack.
export function pushToast(list, toast, cap = 4) {
  const rest = list.filter((t) => t.id !== toast.id);
  return [toast, ...rest].slice(0, cap);
}
