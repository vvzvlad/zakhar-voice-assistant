// Pure helpers for run toast notifications (kept free of React for testing).
import { statusMeta } from "./runsModel.js";

// Pages that already render live runs — no toasts while the user is there.
export const QUIET_PAGES = new Set(["dashboard", "log"]);

// Every run frame notifies: live in-progress snapshots update an existing
// toast in place (upserted by key), and the finalized frame replaces it.
export function shouldNotify(run, activeId) {
  if (!run) return false;
  return !QUIET_PAGES.has(activeId);
}

// Build the toast view-model for a run row (live snapshot or finalized).
export function toastFromRun(run) {
  // Same identity scheme as runsModel.mapRun: live rows are upserted per device
  // ("live:<device>"), finalized rows are keyed by their DB id. Unlike mapRun,
  // device is normalized to "—" here, so the live key and the finalized drop key
  // ("live:" + toast.device in pushToast / Toasts.jsx) always agree.
  const device = run.device || "—";
  const live = !!run.live;
  const key = live ? "live:" + device : run.id;
  const meta = statusMeta(run);
  return {
    key,
    id: run.id ?? null, // null while live — no deep-link target yet
    live,
    device,
    label: meta.label,
    tone: meta.tone,
    text: run.stt_text || (live ? "(in progress)" : "(silence)"),
    totalMs: run.t_total ?? null,
  };
}

// Prepend a toast, dropping any duplicate key; cap the visible stack.
// A finalized toast also drops the live toast it supersedes for the same
// device, mirroring runsModel.applyStreamedRun.
export function pushToast(list, toast, cap = 4) {
  const drop = new Set([toast.key]);
  if (!toast.live && toast.device != null) drop.add("live:" + toast.device);
  const rest = list.filter((t) => !drop.has(t.key));
  return [toast, ...rest].slice(0, cap);
}
