import React, { useEffect, useRef, useState } from "react";
import { useAppData } from "../appData.jsx";
import { nav } from "../navStore.js";
import { fmtSec } from "../runsModel.js";
import { shouldNotify, toastFromRun, pushToast } from "../toastModel.js";

const TOAST_TTL_MS = 6000; // auto-dismiss delay per toast

// Popup notifications for pipeline runs: live in-progress snapshots are shown
// too, upserted per device, and the finalized frame replaces the live toast.
// Suppressed while the user is on the pages that already display live runs
// (Dashboard / Request Log). Clicking a toast deep-links to that run in the
// Log page (finalized runs only — live ones have no id yet).
export function Toasts({ active }) {
  const { subscribeRuns } = useAppData();
  const [toasts, setToasts] = useState([]);
  // The subscription is opened once; read the current page through a ref so
  // the listener never holds a stale `active`.
  const activeRef = useRef(active);
  useEffect(() => { activeRef.current = active; }, [active]);
  const timersRef = useRef(new Map());

  useEffect(() => {
    const unsub = subscribeRuns((run) => {
      if (!shouldNotify(run, activeRef.current)) return;
      const toast = toastFromRun(run);
      setToasts((prev) => pushToast(prev, toast));
      const timers = timersRef.current;
      // A finalized toast supersedes the device's live toast — retire its timer too.
      if (!toast.live) {
        const liveKey = "live:" + toast.device;
        if (timers.has(liveKey)) { clearTimeout(timers.get(liveKey)); timers.delete(liveKey); }
      }
      // (Re)arm this key's auto-dismiss timer on every frame, so a live toast
      // survives while frames keep arriving and expires after the last one.
      if (timers.has(toast.key)) clearTimeout(timers.get(toast.key));
      timers.set(toast.key, setTimeout(() => {
        timers.delete(toast.key);
        setToasts((prev) => prev.filter((t) => t.key !== toast.key));
      }, TOAST_TTL_MS));
    });
    return () => {
      unsub();
      for (const t of timersRef.current.values()) clearTimeout(t);
      timersRef.current.clear();
    };
  }, [subscribeRuns]);

  const dismiss = (key) => {
    const timers = timersRef.current;
    if (timers.has(key)) { clearTimeout(timers.get(key)); timers.delete(key); }
    setToasts((prev) => prev.filter((t) => t.key !== key));
  };

  const open = (t) => {
    // Live toasts have no DB id yet, so there is nothing to deep-link to —
    // just go to the Log page where the live row is visible.
    if (t.id != null) {
      try { localStorage.setItem("z-openreq", String(t.id)); } catch { /* ignore */ }
    }
    dismiss(t.key);
    nav("log");
  };

  if (!toasts.length) return null;
  return <div className="z-toasts">
    {toasts.map((t) => (
      <div key={t.key} className="z-toast" role="status" onClick={() => open(t)}>
        <span className={"z-dot " + (t.tone === "good" ? "ok" : t.tone === "bad" ? "error" : "off")} />
        <div className="z-toast-body">
          <div className="z-toast-head">
            <b>{t.device}</b>
            <span className={"z-st " + t.tone}>{t.label}</span>
            {t.totalMs != null && <span className="z-toast-time">{fmtSec(t.totalMs)}</span>}
          </div>
          <div className="z-toast-text">{t.text}</div>
        </div>
        <button className="z-toast-x" aria-label="Dismiss"
          onClick={(e) => { e.stopPropagation(); dismiss(t.key); }}>×</button>
      </div>
    ))}
  </div>;
}
