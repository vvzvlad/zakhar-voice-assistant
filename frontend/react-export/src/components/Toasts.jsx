import React, { useEffect, useRef, useState } from "react";
import { useAppData } from "../appData.jsx";
import { nav } from "../navStore.js";
import { fmtSec } from "../runsModel.js";
import { shouldNotify, toastFromRun, pushToast } from "../toastModel.js";

const TOAST_TTL_MS = 6000; // auto-dismiss delay per toast

// Popup notifications for finalized runs, shown only while the user is away
// from the pages that already display live runs (Dashboard / Request Log).
// Clicking a toast deep-links to that run in the Log page.
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
      // (Re)arm this toast's auto-dismiss timer.
      const timers = timersRef.current;
      if (timers.has(toast.id)) clearTimeout(timers.get(toast.id));
      timers.set(toast.id, setTimeout(() => {
        timers.delete(toast.id);
        setToasts((prev) => prev.filter((t) => t.id !== toast.id));
      }, TOAST_TTL_MS));
    });
    return () => {
      unsub();
      for (const t of timersRef.current.values()) clearTimeout(t);
      timersRef.current.clear();
    };
  }, [subscribeRuns]);

  const dismiss = (id) => {
    const timers = timersRef.current;
    if (timers.has(id)) { clearTimeout(timers.get(id)); timers.delete(id); }
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const open = (id) => {
    try { localStorage.setItem("z-openreq", String(id)); } catch { /* ignore */ }
    dismiss(id);
    nav("log");
  };

  if (!toasts.length) return null;
  return <div className="z-toasts">
    {toasts.map((t) => (
      <div key={t.id} className="z-toast" role="status" onClick={() => open(t.id)}>
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
          onClick={(e) => { e.stopPropagation(); dismiss(t.id); }}>×</button>
      </div>
    ))}
  </div>;
}
