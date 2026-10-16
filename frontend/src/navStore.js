// Module-level navigate store. App.jsx calls setNavigate() once on mount;
// pages import { nav } and call nav(id) to switch sections.
import { useEffect } from "react";

let _navigate = () => {};
export const setNavigate = (fn) => { _navigate = fn; };

// Unsaved changes. A page holding unsaved edits registers itself with
// useLeaveGuard(); every in-app navigation asks confirmLeave() first, and closing
// or reloading the browser tab gets the browser's own "leave site?" prompt.
const _dirty = new Set();
export const confirmDiscard = (dirty) => !dirty || window.confirm("Discard unsaved changes?");
export const confirmLeave = () => confirmDiscard(_dirty.size > 0);
export function useLeaveGuard(dirty) {
  useEffect(() => {
    if (!dirty) return undefined;
    const token = {};
    const onUnload = (e) => { e.preventDefault(); e.returnValue = ""; };
    _dirty.add(token);
    window.addEventListener("beforeunload", onUnload);
    return () => { _dirty.delete(token); window.removeEventListener("beforeunload", onUnload); };
  }, [dirty]);
}

export const nav = (id) => { if (confirmLeave()) _navigate(id); };

// Provider tab requested by a cross-page "configure" link. navTab() records it
// and navigates; the target page reads it with pendingTab() when it mounts and
// drops it with clearTab() afterwards, so a later plain visit opens the default
// tab. Reading and clearing are separate so a StrictMode double-run of the state
// initializer still sees the request. Nothing is recorded when the user keeps
// unsaved changes and stays.
const _tabs = {};
export const navTab = (id, tab) => {
  if (!confirmLeave()) return;
  _tabs[id] = tab;
  _navigate(id);
};
export const pendingTab = (id) => _tabs[id];
export const clearTab = (id) => { delete _tabs[id]; };
