// Module-level navigate store. App.jsx calls setNavigate() once on mount;
// pages import { nav } and call nav(id) to switch sections.
let _navigate = () => {};
export const setNavigate = (fn) => { _navigate = fn; };
export const nav = (id) => _navigate(id);

// Provider tab requested by a cross-page "configure" link. navTab() records it
// and navigates; the target page reads it with pendingTab() when it mounts and
// drops it with clearTab() afterwards, so a later plain visit opens the default
// tab. Reading and clearing are separate so a StrictMode double-run of the state
// initializer still sees the request.
const _tabs = {};
export const navTab = (id, tab) => { _tabs[id] = tab; _navigate(id); };
export const pendingTab = (id) => _tabs[id];
export const clearTab = (id) => { delete _tabs[id]; };
