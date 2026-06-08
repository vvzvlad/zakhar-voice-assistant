// Module-level navigate store. App.jsx calls setNavigate() once on mount;
// pages import { nav } and call nav(id) to switch sections.
let _navigate = () => {};
export const setNavigate = (fn) => { _navigate = fn; };
export const nav = (id) => _navigate(id);
