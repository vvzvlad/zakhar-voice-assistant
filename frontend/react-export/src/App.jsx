import React, { useState, useEffect } from "react";
import { TITLES } from "./nav.js";
import { setNavigate } from "./navStore.js";
import { Sidebar } from "./components/Sidebar.jsx";
import { Topbar } from "./components/Topbar.jsx";
import { PAGES } from "./pages/index.js";

export default function App() {
  const [active, setActive] = useState(() => {
    try { return localStorage.getItem("z-active") || "dashboard"; } catch { return "dashboard"; }
  });
  const navigate = (id) => {
    setActive(id);
    try { localStorage.setItem("z-active", id); } catch { /* ignore */ }
    const sc = document.querySelector(".z-scroll");
    if (sc) sc.scrollTop = 0;
  };
  // Register the navigate fn so pages can call nav(id) from the module-level store.
  useEffect(() => { setNavigate(navigate); });

  const Page = PAGES[active] || (() => (
    <div className="z-page"><div className="z-empty"><b>{TITLES[active]}</b>Section coming up.</div></div>
  ));

  return (
    <div className="z-app">
      <Sidebar active={active} onNav={navigate} />
      <div className="z-main">
        <Topbar active={active} />
        <div className="z-scroll"><Page /></div>
      </div>
    </div>
  );
}
