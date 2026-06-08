import React from "react";
import { Ic } from "../components/icons.jsx";
import { PageHeader } from "../components/primitives.jsx";

// Placeholder: there is no run/metrics storage in the backend yet.
function Log() {
  return <div className="z-page">
    <PageHeader title="Request log" desc="Every pipeline run with per-stage timings, tool calls and audio." />
    <div className="z-card"><div className="z-empty">
      <div className="ic"><Ic n="log" w={20} /></div>
      <b>Журнал запросов — скоро</b>
      Появится после реализации хранилища прогонов (шаг 4). Пока что бэкенд не сохраняет историю запросов.
    </div></div>
  </div>;
}

export default Log;
