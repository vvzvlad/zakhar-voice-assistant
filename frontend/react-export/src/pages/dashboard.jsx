import React, { useState } from "react";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { PageHeader } from "../components/primitives.jsx";
import { useAppData } from "../appData.jsx";
import { STAGES, STAGE_COLOR } from "../stageMeta.js";

const SC = STAGE_COLOR;

// Pick a short, human "detail" line for a stage from its selected provider values.
function detailFor(stage, catalog, config) {
  if (stage.key === "vad") {
    const v = config?.core?.vad;
    return v ? `silence ${v.silence_ms} ms` : "—";
  }
  const cat = catalog.categories.find((c) => c.id === stage.cat);
  if (!cat) return "—";
  const prov = cat.providers.find((p) => p.id === cat.selected);
  const v = prov?.values || {};
  if (stage.cat === "stt") return v.model || prov?.label || cat.selected;
  if (stage.cat === "llm") return v.model || prov?.label || cat.selected;
  if (stage.cat === "tts") return v.voice ? `voice · ${v.voice}` : (prov?.label || cat.selected);
  return prov?.label || cat.selected;
}
function providerOf(stage, catalog, config) {
  if (stage.key === "vad") return "WebRTC";
  const cat = catalog.categories.find((c) => c.id === stage.cat);
  return cat ? cat.selected : "—";
}

function Dashboard() {
  const { catalog, config, system, pendingRestart, restart, refreshSystem } = useAppData();
  const [busy, setBusy] = useState(false);

  const doRestart = async () => {
    setBusy(true);
    try { await restart(); } catch { /* ignore */ }
    setBusy(false);
    setTimeout(refreshSystem, 1500);
  };

  return <div className="z-page">
    <PageHeader title="Pipeline overview" desc="Live voice loop across all stages. Click a stage to configure it." />

    {pendingRestart && <div className="z-banner warn">
      <Ic n="restart" w={15} />
      <span><b>Restart required.</b> Some staged changes apply only after the service restarts.</span>
      <span className="act"><button className="z-btn warn sm" disabled={busy} onClick={doRestart}>{busy ? "Restarting…" : "Restart now"}</button></span>
    </div>}

    <div className="z-sl">Pipeline service map<div className="ln" /></div>
    <div className="z-card"><div className="z-map">
      {STAGES.map((s, i) => (
        <React.Fragment key={s.key}>
          <div className="z-svc" onClick={() => nav(s.key)}>
            <div className="z-svc-h"><span className="z-dot ok" /><b>{s.name}</b><span className="prov">{providerOf(s, catalog, config)}</span></div>
            <div className="mdl">{detailFor(s, catalog, config)}</div>
            <div className="lat"><b style={{ color: SC[s.key] }}>{s.role}</b></div>
            {s.key === "llm" && config?.core?.mcp?.url
              ? <span className="z-mcpchip" onClick={(e) => { e.stopPropagation(); nav("mcp"); }}>◆ MCP</span>
              : <span style={{ height: 16 }} />}
            <span className="cfg">Configure <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M2 6h7M6 3l3 3-3 3" /></svg></span>
          </div>
          {i < STAGES.length - 1 && <div className="z-arrow"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 8h11M9 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" /></svg></div>}
        </React.Fragment>
      ))}
    </div></div>

    <div className="z-sl">Метрики<div className="ln" /></div>
    <div className="z-card"><div className="z-empty">
      <div className="ic"><Ic n="dashboard" w={20} /></div>
      <b>Метрики появятся позже</b>
      Нет эндпоинтов <span className="z-paramtag">/runs</span> / <span className="z-paramtag">/metrics</span> — KPI и недавние запросы будут добавлены после реализации хранилища прогонов.
    </div></div>
  </div>;
}

export default Dashboard;
