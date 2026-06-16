// Pipeline overview. KPIs + per-stage avg latency from /api/metrics, the live
// service map from the catalog/config, and a recent-requests table from
// /api/runs (rows deep-link into the Log page).
import React, { useState, useEffect } from "react";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { PageHeader, Waterfall, Loading } from "../components/primitives.jsx";
import { useAppData } from "../appData.jsx";
import { STAGES } from "../stageMeta.js";
import { getMetrics, getRuns, openRunsStream } from "../api.js";
import { STAGE_COLOR, fmtSec, mapRun, totalMs, statusMeta, applyStreamedRun } from "../runsModel.js";

const SC = STAGE_COLOR;

// Pick a short, human "detail" line for a stage from its selected provider values.
// Exported for unit tests.
export function detailFor(stage, catalog, config) {
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
// Exported for unit tests. Two-argument signature: (stage, catalog).
export function providerOf(stage, catalog) {
  const cat = catalog.categories.find((c) => c.id === stage.cat);
  if (!cat) return "—";
  // vad shows the human provider label ("WebRTC VAD"); stt/llm/tts keep the id.
  if (stage.key === "vad") {
    const prov = cat.providers.find((p) => p.id === cat.selected);
    return prov?.label || cat.selected || "—";
  }
  return cat.selected || "—";
}

function Dashboard() {
  const { catalog, config, system } = useAppData();

  const [metrics, setMetrics] = useState(null);
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    Promise.all([getMetrics(), getRuns({ limit: 5 })])
      .then(([m, r]) => {
        if (!alive) return;
        setMetrics(m);
        setRecent((r.runs || []).map(mapRun));
      })
      .catch(() => { if (alive) { setMetrics(null); setRecent([]); } })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  // Live updates: prepend each pushed run (dedupe + cap 5) and refresh KPIs.
  useEffect(() => {
    const stop = openRunsStream((row) => {
      const mapped = mapRun(row);
      setRecent((prev) => applyStreamedRun(prev, mapped, true, 5));
      if (!mapped.live) getMetrics().then((m) => setMetrics(m)).catch(() => { /* keep last good */ });
    });
    return stop;
  }, []);

  const m = metrics || {};
  const stageAvg = m.per_stage_avg_ms || {};
  const kpis = [
    { k: "Requests · 24h", v: m.requests_24h != null ? m.requests_24h : 0 },
    { k: "p50 latency", v: fmtSec(m.p50_ms) },
    { k: "p95 latency", v: fmtSec(m.p95_ms) },
    { k: "Error rate", v: m.error_rate != null ? (m.error_rate * 100).toFixed(1) + "%" : "—" },
    { k: "Rejected · 24h", v: m.rejected_24h != null ? m.rejected_24h : 0 },
  ];

  // Backend categories whose hot-reload is in flight (from the system heartbeat).
  const reloading = new Set(system?.reloading || []);

  return <div className="z-page">
    <PageHeader title="Pipeline overview" desc="Live voice loop across all stages. Click a stage to configure it." />

    <div className="z-kpis">
      {kpis.map((c, i) => <div className="z-kpi" key={i}>
        <span className="k">{c.k}</span>
        <div className="v">{c.v}</div>
      </div>)}
    </div>

    <div className="z-sl">Pipeline service map<div className="ln" /></div>
    <div className="z-card"><div className="z-map">
      {STAGES.map((s, i) => {
        const avg = stageAvg[s.key];
        const isLoading = reloading.has(s.cat);
        return <React.Fragment key={s.key}>
          <div className="z-svc" onClick={() => nav(s.key)}>
            <div className="z-svc-h"><span className={"z-dot " + (isLoading ? "loading" : "ok")} /><b>{s.name}</b><span className="prov">{providerOf(s, catalog)}</span></div>
            <div className="mdl">{isLoading ? <span className="z-loading-inline"><span className="z-spin" />loading…</span> : detailFor(s, catalog, config)}</div>
            <div className="lat">{avg != null
              ? <><b style={{ color: SC[s.key] }}>{(avg / 1000).toFixed(2)}</b><s>s avg</s></>
              : <span style={{ fontSize: 12, color: "var(--mut)" }}>{s.role}</span>}</div>
            {s.key === "llm" && (config?.core?.mcp_servers?.length > 0)
              ? <span className="z-mcpchip" onClick={(e) => { e.stopPropagation(); nav("mcp"); }}>◆ MCP</span>
              : <span style={{ height: 16 }} />}
            <span className="cfg">Configure <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M2 6h7M6 3l3 3-3 3" /></svg></span>
          </div>
          {i < STAGES.length - 1 && <div className="z-arrow"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 8h11M9 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" /></svg></div>}
        </React.Fragment>;
      })}
    </div></div>

    <div className="z-sl">Recent requests<div className="ln" /><a onClick={() => nav("log")}>View full log →</a></div>
    <div className="z-card">
      {loading ? <Loading />
        : recent.length === 0 ? <div className="z-empty"><div className="ic"><Ic n="log" w={20} /></div><b>No runs</b>Recent requests will appear here after the first one is processed.</div>
          : <>
            <div className="z-tblwrap">
              <table className="z-tbl">
                <thead><tr><th>Time</th><th>Device</th><th>Recognized</th><th>Response</th><th style={{ textAlign: "right" }}>Σ</th><th>Stage waterfall</th><th>Status</th></tr></thead>
                <tbody>
                  {recent.map((r) => {
                    const rm = statusMeta(r);
                    return <tr key={r.key} onClick={() => { if (r.id == null) return; try { localStorage.setItem("z-openreq", String(r.id)); } catch { /* ignore */ } nav("log"); }}>
                      <td className="tm">{r.time}</td>
                      <td style={{ fontWeight: 600 }}>{r.device}</td>
                      <td><div className={"z-tx" + (r.stt ? "" : " mut")}>{r.stt || (r.live ? "…" : "(silence)")}</div></td>
                      <td><div className={"z-tx wide" + (r.llm ? "" : " mut")}>{r.llm || (r.live ? "…" : "—")}</div></td>
                      <td className="num" style={{ fontWeight: 600 }}>{fmtSec(totalMs(r))}</td>
                      <td><Waterfall r={r} /></td>
                      <td><span className={"z-st " + rm.tone}><span className={"z-dot " + (rm.tone === "good" ? "ok" : rm.tone === "bad" ? "error" : "off")} />{rm.label}</span></td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>
            <div className="z-tfoot">Showing {recent.length} recent runs</div>
          </>}
    </div>
  </div>;
}

export default Dashboard;
