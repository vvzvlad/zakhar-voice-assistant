// Pipeline overview. KPIs + per-stage and per-chain stats from /api/metrics, the live
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
import { chainsOf, chainRows, ttsVariantsOf } from "../chains.js";

const SC = STAGE_COLOR;

// The provider a stage node shows: tts speaks with its first enabled variant,
// the other categories with their selected provider.
function activeId(stage, cat, catalog) {
  if (stage.cat === "tts") return ttsVariantsOf(catalog).find((v) => v.enabled)?.provider;
  return cat.selected;
}

// Pick a short, human "detail" line for a stage from its active provider values.
// Exported for unit tests.
export function detailFor(stage, catalog, config) {
  if (stage.key === "vad") {
    const v = config?.core?.vad;
    return v ? `silence ${v.silence_ms} ms` : "—";
  }
  const cat = catalog.categories.find((c) => c.id === stage.cat);
  if (!cat) return "—";
  const id = activeId(stage, cat, catalog);
  if (!id) return "—";
  const prov = cat.providers.find((p) => p.id === id);
  const v = prov?.values || {};
  if (stage.cat === "tts") {
    const fallbacks = ttsVariantsOf(catalog).filter((x) => x.enabled).length - 1;
    const main = v.voice ? `voice · ${v.voice}` : (prov?.label || id);
    return fallbacks > 0 ? `${main} · +${fallbacks} fallback` : main;
  }
  return prov?.label || id;
}
// Exported for unit tests. Two-argument signature: (stage, catalog).
export function providerOf(stage, catalog) {
  const cat = catalog.categories.find((c) => c.id === stage.cat);
  if (!cat) return "—";
  const id = activeId(stage, cat, catalog);
  // vad shows the human provider label ("WebRTC VAD"); the others keep the id.
  if (stage.key === "vad") {
    const prov = cat.providers.find((p) => p.id === id);
    return prov?.label || id || "—";
  }
  return id || "—";
}

// Service-map order: the per-category stages around one Chains block that
// stands for STT and the handler (which pair runs is set per chain).
const MAP_NODES = ["vad", "wakeword", "chains", "stress", "tts"];

const fmtShare = (share) => (share == null ? "—" : `${Math.round(share * 100)}%`);

function Configure() {
  return <span className="cfg">Configure <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M2 6h7M6 3l3 3-3 3" /></svg></span>;
}

function StageNode({ s, avg, loading, catalog, config }) {
  return <div className="z-svc" onClick={() => nav(s.key)}>
    <div className="z-svc-h"><span className={"z-dot " + (loading ? "loading" : "ok")} /><b>{s.name}</b><span className="prov">{providerOf(s, catalog)}</span></div>
    <div className="mdl">{loading ? <span className="z-loading-inline"><span className="z-spin" />loading…</span> : detailFor(s, catalog, config)}</div>
    <div className="lat">{avg != null
      ? <><b style={{ color: SC[s.key] }}>{(avg / 1000).toFixed(2)}</b><s>s avg</s></>
      : <span style={{ fontSize: 12, color: "var(--mut)" }}>{s.role}</span>}</div>
    <span style={{ height: 16 }} />
    <Configure />
  </div>;
}

function McpChip({ config }) {
  return config?.core?.mcp_servers?.length > 0
    ? <span className="z-mcpchip" onClick={(e) => { e.stopPropagation(); nav("mcp"); }}>◆ MCP</span>
    : <span style={{ height: 16 }} />;
}

// The Chains block: one row per enabled chain with "stt → handler", its share
// of the served requests and its average time.
function ChainsNode({ rows, loading, config }) {
  return <div className="z-svc" style={{ flex: 2 }} onClick={() => nav("chains")}>
    <div className="z-svc-h"><span className={"z-dot " + (loading ? "loading" : "ok")} /><b>Chains</b><span className="prov">{rows.length} enabled</span></div>
    {loading && <div className="mdl"><span className="z-loading-inline"><span className="z-spin" />loading…</span></div>}
    {rows.length === 0 && <div className="mdl">no enabled chain</div>}
    {rows.map((r) => <div key={r.name} className="mdl" data-testid="chain-row" title={r.name} style={{ display: "flex", gap: 8 }}>
      <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{r.stt} → {r.handler}</span>
      <span style={{ color: "var(--mut)" }}>{fmtShare(r.share)}</span>
      <span style={{ color: SC.llm }}>{fmtSec(r.avg_ms)}</span>
    </div>)}
    <McpChip config={config} />
    <Configure />
  </div>;
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
      {MAP_NODES.map((key, i) => {
        const s = STAGES.find((x) => x.key === key);
        return <React.Fragment key={key}>
          {s
            ? <StageNode s={s} avg={stageAvg[s.key]} loading={reloading.has(s.cat)} catalog={catalog} config={config} />
            : <ChainsNode rows={chainRows(chainsOf(config), m.per_chain)} config={config}
                loading={reloading.has("stt") || reloading.has("llm")} />}
          {i < MAP_NODES.length - 1 && <div className="z-arrow"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 8h11M9 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" /></svg></div>}
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
