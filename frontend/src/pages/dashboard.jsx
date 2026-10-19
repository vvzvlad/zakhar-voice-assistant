// Pipeline overview. KPIs + per-stage and per-chain stats from /api/metrics, the
// Capture → Logic → Speech service map from the catalog/config, and a
// recent-requests table from /api/runs (rows deep-link into the Log page).
import React, { useState, useEffect } from "react";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { PageHeader, Waterfall, Loading } from "../components/primitives.jsx";
import { useAppData } from "../appData.jsx";
import { getMetrics, getRuns, openRunsStream } from "../api.js";
import { STAGE_COLOR, fmtSec, mapRun, totalMs, statusMeta, isAnnouncement, applyStreamedRun } from "../runsModel.js";
import { chainsOf, chainRows, ttsVariantsOf, voiceOf } from "../chains.js";
import { deviceLabel } from "../format.js";

const SC = STAGE_COLOR;

const categoryOf = (catalog, id) => catalog?.categories?.find((c) => c.id === id);

// The selected provider of a single-choice category (vad, wakeword, stress):
// its id, its human label (the id when the catalog has no entry) and its values.
function selectedOf(catalog, cat) {
  const c = categoryOf(catalog, cat);
  const p = c?.providers?.find((x) => x.id === c.selected);
  return { id: c?.selected, label: p?.label || c?.selected, values: p?.values || {} };
}

// What the Speech tile shows: the first enabled TTS variant's provider and voice
// name, and how many enabled variants stand behind it as fallbacks.
function speechOf(catalog) {
  const enabled = ttsVariantsOf(catalog).filter((v) => v.enabled);
  const first = enabled[0];
  const p = first && categoryOf(catalog, "tts")?.providers?.find((x) => x.id === first.provider);
  return { provider: p?.label || first?.provider, voice: voiceOf(p?.values), fallbacks: Math.max(0, enabled.length - 1) };
}

const fmtShare = (share) => (share == null ? "—" : `${Math.round(share * 100)}%`);

function Configure() {
  return <span className="cfg">Configure <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M2 6h7M6 3l3 3-3 3" /></svg></span>;
}

// One service-map tile. Its dot is grey when what it stands for is off (no VAD
// provider, no enabled chain, no enabled TTS variant) and blinks while reloading.
function Tile({ title, page, on, loading, prov, flex, children }) {
  return <div className="z-svc" style={flex ? { flex } : undefined} onClick={() => nav(page)} data-testid={`tile-${page}`}>
    <div className="z-svc-h"><span className={"z-dot " + (loading ? "loading" : on ? "ok" : "off")} /><b>{title}</b><span className="prov">{prov}</span></div>
    {loading && <div className="mdl"><span className="z-loading-inline"><span className="z-spin" />loading…</span></div>}
    {children}
    <Configure />
  </div>;
}

// A switch inside a tile (wake-word verify, stress): grey dot when off.
function Mark({ label, on }) {
  return <div className="mdl" style={{ display: "flex", alignItems: "center", gap: 6 }}>
    <span className={"z-dot " + (on ? "ok" : "off")} />{label} {on ? "on" : "off"}
  </div>;
}

function Avg({ ms, color, role }) {
  return <div className="lat">{ms != null
    ? <><b style={{ color }}>{(ms / 1000).toFixed(2)}</b><s>s avg</s></>
    : <span style={{ fontSize: 12, color: "var(--mut)" }}>{role}</span>}</div>;
}

function McpChip({ config }) {
  return config?.core?.mcp_servers?.length > 0
    ? <span className="z-mcpchip" onClick={(e) => { e.stopPropagation(); nav("tools"); }}>◆ MCP</span>
    : null;
}

// Capture → Logic → Speech. Logic has one row per enabled chain: its name, its
// share of the served requests and its average time.
function ServiceMap({ catalog, config, stageAvg, perChain, reloading }) {
  const vad = selectedOf(catalog, "vad");
  const wake = selectedOf(catalog, "wakeword");
  const stress = selectedOf(catalog, "stress");
  const followup = selectedOf(catalog, "followup");
  const speech = speechOf(catalog);
  const rows = chainRows(chainsOf(config), perChain);
  const busy = (...cats) => cats.some((c) => reloading.has(c));
  const arrow = <div className="z-arrow"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 8h11M9 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" /></svg></div>;
  return <div className="z-card"><div className="z-map">
    <Tile title="Capture" page="capture" on={!!vad.id} loading={busy("vad", "wakeword")} prov={vad.label || "—"}>
      <Mark label="wake-word verify" on={!!wake.values.enabled} />
      <Avg ms={stageAvg.vad} color={SC.vad} role="Voice capture" />
    </Tile>
    {arrow}
    <Tile title="Logic" page="logic" on={rows.length > 0} loading={busy("stt", "llm")} prov={`${rows.length} enabled`} flex={2}>
      {rows.length === 0 && <div className="mdl">no enabled chain</div>}
      {rows.map((r) => <div key={r.name} className="mdl" data-testid="chain-row" title={`${r.stt} → ${r.handler}`} style={{ display: "flex", gap: 8 }}>
        <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{r.name}</span>
        <span style={{ color: "var(--mut)" }}>{fmtShare(r.share)}</span>
        <span style={{ color: SC.llm }}>{fmtSec(r.avg_ms)}</span>
      </div>)}
      <McpChip config={config} />
    </Tile>
    {arrow}
    <Tile title="Speech" page="speech" on={!!speech.provider} loading={busy("tts", "stress")} prov={speech.provider || "—"}>
      <div className="mdl" data-testid="speech-voice">{speech.voice || "—"}{speech.fallbacks > 0 ? ` · +${speech.fallbacks} fallback` : ""}</div>
      <Mark label="stress" on={!!stress.values.enabled} />
      <Mark label="follow-up" on={!!followup.values.enabled} />
      <Avg ms={stageAvg.tts} color={SC.tts} role="Text → speech" />
    </Tile>
  </div></div>;
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
    <ServiceMap catalog={catalog} config={config} stageAvg={stageAvg} perChain={m.per_chain} reloading={reloading} />

    <div className="z-sl">Recent requests<div className="ln" /><a onClick={() => nav("log")}>View full log →</a></div>
    <div className="z-card">
      {loading ? <Loading />
        : recent.length === 0 ? <div className="z-empty"><div className="ic"><Ic n="log" w={20} /></div><b>No runs</b>Recent requests will appear here after the first one is processed.</div>
          : <>
            <div className="z-tblwrap">
              <table className="z-tbl">
                <thead><tr><th style={{ width: 80 }}>Time</th><th style={{ width: 150 }}>Device</th><th>Recognized</th><th>Response</th><th style={{ width: 80, textAlign: "right" }}>Σ</th><th style={{ width: "20%" }}>Stage waterfall</th><th style={{ width: 92 }}>Status</th></tr></thead>
                <tbody>
                  {recent.map((r) => {
                    const rm = statusMeta(r);
                    return <tr key={r.key} onClick={() => { if (r.id == null) return; try { localStorage.setItem("z-openreq", String(r.id)); } catch { /* ignore */ } nav("log"); }}>
                      <td className="tm">{r.time}</td>
                      <td style={{ fontWeight: 600 }}>{deviceLabel(catalog?.core?.values?.devices, r.device)}</td>
                      <td><div className={"z-tx" + (r.stt ? "" : " mut")}>{r.stt || (r.live ? "…" : isAnnouncement(r) ? "—" : "(silence)")}</div></td>
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
