import React, { useState, useEffect, useRef } from "react";
import Z from "../data.js";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { Spark, Field, Seg, Selector, Toggle, Slider, Stepper, Select, Pill, StatusPill, Waterfall, segsFor, total, PageHeader, SaveBar, Modal, Player, KV } from "../components/primitives.jsx";
  const SC = Z.stageColor;
  const fmt = (ms) => (ms / 1000).toFixed(2) + "s";

  function Dashboard() {
    return <div className="z-page">
      <PageHeader title="Pipeline overview" desc="Live voice loop across all stages. Click a stage to configure it."
        actions={<><button className="z-btn g"><Ic n="test" w={14} />Run self-test</button></>} />

      {Z.meta.pendingRestart && <div className="z-banner warn">
        <Ic n="restart" w={15} />
        <span><b>Restart required.</b> Some staged changes apply only after the service restarts.</span>
        <span className="act"><button className="z-btn warn sm">Restart now</button></span>
      </div>}

      <div className="z-kpis">
        {Z.kpis.map((c, i) => <div className="z-kpi" key={i}>
          <span className="k">{c.k}</span>
          <div className="v">{c.v}{c.u && <small> {c.u}</small>}</div>
          <div className={"d " + c.d[0]}>{c.d[0] === "up" ? "▲" : c.d[0] === "down" ? "▼" : "■"} {c.d[1]}</div>
          <span className="spk"><Spark pts={c.spark} color={c.color} /></span>
        </div>)}
      </div>

      <div className="z-sl">Pipeline service map<div className="ln" /></div>
      <div className="z-card"><div className="z-map">
        {Z.stages.map((s, i) => {
          const avg = s.key === "ruaccent" ? 0 : Math.round(Z.requests.reduce((a, r) => a + r.t[s.key], 0) / Z.requests.length);
          return <React.Fragment key={s.key}>
            <div className={"z-svc" + (s.status === "off" ? " off" : "")} onClick={() => nav(s.key)}>
              <div className="z-svc-h"><span className={"z-dot " + s.status} /><b>{s.name}</b><span className="prov">{s.provider}</span></div>
              <div className="mdl">{s.detail}</div>
              <div className="lat">{s.key === "ruaccent" ? <span style={{ fontSize: 12, color: "var(--mut)" }}>bypassed</span> : <><b style={{ color: SC[s.key] }}>{(avg / 1000).toFixed(2)}</b><s>s avg</s></>}</div>
              {s.mcp ? <span className="z-mcpchip" onClick={(e) => { e.stopPropagation(); nav("mcp"); }}>◆ MCP {s.mcp.servers}·{s.mcp.tools}</span>
                : s.key === "ruaccent" ? <span className="z-toggle sm" style={{ pointerEvents: "none" }} />
                  : <Spark pts={[avg * 0.8, avg * 1.1, avg * 0.9, avg, avg * 1.2, avg * 0.95].map((x) => x || 1)} color={SC[s.key]} w={92} h={16} />}
              <span className="cfg">Configure <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M2 6h7M6 3l3 3-3 3" /></svg></span>
            </div>
            {i < Z.stages.length - 1 && <div className="z-arrow"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 8h11M9 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" /></svg></div>}
          </React.Fragment>;
        })}
      </div></div>

      <div className="z-sl">Recent requests<div className="ln" /><a onClick={() => nav("log")}>View full log →</a></div>
      <div className="z-card">
        <div className="z-tblwrap">
        <table className="z-tbl">
          <thead><tr><th>Time</th><th>Device</th><th>Recognized</th><th>Response</th><th style={{ textAlign: "right" }}>Σ</th><th>Stage waterfall</th><th>Status</th></tr></thead>
          <tbody>
            {Z.requests.slice(0, 5).map((r) => {
              const m = Z.resultMeta[r.result];
              return <tr key={r.id} onClick={() => { try { localStorage.setItem("z-openreq", r.id); } catch {} nav("log"); }}>
                <td className="tm">{r.time}</td>
                <td style={{ fontWeight: 600 }}>{r.device}</td>
                <td><div className={"z-tx" + (r.stt ? "" : " mut")}>{r.stt || "(silence)"}</div></td>
                <td><div className={"z-tx" + (r.llm && r.llm !== "—" ? "" : " mut")}>{r.llm && r.llm !== "—" ? r.llm : "—"}</div></td>
                <td className="num" style={{ fontWeight: 600 }}>{r.result === "empty" ? "8.00s" : fmt(total(r.t))}</td>
                <td><Waterfall r={r} /></td>
                <td><span className={"z-st " + m.tone}><span className={"z-dot " + (m.tone === "good" ? "ok" : m.tone === "bad" ? "error" : "off")} />{m.label}</span></td>
              </tr>;
            })}
          </tbody>
        </table>
        </div>
        <div className="z-tfoot">Showing 5 of 142 runs · p50 2.41s · p95 3.62s · 1 error · 1 empty · updated 4s ago</div>
      </div>
    </div>;
  }

export default Dashboard;
