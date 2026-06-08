import React, { useState, useEffect, useRef } from "react";
import Z from "../data.js";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { Spark, Field, Seg, Selector, Toggle, Slider, Stepper, Select, Pill, StatusPill, Waterfall, segsFor, total, PageHeader, SaveBar, Modal, Player, KV } from "../components/primitives.jsx";
  const SC = Z.stageColor;
  const fmt = (ms) => (ms / 1000).toFixed(2) + "s";
  const GSTAGES = [
    { k: "vad", label: "VAD capture" }, { k: "stt", label: "STT" },
    { k: "llm", label: "LLM + tools" }, { k: "ruaccent", label: "RUAccent" }, { k: "tts", label: "TTS synth" }
  ];

  function Gantt({ r }) {
    const present = GSTAGES.filter((g) => r.t[g.k] > 0);
    const tot = present.reduce((a, g) => a + r.t[g.k], 0) || 1;
    const maxK = present.reduce((m, g) => r.t[g.k] > r.t[m.k] ? g : m, present[0] || { k: "vad" });
    let off = 0;
    return <div className="z-gantt">
      {present.map((g) => {
        const w = r.t[g.k] / tot * 100; const left = off; off += w;
        const bott = g.k === maxK.k;
        return <div className="z-grow" key={g.k}>
          <div className="lbl"><s style={{ background: SC[g.k] }} />{g.label}{bott && <span style={{ fontSize: 9.5, fontWeight: 700, color: "var(--warn)", fontFamily: "var(--mono)" }}>◆ max</span>}</div>
          <div className="z-gtrack"><div className="z-gseg" style={{ left: left + "%", width: w + "%", background: SC[g.k] }} /></div>
          <div className="ms">{r.t[g.k]}<span style={{ color: "var(--mut2)", fontWeight: 400 }}> ms</span></div>
        </div>;
      })}
      {r.result === "error" && <div className="z-grow">
        <div className="lbl"><s style={{ background: "#dc2626" }} />TTS synth</div>
        <div className="z-gtrack"><div className="z-gseg" style={{ left: off + "%", width: (100 - off) + "%", background: "repeating-linear-gradient(45deg,#dc2626,#dc2626 4px,#fecaca 4px,#fecaca 8px)" }} /></div>
        <div className="ms" style={{ color: "var(--bad)" }}>fail</div>
      </div>}
      <div className="z-grow" style={{ borderTop: "1px solid var(--line)", paddingTop: 8, marginTop: 2 }}>
        <div className="lbl" style={{ fontWeight: 700 }}>Total</div><div />
        <div className="ms" style={{ fontWeight: 700 }}>{r.result === "empty" ? "8.00 s" : fmt(total(r.t))}</div>
      </div>
    </div>;
  }

  function Drawer({ r, onClose }) {
    const m = Z.resultMeta[r.result];
    return <>
      <div className="z-scrim" onClick={onClose} />
      <div className="z-drawer" role="dialog">
        <div className="z-drawer-h">
          <div>
            <h2>{r.device} <span style={{ color: "var(--mut2)", fontWeight: 500, fontFamily: "var(--mono)", fontSize: 13 }}>· {r.time}</span></h2>
            <div style={{ fontSize: 11.5, color: "var(--mut)", marginTop: 2, fontFamily: "var(--mono)" }}>id {r.id} · end: {r.reason}</div>
          </div>
          <span className={"z-pill " + m.tone} style={{ marginLeft: 12 }}><span className={"z-dot " + (m.tone === "good" ? "ok" : m.tone === "bad" ? "error" : "off")} />{m.label}</span>
          <button className="z-x" onClick={onClose}><svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M4 4l8 8M12 4l-8 8" /></svg></button>
        </div>
        <div className="z-drawer-b">
          {r.error && <div className="z-banner" style={{ background: "var(--bad-bg)", border: "1px solid #f3c8c8", color: "#b91c1c", marginBottom: 16 }}>
            <Ic n="restart" w={15} /><span><b>{r.error.stage} error.</b> {r.error.text}</span>
          </div>}

          <div className="z-sl" style={{ marginTop: 0 }}>Stage timeline</div>
          <div className="z-card"><div style={{ padding: "15px 17px" }}><Gantt r={r} /></div></div>

          <div className="z-sl">Transcript</div>
          <div className="z-card"><div style={{ padding: "4px 17px" }}>
            <div className="z-f"><div className="z-fl"><b style={{ color: "var(--mut)", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: ".04em" }}>Recognized (STT)</b></div>
              <div style={{ fontSize: 14, color: r.stt ? "var(--ink)" : "var(--mut2)", fontStyle: r.stt ? "normal" : "italic" }}>{r.stt || "— silence, no speech detected —"}</div></div>
            <div className="z-f"><div className="z-fl"><b style={{ color: "var(--mut)", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: ".04em" }}>Response (LLM)</b></div>
              <div style={{ fontSize: 14, color: r.llm && r.llm !== "—" ? "var(--ink)" : "var(--mut2)", fontStyle: r.llm && r.llm !== "—" ? "normal" : "italic" }}>{r.llm && r.llm !== "—" ? r.llm : "— no response produced —"}</div></div>
          </div></div>

          {r.rounds && r.rounds.length > 0 && <>
            <div className="z-sl">LLM rounds & tool calls<div className="ln" /><span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--mut2)", textTransform: "none", letterSpacing: 0 }}>{r.model} · {r.tokens} tok</span></div>
            {r.rounds.map((rd, i) => <div className="z-round" key={i}>
              <div className="z-round-h"><span className="rn">R{rd.round}</span><span style={{ color: "var(--ink2)" }}>{rd.note}</span><span className="tok">{rd.tokens} tok</span></div>
              {rd.calls.length === 0
                ? <div className="z-tool" style={{ color: "var(--mut)", fontSize: 12 }}>No tool calls — model produced final text.</div>
                : rd.calls.map((c, j) => <div className="z-tool" key={j}>
                  <div className="tn"><s>call</s> {c.name}</div>
                  <div className="z-code">{JSON.stringify(c.args)}</div>
                  <div className="z-codeline"><span className="arr">→ result</span><span className="mono" style={{ color: "var(--ink2)" }}>{c.result}</span></div>
                </div>)}
            </div>)}
          </>}

          <div className="z-sl">Synthesized audio</div>
          {r.audio ? <Player audio={r.audio} bars={54} />
            : <div className="z-card"><div className="z-empty" style={{ padding: "26px 20px" }}><div className="ic"><Ic n="tts" w={18} /></div><b>No audio</b>{r.result === "empty" ? "Empty input — nothing was synthesized." : "TTS failed before producing audio."}</div></div>}

          <div className="z-sl">Metadata</div>
          <div className="z-card"><div style={{ padding: "6px 17px" }}>
            <KV k="Device" v={r.device} />
            <KV k="End reason" v={r.reason} />
            <KV k="Model (provider echo)" v={r.model || "—"} />
            <KV k="Total tokens" v={r.tokens || "—"} />
            <KV k="Audio" v={r.audio ? `${(r.audio.bytes / 1024).toFixed(0)} kB · ${r.audio.fmt}` : "—"} />
            <KV k="Total duration" v={r.result === "empty" ? "8.00 s" : fmt(total(r.t))} />
          </div></div>
        </div>
      </div>
    </>;
  }

  function Log() {
    const [open, setOpen] = useState(null);
    const [result, setResult] = useState("all");
    useEffect(() => {
      let id; try { id = localStorage.getItem("z-openreq"); localStorage.removeItem("z-openreq"); } catch {}
      if (id) { const r = Z.requests.find((x) => x.id === id); if (r) setOpen(r); }
    }, []);
    const rows = Z.requests.filter((r) => result === "all" || (result === "errors" ? r.result === "error" : result === "ok" ? (r.result === "ok" || r.result === "tool") : true));
    return <div className="z-page">
      <PageHeader title="Request log" desc="Every pipeline run with per-stage timings. Click a row for the full waterfall, tool calls and audio."
        actions={<button className="z-btn g"><Ic n="ext" w={14} />Export</button>} />
      <div className="z-card">
        <div className="z-filters">
          <div className="z-search"><Ic n="search" w={13} /><input placeholder="Search recognized / response…" /></div>
          <div className="z-fchip">Device · <b>all</b> ▾</div>
          <div className="z-fchip" onClick={() => setResult(result === "all" ? "errors" : result === "errors" ? "ok" : "all")}>Result · <b>{result}</b> ▾</div>
          <div className="z-fchip">Reason · <b>any</b> ▾</div>
          <span style={{ flex: 1 }} />
          <span style={{ fontSize: 11, color: "var(--mut2)", fontFamily: "var(--mono)" }}>{rows.length} runs</span>
        </div>
        <div className="z-tblwrap">
        <table className="z-tbl">
          <thead><tr><th>Time</th><th>Device</th><th>Recognized</th><th>Response</th><th style={{ textAlign: "right" }}>Tok</th><th>Stage waterfall</th><th style={{ textAlign: "right" }}>Σ</th><th>Status</th></tr></thead>
          <tbody>
            {rows.map((r) => {
              const m = Z.resultMeta[r.result];
              return <tr key={r.id} className={open && open.id === r.id ? "sel" : ""} onClick={() => setOpen(r)}>
                <td className="tm">{r.time}</td>
                <td style={{ fontWeight: 600 }}>{r.device}</td>
                <td><div className={"z-tx" + (r.stt ? "" : " mut")}>{r.stt || "(silence)"}</div></td>
                <td><div className={"z-tx" + (r.llm && r.llm !== "—" ? "" : " mut")}>{r.llm && r.llm !== "—" ? r.llm : "—"}</div></td>
                <td className="num">{r.tokens || "—"}</td>
                <td><Waterfall r={r} /></td>
                <td className="num" style={{ fontWeight: 600 }}>{r.result === "empty" ? "8.00s" : fmt(total(r.t))}</td>
                <td><span className={"z-st " + m.tone}><span className={"z-dot " + (m.tone === "good" ? "ok" : m.tone === "bad" ? "error" : "off")} />{m.label}</span></td>
              </tr>;
            })}
          </tbody>
        </table>
        </div>
        <div className="z-tfoot">{rows.length} of 142 runs · p50 2.41s · p95 3.62s · 1 error · 1 empty · updated 4s ago</div>
      </div>
      {open && <Drawer r={open} onClose={() => setOpen(null)} />}
    </div>;
  }

export default Log;
