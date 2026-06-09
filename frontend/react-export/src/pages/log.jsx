// Request Log — API-driven. Lists pipeline runs from /api/runs, opens a per-run
// drawer (full timeline + transcript + tool calls + metadata) via
// /api/runs/{id}. Reuses the design-system primitives (Waterfall/KV) and
// the existing CSS (z-tbl/z-gantt/z-drawer/...).
import React, { useState, useEffect, useCallback, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import { PageHeader, Waterfall, KV, Loading, ErrorBox } from "../components/primitives.jsx";
import { getRuns, getRun, openRunsStream, runAudioUrl, downloadRunAudio } from "../api.js";
import { RESULT_META, STAGE_COLOR, fmtSec, mapRun, totalMs, statusMeta, applyStreamedRun } from "../runsModel.js";
import { matchesFilters } from "../runsFilters.js";

const SC = STAGE_COLOR;

// Stage rows for the drawer Gantt, in pipeline order. RUAccent is not a real
// backend stage yet, so it is intentionally omitted.
const GSTAGES = [
  { k: "vad", label: "VAD capture" }, { k: "stt", label: "STT" },
  { k: "llm", label: "LLM + tools" }, { k: "tts", label: "TTS synth" },
];

// matchesFilters lives in ../runsFilters.js (extracted for unit tests); it is the
// client-side mirror of the backend /api/runs filters used to decide whether a
// live-pushed run matches the current UI filters.

// Per-stage horizontal Gantt with a "◆ max" marker on the slowest stage and a
// hatched fail segment for error runs.
function Gantt({ r }) {
  const present = GSTAGES.filter((g) => r.t[g.k] > 0);
  const tot = present.reduce((a, g) => a + r.t[g.k], 0) || 1;
  const maxK = present.reduce((m, g) => (r.t[g.k] > r.t[m.k] ? g : m), present[0] || { k: "vad" });
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
    {r.error && <div className="z-grow">
      <div className="lbl"><s style={{ background: "#dc2626" }} />{r.error.stage}</div>
      <div className="z-gtrack"><div className="z-gseg" style={{ left: off + "%", width: (100 - off) + "%", background: "repeating-linear-gradient(45deg,#dc2626,#dc2626 4px,#fecaca 4px,#fecaca 8px)" }} /></div>
      <div className="ms" style={{ color: "var(--bad)" }}>fail</div>
    </div>}
    <div className="z-grow" style={{ borderTop: "1px solid var(--line)", paddingTop: 8, marginTop: 2 }}>
      <div className="lbl" style={{ fontWeight: 700 }}>Total</div><div />
      <div className="ms" style={{ fontWeight: 700 }}>{fmtSec(totalMs(r))}</div>
    </div>
  </div>;
}

// Right-side drawer with the full run detail. `r` is the mapped detail row.
function Drawer({ r, loading, error, onClose }) {
  const m = RESULT_META[r && r.result] || { label: r && r.result, tone: "muted" };
  return <>
    <div className="z-scrim" onClick={onClose} />
    <div className="z-drawer" role="dialog">
      <div className="z-drawer-h">
        <div>
          <h2>{(r && r.device) || "Run"} {r && <span style={{ color: "var(--mut2)", fontWeight: 500, fontFamily: "var(--mono)", fontSize: 13 }}>· {r.time}</span>}</h2>
          {r && <div style={{ fontSize: 11.5, color: "var(--mut)", marginTop: 2, fontFamily: "var(--mono)" }}>id {r.id} · end: {r.reason || "—"}</div>}
        </div>
        {r && <span className={"z-pill " + m.tone} style={{ marginLeft: 12 }}><span className={"z-dot " + (m.tone === "good" ? "ok" : m.tone === "bad" ? "error" : "off")} />{m.label}</span>}
        <button className="z-x" aria-label="Close" onClick={onClose}><svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg></button>
      </div>
      <div className="z-drawer-b">
        {loading && <Loading />}
        {error && <ErrorBox error={error} />}
        {r && !loading && !error && <>
          {r.error && <div className="z-banner" style={{ background: "var(--bad-bg)", border: "1px solid #f3c8c8", color: "#b91c1c", marginBottom: 16 }}>
            <Ic n="restart" w={15} /><span><b>{r.error.stage} error.</b> {r.error.text}</span>
          </div>}

          <div className="z-sl" style={{ marginTop: 0 }}>Stage timeline</div>
          <div className="z-card"><div style={{ padding: "15px 17px" }}><Gantt r={r} /></div></div>

          <div className="z-sl">Transcript</div>
          <div className="z-card"><div style={{ padding: "4px 17px" }}>
            <div className="z-f"><div className="z-fl"><b style={{ color: "var(--mut)", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: ".04em" }}>Recognized (STT)</b></div>
              <div style={{ fontSize: 14, color: r.stt ? "var(--ink)" : "var(--mut2)", fontStyle: r.stt ? "normal" : "italic" }}>{r.stt || "— silence, no speech detected —"}</div></div>
            {r.filler_text && (
              <div className="z-f"><div className="z-fl">
                <b style={{ color: "var(--mut)", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: ".04em" }}>
                  Filler (early reply){r.t_filler != null ? ` · ${fmtSec(r.t_filler)}` : ""}
                </b></div>
                <div style={{ fontSize: 14, color: "var(--ink)" }}>{r.filler_text}</div>
              </div>
            )}
            <div className="z-f"><div className="z-fl"><b style={{ color: "var(--mut)", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: ".04em" }}>Response (LLM)</b></div>
              <div style={{ fontSize: 14, color: r.llm ? "var(--ink)" : "var(--mut2)", fontStyle: r.llm ? "normal" : "italic" }}>{r.llm || "— no response produced —"}</div></div>
          </div></div>

          {r.has_audio ? <>
            <div className="z-sl">Utterance audio</div>
            <div className="z-card"><div style={{ padding: "12px 17px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <audio controls preload="none" src={runAudioUrl(r.id)} style={{ height: 34, flex: 1, minWidth: 220 }} />
              <button className="z-btn" onClick={() => downloadRunAudio(r.id)}>Download WAV</button>
            </div></div>
          </> : null}

          {((r.rounds && r.rounds.length > 0) || r.request) && <>
            <div className="z-sl">LLM rounds & tool calls<div className="ln" /><span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--mut2)", textTransform: "none", letterSpacing: 0 }}>{r.model || "—"} · {r.tokens || 0} tok</span></div>
            {r.request && <div className="z-round">
              <div className="z-round-h"><span className="rn">IN</span><span style={{ color: "var(--ink2)" }}>Model input — full debug</span></div>
              <div className="z-tool">
                <div className="tn"><s>prompt</s> System prompt</div>
                <div className="z-code">{r.request.system_prompt || "—"}</div>
              </div>
              <div className="z-tool">
                <div className="tn"><s>context</s> Context · {(r.request.context || []).length} msg</div>
                {(r.request.context && r.request.context.length)
                  ? r.request.context.map((m, k) => <div className="z-code" style={{ marginTop: 6 }} key={k}><span className="k">{m.role}: </span>{m.content}</div>)
                  : <div className="z-code" style={{ color: "var(--mut)" }}>— no prior context —</div>}
              </div>
              <div className="z-tool">
                <div className="tn"><s>tools</s> Tools · {(r.request.tools || []).length}</div>
                {(r.request.tools && r.request.tools.length)
                  ? r.request.tools.map((t, k) => { const f = (t && t.function) || t || {}; return <div className="z-code" style={{ marginTop: 6 }} key={k}><b>{f.name}</b>{f.description ? ` — ${f.description}` : ""}{f.parameters ? "\n" + JSON.stringify(f.parameters, null, 2) : ""}</div>; })
                  : <div className="z-code" style={{ color: "var(--mut)" }}>— no tools advertised —</div>}
              </div>
              <div className="z-tool">
                <div className="tn"><s>user</s> User message</div>
                <div className="z-code">{r.request.user_text || "—"}</div>
              </div>
            </div>}
            {(r.rounds || []).map((rd, i) => <div className="z-round" key={i}>
              <div className="z-round-h"><span className="rn">R{rd.round}</span><span style={{ color: "var(--ink2)" }}>{rd.note}</span><span className="tok">{rd.tokens} tok</span></div>
              {(!rd.calls || rd.calls.length === 0)
                ? <div className="z-tool">
                    <div className="tn"><s>final</s> Final text</div>
                    <div className="z-code">{rd.content || "— (empty) —"}</div>
                  </div>
                : <>
                    {rd.content ? <div className="z-tool"><div className="tn"><s>note</s> Assistant text</div><div className="z-code">{rd.content}</div></div> : null}
                    {rd.calls.map((c, j) => <div className="z-tool" key={j}>
                      <div className="tn"><s>call</s> {c.name}</div>
                      <div className="z-code">{JSON.stringify(c.args)}</div>
                      <div className="z-codeline"><span className="arr">→ result</span><span className="mono" style={{ color: "var(--ink2)" }}>{c.result}</span></div>
                    </div>)}
                  </>}
            </div>)}
          </>}

          <div className="z-sl">Metadata</div>
          <div className="z-card"><div style={{ padding: "6px 17px" }}>
            <KV k="Device" v={r.device || "—"} />
            <KV k="End reason" v={r.reason || "—"} />
            <KV k="Model" v={r.model || "—"} />
            <KV k="Total tokens" v={r.tokens || "—"} />
            {r.filler_text && <KV k="Filler spoken at" v={fmtSec(r.t_filler)} />}
            <KV k="Audio" v={r.audio ? `${(r.audio.bytes / 1024).toFixed(0)} kB · ${r.audio.fmt}` : "—"} />
            <KV k="Total duration" v={fmtSec(totalMs(r))} />
          </div></div>
        </>}
      </div>
    </div>
  </>;
}

function Log() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [result, setResult] = useState("all");       // all → errors → ok
  const [search, setSearch] = useState("");
  const [device, setDevice] = useState("");

  // Drawer state: selected summary row + lazily-fetched detail.
  const [openId, setOpenId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    const params = { limit: 100 };
    if (result !== "all") params.result = result;
    if (search.trim()) params.search = search.trim();
    if (device.trim()) params.device = device.trim();
    getRuns(params)
      .then((data) => { setRuns((data.runs || []).map(mapRun)); setError(null); })
      .catch((e) => setError(e))
      .finally(() => setLoading(false));
  }, [result, search, device]);

  useEffect(() => { load(); }, [load]);

  // Keep the latest filter values in a ref so the single live subscription below
  // always matches against current filters without re-subscribing per keystroke.
  const filtersRef = useRef({ result, search, device });
  useEffect(() => { filtersRef.current = { result, search, device }; }, [result, search, device]);

  useEffect(() => {
    const stop = openRunsStream((row) => {
      const mapped = mapRun(row);
      const match = matchesFilters(row, filtersRef.current);
      setRuns((prev) => applyStreamedRun(prev, mapped, match, 100));
    });
    return stop;
  }, []);

  const openRow = useCallback((id) => {
    setOpenId(id);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
    getRun(id)
      .then((row) => setDetail(mapRun(row)))
      .catch((e) => setDetailError(e))
      .finally(() => setDetailLoading(false));
  }, []);

  // Open the run requested by the Dashboard (cross-page deep link), once loaded.
  useEffect(() => {
    let id; try { id = localStorage.getItem("z-openreq"); localStorage.removeItem("z-openreq"); } catch { /* ignore */ }
    if (id) openRow(Number(id));
  }, [openRow]);

  const close = () => { setOpenId(null); setDetail(null); setDetailError(null); };

  return <div className="z-page">
    <PageHeader title="Request log" desc="Every pipeline run with per-stage timings. Click a row for the full waterfall, tool calls and metadata." />
    <div className="z-card">
      <div className="z-filters">
        <div className="z-search"><Ic n="search" w={13} />
          <input placeholder="Search recognized / response…" value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") load(); }} />
        </div>
        <div className="z-search" style={{ maxWidth: 180 }}>
          <input placeholder="Device…" value={device}
            onChange={(e) => setDevice(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") load(); }} />
        </div>
        <div className="z-fchip" onClick={() => setResult(result === "all" ? "errors" : result === "errors" ? "ok" : "all")}>Result · <b>{result}</b> ▾</div>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--mut2)", fontFamily: "var(--mono)" }}>{runs.length} runs</span>
      </div>

      {loading ? <Loading />
        : error ? <ErrorBox error={error} onRetry={load} />
          : runs.length === 0 ? <div className="z-empty"><div className="ic"><Ic n="log" w={20} /></div><b>No recorded runs yet</b>Once the assistant processes a request, it will appear here.</div>
            : <>
              <div className="z-tblwrap">
                <table className="z-tbl">
                  <thead><tr><th>Time</th><th>Device</th><th>Recognized</th><th>Response</th><th style={{ textAlign: "right" }}>Tok</th><th>Stage waterfall</th><th style={{ textAlign: "right" }}>Σ</th><th>Status</th></tr></thead>
                  <tbody>
                    {runs.map((r) => {
                      const m = statusMeta(r);
                      return <tr key={r.key} className={openId === r.id ? "sel" : ""} onClick={() => { if (r.id != null) openRow(r.id); }}>
                        <td className="tm">{r.time}</td>
                        <td style={{ fontWeight: 600 }}>{r.device}</td>
                        <td><div className={"z-tx" + (r.stt ? "" : " mut")}>{r.stt || (r.live ? "…" : "(silence)")}</div></td>
                        <td><div className={"z-tx" + (r.llm ? "" : " mut")}>{r.llm || (r.live ? "…" : "—")}</div></td>
                        <td className="num">{r.tokens || "—"}</td>
                        <td><Waterfall r={r} /></td>
                        <td className="num" style={{ fontWeight: 600 }}>{fmtSec(totalMs(r))}</td>
                        <td><span className={"z-st " + m.tone}><span className={"z-dot " + (m.tone === "good" ? "ok" : m.tone === "bad" ? "error" : "off")} />{m.label}</span></td>
                      </tr>;
                    })}
                  </tbody>
                </table>
              </div>
              <div className="z-tfoot">{runs.length} runs</div>
            </>}
    </div>
    {openId != null && <Drawer r={detail} loading={detailLoading} error={detailError} onClose={close} />}
  </div>;
}

export default Log;
