// Request Log — API-driven. Lists pipeline runs from /api/runs, opens a per-run
// drawer (full timeline + transcript + tool calls + metadata) via
// /api/runs/{id}. Reuses the design-system primitives (Waterfall/KV) and
// the existing CSS (z-tbl/z-gantt/z-drawer/...).
import React, { useState, useEffect, useCallback, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import { PageHeader, Waterfall, KV, Loading, ErrorBox, Select, fillerMarkerPct } from "../components/primitives.jsx";
import { getRuns, getRun, openRunsStream, runAudioUrl, downloadRunAudio, runTtsAudioUrl, downloadRunTtsAudio } from "../api.js";
import { RESULT_META, STAGE_COLOR, fmtSec, mapRun, totalMs, statusMeta, applyStreamedRun, pageWindow } from "../runsModel.js";
import { FILLER_COLOR } from "../stageMeta.js";
import { matchesFilters } from "../runsFilters.js";

const SC = STAGE_COLOR;

const PAGE_SIZES = [50, 100, 200]; // selectable rows-per-page for numbered pagination
const DEFAULT_PAGE_SIZE = 100;

// Stage rows for the drawer Gantt, in pipeline order. The accent (RuAccent)
// stage sits between LLM and TTS; the Gantt only renders stages whose timing is
// > 0 (see `present` below), so it shows up only on runs where it actually ran.
const GSTAGES = [
  { k: "vad", label: "VAD capture" }, { k: "wakeword", label: "Wakeword verify" },
  { k: "stt", label: "STT" }, { k: "llm", label: "LLM + tools" },
  { k: "stress", label: "Accents" }, { k: "tts", label: "TTS synth" },
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
  // Early-filler marker: the announcement spoken mid-run. fillerMarkerPct gives
  // its x-position on the same 0..100% axis the stage bars use (null when no
  // filler fired). The tick is drawn inside the track of the stage it falls in
  // (in practice LLM, since the filler is the LLM's early reply) instead of on a
  // row of its own.
  const fpct = fillerMarkerPct(r);
  let fillerHost = null;
  if (fpct != null) {
    let acc = 0;
    for (let i = 0; i < present.length; i++) {
      const w = r.t[present[i].k] / tot * 100;
      if (fpct >= acc && fpct < acc + w) { fillerHost = i; break; }
      acc += w;
    }
    // fpct clamped to the right edge (=== 100) or past the last segment: attach to
    // the last present stage so the tick always renders.
    if (fillerHost == null && present.length) fillerHost = present.length - 1;
  }
  let off = 0;
  return <div className="z-gantt">
    {present.map((g, i) => {
      const w = r.t[g.k] / tot * 100; const left = off; off += w;
      const bott = g.k === maxK.k;
      return <div className="z-grow" key={g.k}>
        <div className="lbl"><s style={{ background: SC[g.k] }} />{g.label}{bott && <span style={{ fontSize: 9.5, fontWeight: 700, color: "var(--warn)", fontFamily: "var(--mono)" }}>◆ max</span>}</div>
        <div className="z-gtrack">
          <div className="z-gseg" style={{ left: left + "%", width: w + "%", background: SC[g.k] }} />
          {i === fillerHost && <div className="z-wffiller" style={{ left: fpct + "%", background: FILLER_COLOR }}
            title={`🗣 «${r.filler_text}» — early reply at ${fmtSec(r.t_filler)}`} />}
        </div>
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

// A click-to-expand row: renders only `summary` until toggled open, so heavy
// blocks (full system prompt, a tool's JSON schema) stay collapsed by default.
function Collapsible({ summary, children }) {
  const [open, setOpen] = useState(false);
  return <>
    <div className="tn" style={{ cursor: "pointer", userSelect: "none" }} onClick={() => setOpen((o) => !o)}>
      <span style={{ display: "inline-block", width: 10, flex: "none", color: "var(--mut2)", fontSize: 9, transform: open ? "rotate(90deg)" : "none", transition: "transform .12s" }}>▸</span>
      {summary}
    </div>
    {open && children}
  </>;
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

          {/* The server-side wake-word verifier blocked this run before STT (a false
              trigger). Mirrors the error banner, but muted — it is an expected reject,
              not a failure. reason is wakeword_reject (low score) or wakeword_error. */}
          {r.result === "rejected" && <div className="z-banner" style={{ background: "var(--panel2)", border: "1px solid var(--line)", color: "var(--mut)", marginBottom: 16 }}>
            <Ic n="wakeword" w={15} /><span><b>Rejected by wake-word verify.</b> {r.reason === "wakeword_error" ? "verifier error" : "wake word not confirmed"}{r.wakeword_score != null ? ` · score ${r.wakeword_score.toFixed(2)}` : ""}</span>
          </div>}

          <div className="z-sl" style={{ marginTop: 0 }}>Stage timeline</div>
          <div className="z-card"><div style={{ padding: "15px 17px" }}><Gantt r={r} /></div></div>

          {/* Wake-word verifier confidence + time, shown whenever the stage produced a
              score (a passed run) or actively rejected the phrase. */}
          {(r.wakeword_score != null || r.result === "rejected") && <>
            <div className="z-sl">Wake-word verify<div className="ln" />{r.t && r.t.wakeword ? <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--mut2)", textTransform: "none", letterSpacing: 0 }}>{fmtSec(r.t.wakeword)}</span> : null}</div>
            <div className="z-card"><div style={{ padding: "6px 17px" }}>
              <KV k="Score" v={r.wakeword_score != null ? r.wakeword_score.toFixed(3) : "—"} />
              <KV k="Verdict" v={r.result === "rejected" ? "Rejected" : "Passed"} />
              <KV k="Verify time" v={fmtSec(r.t && r.t.wakeword)} />
            </div></div>
          </>}

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
            {r.audio_channels === 2
              ? <div className="z-card"><div style={{ padding: "12px 17px", display: "flex", flexDirection: "column", gap: 10 }}>
                  {/* Stereo recording: channel 0 (left) is what STT received, channel 1
                      (right) is the other raw mic channel. Each player streams a mono
                      WAV split server-side; Download keeps the full stereo file. */}
                  {[["stt", "STT channel"], ["raw", "Raw channel"]].map(([ch, label]) => (
                    <div key={ch} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <b style={{ width: 90, flex: "none", color: "var(--mut)", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: ".04em" }}>{label}</b>
                      <audio controls preload="none" src={runAudioUrl(r.id, ch)} style={{ height: 34, flex: 1, minWidth: 220 }} />
                    </div>
                  ))}
                  <div><button className="z-btn" onClick={() => downloadRunAudio(r.id)}>Download WAV</button></div>
                </div></div>
              : <div className="z-card"><div style={{ padding: "12px 17px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                  <audio controls preload="none" src={runAudioUrl(r.id)} style={{ height: 34, flex: 1, minWidth: 220 }} />
                  <button className="z-btn" onClick={() => downloadRunAudio(r.id)}>Download WAV</button>
                </div></div>}
          </> : null}

          {((r.rounds && r.rounds.length > 0) || r.request) && <>
            <div className="z-sl">LLM rounds & tool calls<div className="ln" /><span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--mut2)", textTransform: "none", letterSpacing: 0 }}>{r.model || "—"} · {r.tokens || 0} tok</span></div>
            {r.request && <div className="z-round">
              <div className="z-round-h"><span className="rn">IN</span><span style={{ color: "var(--ink2)" }}>Model input — full debug</span></div>
              <div className="z-tool">
                <Collapsible summary={<><s>prompt</s> System prompt</>}>
                  <div className="z-code" style={{ marginTop: 7 }}>{r.request.system_prompt || "—"}</div>
                </Collapsible>
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
                  ? r.request.tools.map((t, k) => {
                      const f = (t && t.function) || t || {};
                      return <div style={{ marginTop: 8 }} key={k}>
                        <Collapsible summary={<span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}><b>{f.name}</b>{f.description ? ` — ${f.description}` : ""}</span>}>
                          <div className="z-code" style={{ marginTop: 6 }}>{f.parameters ? JSON.stringify(f.parameters, null, 2) : "— no parameters —"}</div>
                        </Collapsible>
                      </div>;
                    })
                  : <div className="z-code" style={{ color: "var(--mut)", marginTop: 6 }}>— no tools advertised —</div>}
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

          {r.stress && <>
            <div className="z-sl">Accents<div className="ln" />{r.t && r.t.stress ? <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--mut2)", textTransform: "none", letterSpacing: 0 }}>{fmtSec(r.t.stress)}</span> : null}</div>
            <div className="z-card"><div style={{ padding: "12px 17px" }}>
              <div style={{ fontSize: 14, color: "var(--ink)", lineHeight: 1.6 }}>{r.stress}</div>
            </div></div>
          </>}

          {r.has_tts_audio ? <>
            <div className="z-sl">TTS audio<div className="ln" />{r.t && r.t.tts ? <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--mut2)", textTransform: "none", letterSpacing: 0 }}>{fmtSec(r.t.tts)}</span> : null}</div>
            <div className="z-card"><div style={{ padding: "12px 17px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <audio controls preload="none" src={runTtsAudioUrl(r.id)} style={{ height: 34, flex: 1, minWidth: 220 }} />
              <button className="z-btn" onClick={() => downloadRunTtsAudio(r.id)}>Download</button>
            </div></div>
          </> : null}

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
  const [result, setResult] = useState("all");       // all → errors → ok → rejected
  const [search, setSearch] = useState("");
  const [device, setDevice] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [total, setTotal] = useState(0);              // total matching rows across all pages

  // Drawer state: selected summary row + lazily-fetched detail.
  const [openId, setOpenId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(null);

  // Mirror of `runs` so the live SSE subscription can read it without depending on
  // the `runs` array (which the stream itself mutates on every frame).
  const runsRef = useRef(runs);
  useEffect(() => { runsRef.current = runs; }, [runs]);
  // Generation token: each load() bumps it; only the latest load may apply its
  // response, so stale out-of-order fetches (rapid typing/page changes) are dropped.
  const genRef = useRef(0);

  const load = useCallback(() => {
    genRef.current += 1; const gen = genRef.current;
    setLoading(true);
    const params = { limit: pageSize, offset: (page - 1) * pageSize };
    if (result !== "all") params.result = result;
    if (search.trim()) params.search = search.trim();
    if (device.trim()) params.device = device.trim();
    getRuns(params)
      .then((data) => {
        if (gen !== genRef.current) return;            // a newer load superseded this one
        setRuns((data.runs || []).map(mapRun)); setTotal(data.total || 0); setError(null);
      })
      .catch((e) => { if (gen === genRef.current) setError(e); })
      .finally(() => { if (gen === genRef.current) setLoading(false); }); // a newer load owns the spinner otherwise
  }, [result, search, device, page, pageSize]);

  useEffect(() => { load(); }, [load]);

  // Keep the latest filter values in a ref so the single live subscription below
  // always matches against current filters without re-subscribing per keystroke.
  const filtersRef = useRef({ result, search, device });
  useEffect(() => { filtersRef.current = { result, search, device }; }, [result, search, device]);
  // Mirror page/pageSize so the single live subscription can read them without
  // resubscribing (same pattern as filtersRef).
  const pageRef = useRef(page);
  useEffect(() => { pageRef.current = page; }, [page]);
  const pageSizeRef = useRef(pageSize);
  useEffect(() => { pageSizeRef.current = pageSize; }, [pageSize]);

  useEffect(() => {
    const stop = openRunsStream((row) => {
      if (pageRef.current !== 1) return;             // only page 1 reflects live runs
      const mapped = mapRun(row);
      const match = matchesFilters(row, filtersRef.current);
      // A brand-new finalized matching run grows the dataset -> bump total. Compute
      // newness from the runs ref BEFORE the state update (no side effects inside the
      // setRuns updater, which React may invoke twice).
      const isNew = match && !mapped.live && !runsRef.current.some((r) => r.key === mapped.key);
      setRuns((prev) => applyStreamedRun(prev, mapped, match, pageSizeRef.current));
      if (isNew) setTotal((t) => t + 1);
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

  // Numbered pagination math: a filter/page-size change always resets `page` to 1
  // (see the handlers below), so `offset` can never point past the dataset end.
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, total);

  // Safety net: if the dataset shrinks (e.g. retention pruning) so the current page
  // is now past the end, snap back to the last valid page. totalPages is always >= 1,
  // so this converges in one step and never loops.
  useEffect(() => { if (page > totalPages) setPage(totalPages); }, [page, totalPages]);

  return <div className="z-page">
    <PageHeader title="Request log" desc="Every pipeline run with per-stage timings. Click a row for the full waterfall, tool calls and metadata." />
    <div className="z-card">
      <div className="z-filters">
        <div className="z-search"><Ic n="search" w={13} />
          <input placeholder="Search recognized / response…" value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }} />
        </div>
        <div className="z-search" style={{ maxWidth: 180 }}>
          <input placeholder="Device…" value={device}
            onChange={(e) => { setDevice(e.target.value); setPage(1); }} />
        </div>
        <div className="z-fchip" onClick={() => { setResult(result === "all" ? "errors" : result === "errors" ? "ok" : result === "ok" ? "rejected" : "all"); setPage(1); }}>Result · <b>{result}</b> ▾</div>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--mut2)", fontFamily: "var(--mono)" }}>{total} runs</span>
      </div>

      {loading ? <Loading />
        : error ? <ErrorBox error={error} onRetry={load} />
          : runs.length === 0 && total === 0 ? <div className="z-empty"><div className="ic"><Ic n="log" w={20} /></div><b>No recorded runs yet</b>Once the assistant processes a request, it will appear here.</div>
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
              <div className="z-tfoot" style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span>{from}–{to} of {total}</span>
                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  Rows:
                  <Select w={90} value={String(pageSize)} options={PAGE_SIZES.map(String)}
                    onChange={(v) => { setPageSize(Number(v)); setPage(1); }} />
                </span>
                <span style={{ flex: 1 }} />
                {totalPages > 1 && <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <button className="z-btn g sm" disabled={page <= 1}
                    onClick={() => setPage((p) => Math.max(1, p - 1))} aria-label="Previous page">‹</button>
                  {pageWindow(page, totalPages).map((p, i) => (
                    p === "…"
                      ? <span key={"gap" + i} style={{ padding: "0 4px", color: "var(--mut2)" }}>…</span>
                      : <button key={p} className={"z-btn sm " + (p === page ? "p" : "g")}
                          onClick={() => setPage(p)} aria-current={p === page ? "page" : undefined}>{p}</button>
                  ))}
                  <button className="z-btn g sm" disabled={page >= totalPages}
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))} aria-label="Next page">›</button>
                </span>}
              </div>
            </>}
    </div>
    {openId != null && <Drawer r={detail} loading={detailLoading} error={detailError} onClose={close} />}
  </div>;
}

export default Log;
