import React, { useState, useEffect, useRef } from "react";
import { STAGE_COLOR, STAGE_ORDER, FILLER_COLOR } from "../stageMeta.js";
import { Ic } from "./icons.jsx";

// ── Sparkline ──────────────────────────────────────────
export function Spark({ pts, color, w = 56, h = 22 }) {
  const max = Math.max(...pts), min = Math.min(...pts);
  const d = pts.map((p, i) => `${(i / (pts.length - 1) * w).toFixed(1)},${(h - ((p - min) / (max - min || 1)) * h).toFixed(1)}`).join(" ");
  return <svg width={w} height={h}><polyline points={d} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" /></svg>;
}

// ── Form primitives ────────────────────────────────────
export function Field({ label, hint, children, row }) {
  return <div className={"z-f" + (row ? " row" : "")}>
    <div className={row ? "z-fmeta" : ""}>
      <div className="z-fl"><b>{label}</b></div>
      {hint && <div className="z-fh">{hint}</div>}
    </div>
    <div className={row ? "z-fctl" : ""} style={row ? { display: "flex", alignItems: "center", gap: 10 } : {}}>
      {children}
    </div>
  </div>;
}
// Masked secret input (API keys / tokens / PSK) with a SHOW/HIDE reveal toggle.
export function KeyInput({ value, onChange, placeholder }) {
  const [show, setShow] = useState(false);
  return (
    <div className="z-inp mono">
      <input
        type={show ? "text" : "password"}
        value={value ?? ""}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
      <button className="z-mini" type="button" onClick={() => setShow((s) => !s)}>
        {show ? "HIDE" : "SHOW"}
      </button>
    </div>
  );
}
export function Seg({ options, value, onChange, full }) {
  return <div className={"z-seg" + (full ? " full" : "")}>
    {options.map((o) => <button key={o} className={o === value ? "on" : ""} onClick={() => onChange && onChange(o)}>{o}</button>)}
  </div>;
}
// Segmented control for a small integer scale. Options may carry custom labels
// (option.label shown on the button) while the numeric option.value is stored.
// Optional `poles` renders captions under the first/last segment so the scale is
// self-describing; optional `readout` shows the current "label · value" below.
export function ScaleSeg({ options, value, onChange, poles, readout }) {
  const opts = (options || []).map((o) => (o && typeof o === "object" ? o : { value: o, label: String(o) }));
  const cur = opts.find((o) => o.value === value);
  return <div>
    <div className="z-seg full">
      {opts.map((o) => <button key={String(o.value)} className={o.value === value ? "on" : ""} onClick={() => onChange && onChange(o.value)}>{o.label}</button>)}
    </div>
    {poles && <div style={{ display: "flex", justifyContent: "space-between", marginTop: 5, fontSize: 10.5, color: "var(--mut2)" }}>
      <span>{poles.left}</span><span style={{ textAlign: "right" }}>{poles.right}</span>
    </div>}
    {readout && cur && <div style={{ marginTop: 5, fontSize: 11, color: "var(--mut)", fontFamily: "var(--mono)" }}>{cur.label} · {String(value)}</div>}
  </div>;
}
export function Selector({ label, caption, options, value, onChange }) {
  return <div style={{ marginBottom: 18 }}>
    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 8, gap: 12 }}>
      <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: ".05em", textTransform: "uppercase", color: "var(--ink2)" }}>{label}</span>
      {caption && <span style={{ fontSize: 11.5, color: "var(--mut2)", textAlign: "right" }}>{caption}</span>}
    </div>
    <Seg full options={options} value={value} onChange={onChange} />
  </div>;
}
export function Toggle({ on, onChange, sm }) {
  const toggle = () => onChange && onChange(!on);
  return <span className={"z-toggle" + (sm ? " sm" : "") + (on ? " on" : "")} onClick={toggle}
    role="switch" aria-checked={on} tabIndex={0}
    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } }} />;
}
export function Slider({ min = 0, max = 100, step = 1, value, onChange, fmt }) {
  const ref = useRef(null);
  const pct = ((value - min) / (max - min)) * 100;
  const set = (clientX) => {
    const r = ref.current.getBoundingClientRect();
    let p = (clientX - r.left) / r.width; p = Math.max(0, Math.min(1, p));
    let v = min + p * (max - min); v = Math.round(v / step) * step;
    v = Math.max(min, Math.min(max, +v.toFixed(4)));
    onChange && onChange(v);
  };
  const down = (e) => { set(e.clientX); const mv = (ev) => set(ev.clientX); const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); }; window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up); };
  return <div className="z-slider">
    <div className="z-trk" ref={ref} onPointerDown={down}><i style={{ width: pct + "%" }} /><b style={{ left: pct + "%" }} /></div>
    <span className="z-sval">{fmt ? fmt(value) : value}</span>
  </div>;
}
export function Stepper({ value, onChange, min = -Infinity, max = Infinity, step = 1, unit }) {
  const clamp = (v) => Math.max(min, Math.min(max, v));
  // Integer steppers (capture seconds, ports, TTLs, default step=1) must never emit a
  // fractional value: a typed "1.5" would propagate a float that pydantic int fields reject
  // (POST /api/capture → 400/422). Round to an integer ONLY when step is integer, so genuinely
  // fractional steppers (e.g. step=0.1 for LLM temperature) keep their decimals.
  const norm = (v) => (Number.isInteger(step) ? Math.round(v) : v);
  // Local text mirrors the input while typing so a transient empty/partial value
  // (e.g. "" or "-") doesn't fight the user; commits propagate a clamped number.
  const [text, setText] = useState(String(value));
  useEffect(() => { setText(String(value)); }, [value]);
  const commit = (raw) => {
    const n = parseFloat(raw);
    if (Number.isNaN(n)) { setText(String(value)); return; }  // revert junk/empty, don't propagate
    const c = clamp(norm(n));
    setText(String(c));
    if (c !== value) onChange(c);  // onChange always gets a clamped, step-normalised number
  };
  return <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
    <div className="z-stepper">
      <button onClick={() => onChange(clamp(norm(value - step)))}>−</button>
      <input value={text} inputMode="numeric"
        onChange={(e) => setText(e.target.value.replace(/[^\d.-]/g, ""))}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commit(e.target.value); e.currentTarget.blur(); } }} />
      <button onClick={() => onChange(clamp(norm(value + step)))}>+</button>
    </div>
    {unit && <span style={{ fontSize: 11.5, color: "var(--mut)" }}>{unit}</span>}
  </div>;
}
export function Select({ value, options, onChange, w, itemAction, itemActionBusy, searchable, allowCustom }) {
  const [open, setOpen] = useState(false);
  const [up, setUp] = useState(false); // open upward when the trigger sits near the viewport bottom
  const [query, setQuery] = useState("");
  const ref = useRef(null);
  useEffect(() => { const h = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); }; document.addEventListener("pointerdown", h); return () => document.removeEventListener("pointerdown", h); }, []);
  // The search query never survives an open/close cycle: reopening starts clean.
  useEffect(() => { setQuery(""); }, [open]);
  // Options may be plain strings/numbers OR {value, label} objects (label is shown,
  // value is stored/emitted). Normalize so the rest is uniform; plain arrays keep their
  // existing behavior (value === label).
  const opts = (options || []).map((o) => (o && typeof o === "object" ? o : { value: o, label: String(o) }));
  const selected = opts.find((o) => o.value === value);
  // The search input (and the whole search/custom machinery) renders when EITHER
  // prop is set: `searchable` opts into filtering long lists, while `allowCustom`
  // alone still needs the input as the only way to type an arbitrary value —
  // otherwise short/empty fetched lists would make custom values unreachable.
  // With neither prop the Select stays a plain dropdown (strict no-op).
  const hasSearch = !!(searchable || allowCustom);
  // Case-insensitive filter on label OR value; without the input (or an empty
  // query) the full list renders as before.
  const q = hasSearch ? query.trim().toLowerCase() : "";
  const shown = q ? opts.filter((o) => o.label.toLowerCase().includes(q) || String(o.value).toLowerCase().includes(q)) : opts;
  // Freeform escape hatch: a typed query that matches no option value exactly gets a
  // synthetic "Use ..." row emitting the raw query as the value. The query can only
  // be non-empty when the input is rendered, so no extra hasSearch guard is needed.
  const customRow = !!(allowCustom && query.trim() && !opts.some((o) => String(o.value) === query.trim()));
  const pick = (v) => { onChange && onChange(v); setOpen(false); };
  // Decide drop direction from the live trigger position: flip up only when there
  // isn't enough room below AND there's more room above. Menu height is estimated
  // from the row count, capped at the 240px max-height.
  const wantUp = () => {
    const el = ref.current;
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top;
    const est = Math.min(240, opts.length * 34 + 8);
    return spaceBelow < est + 8 && spaceAbove > spaceBelow;
  };
  const toggle = () => setOpen((o) => { const next = !o; if (next) setUp(wantUp()); return next; });
  return <div ref={ref} style={{ position: "relative", width: w || "100%" }}>
    <div className="z-select" role="button" tabIndex={0} aria-haspopup="listbox" aria-expanded={open}
      onClick={toggle}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } }}>
      {selected ? selected.label : value}<svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M2 4l3.5 3.5L9 4" /></svg></div>
    {open && <div role="listbox" style={{ position: "absolute", left: 0, right: 0, ...(up ? { bottom: "100%", marginBottom: 4 } : { top: "100%", marginTop: 4 }), background: "#fff", border: "1px solid var(--line)", borderRadius: 7, boxShadow: "0 8px 28px rgba(16,24,40,.16)", padding: 4, zIndex: 20, maxHeight: 240, overflowY: "auto" }}>
      {hasSearch && <div style={{ position: "sticky", top: -4, margin: "-4px -4px 4px", padding: 0, background: "#fff", borderBottom: "1px solid var(--line)", zIndex: 1 }}
        onPointerDown={(e) => e.stopPropagation()} onClick={(e) => e.stopPropagation()}>
        <input
          autoFocus
          value={query}
          placeholder="Search…"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              // Priority: a filtered match wins over the custom row. A partial match
              // usually means the user is narrowing down to a known option, so Enter
              // picks the first one; the custom "Use ..." row stays reachable by click.
              if (shown.length > 0) pick(shown[0].value);
              else if (customRow) pick(query.trim());
            } else if (e.key === "Escape") { e.preventDefault(); setOpen(false); }
          }}
          style={{ width: "100%", boxSizing: "border-box", padding: "7px 10px", border: "none", outline: "none", fontSize: 12, fontFamily: "inherit", color: "var(--ink)", background: "transparent" }}
        />
      </div>}
      {shown.map((o) => <div key={String(o.value)} role="option" aria-selected={o.value === value} tabIndex={0} onClick={() => { onChange && onChange(o.value); setOpen(false); }} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onChange && onChange(o.value); setOpen(false); } }} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderRadius: 5, fontSize: 12.5, fontWeight: o.value === value ? 600 : 400, color: o.value === value ? "var(--acc-ink)" : "var(--ink)", background: o.value === value ? "var(--acc-bg)" : "transparent", cursor: "pointer" }} onMouseEnter={(e) => { if (o.value !== value) e.currentTarget.style.background = "var(--panel2)"; }} onMouseLeave={(e) => { if (o.value !== value) e.currentTarget.style.background = "transparent"; }}>
        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{o.label}</span>
        {itemAction && <button type="button" className="z-mini" title="Play this chime" disabled={itemActionBusy != null}
          onClick={(e) => { e.stopPropagation(); itemAction(o.value); }} style={{ flex: "0 0 auto" }}>
          {itemActionBusy === o.value
            ? "…"
            : <svg width="10" height="10" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true"><path d="M2 1l9 5-9 5z" /></svg>}
        </button>}
      </div>)}
      {customRow && <div role="option" aria-selected={false} tabIndex={0}
        onClick={() => pick(query.trim())}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(query.trim()); } }}
        style={{ padding: "7px 10px", borderRadius: 5, fontSize: 12.5, color: "var(--mut)", fontStyle: "italic", cursor: "pointer" }}
        onMouseEnter={(e) => { e.currentTarget.style.background = "var(--panel2)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}>
        Use "{query.trim()}"
      </div>}
    </div>}
  </div>;
}
export function Pill({ tone, children }) { return <span className={"z-pill " + tone}>{children}</span>; }
export function StatusPill({ status }) {
  const map = { online: ["good", "Online"], ok: ["good", "OK"], offline: ["muted", "Offline"], off: ["muted", "Off"], error: ["bad", "Error"] };
  const [tone, label] = map[status] || ["muted", status];
  return <span className={"z-st " + tone}><span className={"z-dot " + (status === "online" || status === "ok" ? "ok" : status === "error" ? "error" : "off")} />{label}</span>;
}

// ── Waterfall ──────────────────────────────────────────
const SC = STAGE_COLOR;
export const total = (t) => Object.values(t).reduce((a, b) => a + b, 0);
export function segsFor(r) {
  if (r.result === "empty") {
    const ms = r.t_total != null ? r.t_total : total(r.t);  // real capture duration, not a mock constant
    return [{ label: "no speech · " + (ms / 1000).toFixed(2) + "s", pct: 100, bg: "#cbd2dd", col: "#8a93a4" }];
  }
  const tot = total(r.t) || 1;
  const arr = STAGE_ORDER.filter((k) => r.t[k] > 0).map((k) => { const pct = r.t[k] / tot * 100; return { label: pct >= 15 ? `${k} ${r.t[k]}` : String(r.t[k]), pct, bg: SC[k], col: SC[k] }; });
  if (r.result === "error") arr.push({ label: "fail", pct: 24, bg: "repeating-linear-gradient(45deg,#dc2626,#dc2626 3px,#fecaca 3px,#fecaca 6px)", col: "#dc2626" });
  return arr;
}
// Left offset (% of the waterfall bar) of the "early filler" marker, or null when
// no filler fired. The bar spans vad→stt→llm→tts normalized to total(r.t); t_filler
// is measured from the start of STT (the right edge of the vad segment), so the
// marker sits at (vad + t_filler) along that same normalized axis. Clamped to [0,100].
export function fillerMarkerPct(r) {
  if (!r || r.t_filler == null || !r.filler_text) return null;
  const t = r.t || {};
  const tot = total(t);
  if (!tot) return null;
  const at = ((t.vad || 0) + r.t_filler) / tot * 100;
  return Math.max(0, Math.min(100, at));
}
export function Waterfall({ r }) {
  const segs = segsFor(r);
  const fpct = fillerMarkerPct(r);
  return <div className="z-wf">
    <div className="z-wfbar">
      {segs.map((s, i) => <span key={i} style={{ width: s.pct + "%", background: s.bg }} />)}
      {fpct != null && <span className="z-wffiller" style={{ left: fpct + "%", background: FILLER_COLOR }}
        title={`🗣 «${r.filler_text}» — early reply at ${(r.t_filler / 1000).toFixed(2)}s`} />}
    </div>
    <div className="z-wfax">{segs.map((s, i) => <span key={i} style={{ width: s.pct + "%", color: s.col }}>{s.label}</span>)}</div>
  </div>;
}

// ── Page chrome ────────────────────────────────────────
export function PageHeader({ title, desc, actions, crumb }) {
  return <div className="z-ph">
    <div>
      {crumb && <div style={{ fontSize: 12, color: "var(--mut2)", marginBottom: 3 }}>{crumb}</div>}
      <h1>{title}</h1>
      {desc && <div className="desc">{desc}</div>}
    </div>
    {actions && <div className="z-ph-actions">{actions}</div>}
  </div>;
}
export function SaveBar({ noTest }) {
  return <div className="z-foot">
    <button className="z-btn p">Save changes</button>
    {!noTest && <button className="z-btn g"><Ic n="test" w={14} />Test connection</button>}
    <span style={{ flex: 1 }} />
    <span className="z-dirty"><s />Unsaved</span>
  </div>;
}

// Live save bar wired to real state: dirty flag, async save, and inline 422 errors.
export function FormSaveBar({ dirty, saving, onSave, errors = [] }) {
  return <>
    {errors.length > 0 && <div className="z-banner" style={{ background: "var(--bad-bg)", border: "1px solid #f3c8c8", color: "#b91c1c", margin: "0 17px 14px", borderRadius: 8 }}>
      <Ic n="restart" w={15} />
      <span><b>Not saved.</b> {errors.join(" · ")}</span>
    </div>}
    <div className="z-foot">
      <button className="z-btn p" disabled={!dirty || saving} onClick={onSave}>{saving ? "Saving…" : "Save changes"}</button>
      <span style={{ flex: 1 }} />
      {dirty && <span className="z-dirty"><s />Unsaved</span>}
    </div>
  </>;
}

export function Loading({ label = "Loading…" }) {
  return <div className="z-empty"><b>{label}</b>Fetching data from the server.</div>;
}
export function ErrorBox({ error, onRetry }) {
  return <div className="z-empty">
    <b>Failed to load</b>
    {(error && (error.message || String(error))) || "Failed to fetch data."}
    {onRetry && <button className="z-btn g sm" style={{ marginTop: 12 }} onClick={onRetry}>Retry</button>}
  </div>;
}
export function Modal({ title, children, footer, onClose }) {
  return <div className="z-modal" onClick={onClose}>
    <div className="z-modal-c" onClick={(e) => e.stopPropagation()}>
      <div className="z-modal-h"><b>{title}</b><button className="z-x" aria-label="Close" style={{ marginLeft: "auto" }} onClick={onClose}><svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg></button></div>
      <div className="z-modal-b">{children}</div>
      {footer && <div className="z-modal-f">{footer}</div>}
    </div>
  </div>;
}
export function Player({ audio, bars = 46 }) {
  if (!audio) return null;
  return <div className="z-player">
    <button className="z-play" aria-label="Play"><svg width="12" height="12" viewBox="0 0 12 12" fill="#fff" aria-hidden="true"><path d="M2 1l9 5-9 5z" /></svg></button>
    <div className="z-wave">{Array.from({ length: bars }).map((_, i) => <i key={i} className={i < bars * 0.4 ? "a" : ""} style={{ height: (8 + Math.abs(Math.sin(i * 0.8)) * 22) + "px" }} />)}</div>
    <span className="tt">0:0{Math.max(1, Math.round(audio.ms / 1000))} · {(audio.bytes / 1024).toFixed(0)} kB · {audio.fmt}</span>
  </div>;
}
export function KV({ k, v }) { return <div className="z-kv"><span className="k">{k}</span><span className="v">{v}</span></div>; }
