import React, { useState, useEffect, useRef } from "react";
import Z from "../data.js";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { Spark, Field, Seg, Selector, Toggle, Slider, Stepper, Select, Pill, StatusPill, Waterfall, segsFor, total, PageHeader, SaveBar, Modal, Player, KV } from "../components/primitives.jsx";
  function Card({ title, sub, children, foot, right }) {
    return <div className="z-card">
      {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}{right}</div>}
      {children !== undefined && <div className="z-card-b">{children}</div>}
      {foot}
    </div>;
  }

  // ── MCP ────────────────────────────────────────────────
  function MCPServer({ s }) {
    const [on, setOn] = useState(s.enabled);
    const [showTools, setShowTools] = useState(false);
    return <div className="z-card" style={{ marginBottom: 14, opacity: on ? 1 : 0.72 }}>
      <div className="z-card-h">
        <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
          <Ic n="mcp" w={17} />
          <div style={{ minWidth: 0 }}>
            <b style={{ display: "block" }}>{s.name}</b>
            <span className="mono" style={{ fontSize: 11, color: "var(--mut)" }}>{s.url}</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginLeft: "auto" }}>
          <StatusPill status={s.enabled ? s.status : "off"} />
          <span style={{ fontSize: 11.5, color: "var(--mut)", fontFamily: "var(--mono)" }}>{s.tools.length} tools</span>
          <Toggle on={on} onChange={setOn} />
        </div>
      </div>
      <div className="z-card-b">
        <Field label="Bearer token" param="(secret · optional)" hint="Empty = no auth." row>
          <div className="z-inp mono sm" style={{ minWidth: 220 }}><input value={s.token || ""} placeholder="— none —" readOnly /></div>
        </Field>
        <Field label="Custom prompt" param="appended to system prompt" hint="Explains these tools to the model.">
          <textarea defaultValue={s.prompt} placeholder="Describe what these tools do and how to use them…" style={{ width: "100%", minHeight: 64, resize: "vertical", border: "1px solid var(--line)", borderRadius: 6, padding: "9px 11px", font: "inherit", fontSize: 12.5, color: "var(--ink2)", outline: "none", lineHeight: 1.5 }} />
        </Field>
        <div className="z-f" style={{ borderBottom: "none" }}>
          <div className="z-fl"><b style={{ fontSize: 12 }}>Advertised tools <span style={{ color: "var(--mut2)", fontWeight: 400 }}>(read-only)</span></b>
            <a style={{ fontSize: 11.5, color: "var(--acc)", cursor: "pointer", fontWeight: 600 }} onClick={() => setShowTools((v) => !v)}>{showTools ? "Hide" : "Show all"}</a></div>
          {s.tools.length === 0 ? <div className="z-fh">No tools — server offline.</div>
            : <div className="z-chiprow">{(showTools ? s.tools : s.tools.slice(0, 5)).map((t) => <span className="z-toolchip" key={t.name} title={t.desc}>{t.name}</span>)}
              {!showTools && s.tools.length > 5 && <span className="z-toolchip" style={{ color: "var(--mut)" }}>+{s.tools.length - 5}</span>}</div>}
        </div>
      </div>
      <div className="z-foot">
        <button className="z-btn g sm"><Ic n="test" w={13} />Test connection</button>
        <span style={{ flex: 1 }} />
        <button className="z-btn d sm"><Ic n="trash" w={13} />Delete</button>
      </div>
    </div>;
  }
  function MCP() {
    const [adding, setAdding] = useState(false);
    return <div className="z-page">
      <PageHeader title="MCP servers" desc="Smart-home integrations the LLM calls for tools. Each server can carry its own prompt."
        actions={<button className="z-btn p" onClick={() => setAdding(true)}><Ic n="add" w={14} />Add server</button>} />
      {Z.mcp.length === 0
        ? <div className="z-card"><div className="z-empty"><div className="ic"><Ic n="mcp" w={20} /></div><b>No MCP servers</b>Smart home is unavailable — the assistant can only talk.</div></div>
        : Z.mcp.map((s) => <MCPServer key={s.id} s={s} />)}
      {adding && <Modal title="Add MCP server" onClose={() => setAdding(false)}
        footer={<><button className="z-btn g" onClick={() => setAdding(false)}>Cancel</button><button className="z-btn p" onClick={() => setAdding(false)}>Add & test</button></>}>
        <Field label="Label" hint="Human-readable name."><div className="z-inp"><input placeholder="e.g. node-red.home" /></div></Field>
        <Field label="Endpoint URL"><div className="z-inp mono"><input placeholder="http://10.0.0.5:8001/mcp" /></div></Field>
        <Field label="Bearer token" hint="Optional — leave empty for no auth."><div className="z-inp mono"><input placeholder="— none —" /></div></Field>
      </Modal>}
    </div>;
  }

  // ── System Prompt ──────────────────────────────────────
  function Prompt() {
    const [text, setText] = useState(Z.prompt.text);
    const [resetting, setResetting] = useState(false);
    return <div className="z-page">
      <PageHeader title="System prompt" crumb="Integrations" desc="Zahar's character, rules and answer format. The placeholder is replaced with live date/time at request time."
        actions={<><button className="z-btn g" onClick={() => setResetting(true)}>Reset to default</button><button className="z-btn g"><Ic n="play" w={13} />Preview</button><button className="z-btn p">Save</button></>} />
      <div className="z-cols wide">
        <Card right={<span className="sub" style={{ marginLeft: "auto" }}>{text.length} chars · {Z.prompt.path}</span>} title="Editor">
          <div style={{ padding: "8px 0" }}>
            <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
              style={{ width: "100%", minHeight: 560, resize: "vertical", border: "1px solid var(--line)", borderRadius: 8, padding: "13px 15px", fontFamily: "var(--mono)", fontSize: 12.5, lineHeight: 1.65, color: "var(--ink)", outline: "none", background: "var(--panel2)" }} />
          </div>
        </Card>
        <div className="z-aside">
          <Card title="The date/time variable">
            <div className="z-info" style={{ padding: "10px 0 12px" }}>
              <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<TDW>>>>>"}</span> is substituted at request time with the current <b>date and time</b>.
            </div>
          </Card>
          <Card title="Preview (substituted)">
            <div className="z-pv-line" style={{ margin: "10px 0", fontSize: 12, lineHeight: 1.6 }}>
              …Текущий контекст: <span className="accent">8 июня 2026, 14:32</span>…
            </div>
          </Card>
          <Card title="Notation">
            <div className="z-info" style={{ padding: "10px 0 12px" }}>Stress marks use <b>«+»</b> before the stressed vowel (e.g. «з+амок»). RUAccent can take over this job — see the <a style={{ color: "var(--acc)", cursor: "pointer" }} onClick={() => nav("ruaccent")}>RUAccent stage</a>.</div>
          </Card>
        </div>
      </div>
      {resetting && <Modal title="Reset prompt to default?" onClose={() => setResetting(false)}
        footer={<><button className="z-btn g" onClick={() => setResetting(false)}>Cancel</button><button className="z-btn d" onClick={() => { setText(Z.prompt.text); setResetting(false); }}>Reset to template</button></>}>
        <div className="z-note" style={{ padding: "8px 0 12px" }}>This replaces the current prompt with <b>templates/default_prompt.md</b>. Your edits will be lost.</div>
      </Modal>}
    </div>;
  }

  // ── Context ────────────────────────────────────────────
  function Context() {
    const [turns, setTurns] = useState(5);
    const [ttl, setTtl] = useState(300);
    return <div className="z-page">
      <PageHeader title="Dialog context" crumb="Integrations" desc="How many past turns the assistant remembers, and how quickly it forgets. Stored separately per speaker." />
      <div>
        <Card title="Memory" foot={<SaveBar noTest />}>
          <Field label="Context depth" param="CONTEXT_MAX_TURNS" hint="How many recent Q&A pairs are remembered and sent to the model." row><Stepper value={turns} min={1} onChange={setTurns} unit="turns" /></Field>
          <Field label="Dialog TTL" param="CONTEXT_TTL_SECONDS" hint="Idle time before a dialog resets (0 = always fresh)." row><Stepper value={ttl} min={0} step={30} onChange={setTtl} unit="s" /></Field>
        </Card>
      </div>
    </div>;
  }

  // ── Devices ────────────────────────────────────────────
  function Devices() {
    const [adding, setAdding] = useState(false);
    return <div className="z-page">
      <PageHeader title="Devices" desc="ESPHome speakers the server connects to. Each name also keys its own dialog context."
        actions={<button className="z-btn p" onClick={() => setAdding(true)}><Ic n="add" w={14} />Add speaker</button>} />
      <Card>
        <table className="z-tbl">
          <thead><tr><th>Name</th><th>Host / IP</th><th>PSK</th><th>Firmware</th><th>Status</th><th></th></tr></thead>
          <tbody>
            {Z.devices.map((d) => <tr key={d.id} style={{ cursor: "default" }}>
              <td style={{ fontWeight: 600 }}>{d.name}</td>
              <td className="mono" style={{ fontSize: 11.5 }}>{d.host}<span style={{ color: "var(--mut2)" }}>:{Z.devicesCommon.ESPHOME_PORT}</span></td>
              <td className="mono" style={{ fontSize: 11.5, color: "var(--mut)" }}>{d.psk}</td>
              <td style={{ color: "var(--mut)", fontSize: 11.5 }}>{d.fw}</td>
              <td><StatusPill status={d.status} /></td>
              <td style={{ textAlign: "right" }}><div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}><button className="z-mini">Test</button><button className="z-mini">Edit</button></div></td>
            </tr>)}
          </tbody>
        </table>
      </Card>
      <div className="z-sl">Common parameters<div className="ln" /></div>
      <Card>
        <Field label="ESPHome API port" param="ESPHOME_PORT" hint="Default 6053." row><Stepper value={6053} onChange={() => {}} /></Field>
        <Field label="Public base URL" param="PUBLIC_BASE_URL" hint="Where speakers download generated audio. If wrong — the speaker won't play the reply.">
          <div className="z-inp mono"><input value={Z.devicesCommon.PUBLIC_BASE_URL} readOnly /><button className="z-mini">TEST</button></div>
        </Field>
        <div className="z-banner warn" style={{ margin: "2px 0 12px" }}><Ic n="devices" w={15} /><span><b>The public base URL</b> is critical: the speaker fetches synthesized audio from it. A wrong value means silence even on a successful run.</span></div>
      </Card>
      {adding && <Modal title="Add speaker" onClose={() => setAdding(false)}
        footer={<><button className="z-btn g" onClick={() => setAdding(false)}>Cancel</button><button className="z-btn p" onClick={() => setAdding(false)}>Add & connect</button></>}>
        <Field label="Name" hint="Unique — also keys the dialog context."><div className="z-inp"><input placeholder="e.g. hallway" /></div></Field>
        <div className="z-row2">
          <Field label="Host / IP"><div className="z-inp mono"><input placeholder="10.0.0.25" /></div></Field>
          <Field label="API port"><div className="z-inp mono"><input value="6053" /></div></Field>
        </div>
        <Field label="PSK" hint="ESPHome API encryption key."><div className="z-inp mono"><input placeholder="base64 key…" /></div></Field>
      </Modal>}
    </div>;
  }

export { MCP, Prompt, Context, Devices };
