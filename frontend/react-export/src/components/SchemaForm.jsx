// Generic renderer for a pydantic JSON Schema object using the existing primitives.
//
//   schema       — a ConfigModel schema OR a core sub-section schema ($ref-resolved
//                  by the caller, or with $defs available on `root`)
//   values       — current field values (plain dict)
//   onChange(f,v)— called with (fieldName, newValue)
//   optionsFor(f)— optional async (field) => string[]  for dynamic lists
//   root         — schema document that holds $defs (defaults to `schema`)
//   skip         — optional array of property names to not render
//
// Field → widget mapping follows settings-storage-design: enum→Seg/Select,
// options:"dynamic"→Select(fetched), slider numbers→Slider, other numbers→Stepper,
// boolean→Toggle, secret-looking strings→masked key input, else text input.
import React, { useEffect, useState } from "react";
import { Field, Seg, Select, Slider, Stepper, Toggle } from "./primitives.jsx";
import { resolve, enumOf, isSecret, humanize } from "../schema.js";

function KeyInput({ value, onChange }) {
  const [show, setShow] = useState(false);
  return (
    <div className="z-inp mono">
      <input
        type={show ? "text" : "password"}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
      />
      <button className="z-mini" type="button" onClick={() => setShow((s) => !s)}>
        {show ? "HIDE" : "SHOW"}
      </button>
    </div>
  );
}

function DynamicSelect({ value, onChange, load }) {
  const [opts, setOpts] = useState(value != null ? [String(value)] : []);
  useEffect(() => {
    let alive = true;
    load()
      .then((list) => {
        if (!alive || !Array.isArray(list)) return;
        // Make sure the current value is selectable even if not in the list.
        const merged = value != null && !list.includes(value) ? [value, ...list] : list;
        setOpts(merged);
      })
      .catch(() => { /* keep current single-option fallback */ });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return <Select value={value} options={opts} onChange={onChange} />;
}

// One property → one <Field> with the right widget.
function SchemaField({ name, node, root, value, onChange, optionsFor }) {
  const r = resolve(node, root);
  const label = node.title || r.title || humanize(name);
  const hint = node.description || r.description;
  // `apply` is the backend-computed action class (reconfig.action_for); only "restart"
  // means a process restart is actually required. Everything else applies live (hot).
  const restart = (node.apply || r.apply) === "restart";
  const finalHint = restart ? (hint ? hint + " · restart required" : "Restart required") : hint;

  const set = (v) => onChange(name, v);
  const enums = enumOf(node, root);
  const dynamic = (node.options || r.options) === "dynamic";
  const widget = node.widget || r.widget;
  const type = r.type;

  let control;
  if (dynamic && optionsFor) {
    control = <DynamicSelect value={value} onChange={set} load={() => optionsFor(name)} />;
  } else if (enums && type !== "number" && type !== "integer" && type !== "boolean") {
    control = enums.length <= 3
      ? <Seg full options={enums} value={value} onChange={set} />
      : <Select value={value} options={enums} onChange={set} />;
  } else if (type === "boolean") {
    control = <Toggle on={!!value} onChange={set} />;
  } else if (type === "integer" || type === "number") {
    const hasRange = r.minimum != null && r.maximum != null;
    const isInt = type === "integer";
    // Fractional fields always step by 0.1 (a step of 1 made e.g. LLM temperature
    // 0.0–2.0 effectively un-editable in the Stepper); integers step by 1.
    const step = r.multipleOf || (isInt ? 1 : 0.1);
    if (hasRange && widget === "slider") {
      const fmt = isInt ? (v) => v : (v) => Number(v).toFixed(1);
      control = (
        <Slider
          min={r.minimum} max={r.maximum} step={step}
          value={value ?? r.minimum} onChange={set} fmt={fmt}
        />
      );
    } else {
      control = (
        <Stepper
          value={value ?? 0} step={step}
          min={r.minimum ?? -Infinity} max={r.maximum ?? Infinity}
          onChange={(v) => set(isInt ? Math.round(v) : v)}
        />
      );
    }
  } else if (widget === "textarea") {
    control = (
      <textarea
        value={value ?? ""}
        onChange={(e) => set(e.target.value)}
        spellCheck={false}
        style={{ width: "100%", minHeight: 90, resize: "vertical", border: "1px solid var(--line)", borderRadius: 8, padding: "10px 12px", fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.6, color: "var(--ink)", outline: "none", background: "var(--panel2)" }}
      />
    );
  } else if (isSecret(name)) {
    control = <KeyInput value={value} onChange={set} />;
  } else {
    control = (
      <div className="z-inp mono">
        <input value={value ?? ""} onChange={(e) => set(e.target.value)} />
      </div>
    );
  }

  // Boolean / stepper render nicely as a "row" field (label left, control right).
  const row = type === "boolean" || (type === "integer" && !(r.minimum != null && r.maximum != null && widget === "slider"));
  return (
    <Field label={label} hint={finalHint} row={row}>
      {control}
    </Field>
  );
}

export default function SchemaForm({ schema, values, onChange, optionsFor, root, skip = [] }) {
  const doc = root || schema;
  const props = (schema && schema.properties) || {};
  const skipSet = new Set(skip);
  return (
    <>
      {Object.entries(props).map(([name, node]) =>
        skipSet.has(name) ? null : (
          <SchemaField
            key={name}
            name={name}
            node={node}
            root={doc}
            value={values ? values[name] : undefined}
            onChange={onChange}
            optionsFor={optionsFor}
          />
        )
      )}
    </>
  );
}

// Does any rendered property carry apply:"restart"? Used to flag the SaveBar.
// Only "restart" requires a real process restart; all other action classes apply live.
export function schemaNeedsRestart(schema, skip = []) {
  if (!schema || !schema.properties) return false;
  const skipSet = new Set(skip);
  return Object.entries(schema.properties).some(
    ([name, node]) => !skipSet.has(name) && node && node.apply === "restart"
  );
}
