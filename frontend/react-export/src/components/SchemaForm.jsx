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
import { Field, KeyInput, ScaleSeg, Seg, Select, Slider, Stepper, Toggle } from "./primitives.jsx";
import { resolve, enumOf, isSecret, humanize } from "../schema.js";

function DynamicSelect({ value, onChange, load }) {
  const norm = (o) => (o && typeof o === "object" ? o : { value: o, label: String(o) });
  const [opts, setOpts] = useState(value != null ? [norm(value)] : []);
  useEffect(() => {
    let alive = true;
    load()
      .then((list) => {
        if (!alive || !Array.isArray(list)) return;
        const normalized = list.map(norm);
        // Keep the current value selectable even if the fetched list omits it.
        const has = value != null && normalized.some((o) => o.value === value);
        const merged = value != null && !has ? [norm(value), ...normalized] : normalized;
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

  const set = (v) => onChange(name, v);
  const enums = enumOf(node, root);
  const dynamic = (node.options || r.options) === "dynamic";
  const widget = node.widget || r.widget;
  const type = r.type;
  const unit = node.unit || r.unit;
  const poles = node.poles || r.poles;
  const choices = node.choices || r.choices;
  const readout = node.readout || r.readout;
  // Labeled/scale segment control: explicit `choices` (value+label per segment), or an
  // enum decorated with `poles` (numeric segments + edge captions). Stores the numeric value.
  const segOptions = choices || (enums && poles ? enums.map((v) => ({ value: v, label: String(v) })) : null);

  let control;
  if (dynamic && optionsFor) {
    control = <DynamicSelect value={value} onChange={set} load={() => optionsFor(name)} />;
  } else if (segOptions) {
    control = <ScaleSeg
      options={segOptions} value={value} onChange={set}
      poles={poles ? { left: poles[0], right: poles[1] } : undefined}
      readout={readout}
    />;
  } else if (enums && type !== "boolean") {
    // Numeric enums (e.g. mic.channel = Literal[0,1]) render as a Seg/Select too,
    // not a Stepper — an explicit enum always means "pick one of these".
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
          unit={unit}
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
  // Numeric enums render as a Seg/Select (handled above), not a stepper, so exclude
  // them here too — they lay out like the other enum selects, not as a row.
  const row = !segOptions && (type === "boolean" || (type === "integer" && !enums && !(r.minimum != null && r.maximum != null && widget === "slider")));
  return (
    <Field label={label} hint={hint} row={row}>
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
