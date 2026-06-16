// Generic renderer for a pydantic JSON Schema object using the existing primitives.
//
//   schema       — a ConfigModel schema OR a core sub-section schema ($ref-resolved
//                  by the caller, or with $defs available on `root`)
//   values       — current field values (plain dict)
//   onChange(f,v)— called with (fieldName, newValue)
//   optionsFor(f,q?)— optional async (field, query?) => string[]  for dynamic lists;
//                  `query` is only passed for fields with `search: "remote"`
//                  (server-side catalog search), otherwise the full baseline
//                  list is loaded once and filtered client-side
//   root         — schema document that holds $defs (defaults to `schema`)
//   skip         — optional array of property names to not render
//
// Field → widget mapping follows settings-storage-design: enum→Seg/Select,
// options:"dynamic"→Select(fetched), slider numbers→Slider, other numbers→Stepper,
// boolean→Toggle, explicit `secret:true` strings→masked key input, else text input.
import React, { useEffect, useRef, useState } from "react";
import { Field, KeyInput, ScaleSeg, Seg, Select, Slider, Stepper, Toggle } from "./primitives.jsx";
import { resolve, enumOf, humanize } from "../schema.js";

function DynamicSelect({ value, currentLabel, onChange, load, itemAction, itemActionBusy, allowCustom, remoteSearch }) {
  const norm = (o) => (o && typeof o === "object" ? o : { value: o, label: String(o) });
  // The current value's seed option: its persisted human label when known
  // (currentLabel, e.g. reference_id_label), otherwise the bare id. Shown on the
  // FIRST render so the collapsed control never flickers id->name when the
  // catalog loads later.
  const normCurrent = () => ({ value, label: currentLabel || String(value) });
  const [opts, setOpts] = useState(value != null ? [normCurrent()] : []);
  // Monotonic request counter: only the LATEST load() result may replace the
  // option list, so a slow earlier search can never clobber a faster later one.
  // Bumped (without a request) on unmount to drop any in-flight response.
  const seqRef = useRef(0);
  const debounceRef = useRef(null);
  // Cache of the most recent BASELINE option list (the resolved result of any
  // load with an empty/absent query, i.e. what the backend returns when not
  // searching — own + popular voices, full model catalog, …). Stored as the
  // raw normalized list, BEFORE the keep-current-value merge. An emptied
  // search query means "back to the baseline", so it is restored from this
  // cache instead of refetching on every dropdown close.
  const baselineRef = useRef(null);
  // Re-apply the keep-current-value merge: the current value stays selectable
  // even if the list omits it.
  const withCurrent = (normalized) => {
    const has = value != null && normalized.some((o) => o.value === value);
    return value != null && !has ? [normCurrent(), ...normalized] : normalized;
  };
  const run = (q) => {
    const seq = ++seqRef.current;
    load(q)
      .then((list) => {
        if (seq !== seqRef.current || !Array.isArray(list)) return;
        const normalized = list.map(norm);
        if (!q) baselineRef.current = normalized; // remember the no-query list
        setOpts(withCurrent(normalized));
      })
      .catch(() => { /* keep current options on failure */ });
  };
  useEffect(() => {
    run(); // baseline list (no query) on mount — unchanged for all fields
    return () => {
      seqRef.current += 1; // invalidate in-flight responses
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Remote-search fields: debounced (300 ms) server-side catalog search on each
  // query change. An emptied query (typed clear OR the dropdown closing, which
  // resets the visual query) restores the baseline synchronously from the local
  // cache — no network call, the backend would just return the same list.
  const onQuery = (q) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!q || !q.trim()) {
      // Restore from cache: bump the seq so an in-flight search response can
      // never clobber the restored baseline. If nothing is cached yet, do
      // nothing (and keep the seq!) — the mount-time baseline load is still
      // pending and will populate both the list and the cache.
      if (baselineRef.current) {
        seqRef.current += 1;
        setOpts(withCurrent(baselineRef.current));
      }
      return;
    }
    debounceRef.current = setTimeout(() => run(q), 300);
  };
  // Auto-enable in-dropdown search for long lists (provider model catalogs);
  // short lists (chimes/voices) keep the plain dropdown. Freeform fields
  // (allowCustom) get the search input regardless of list length inside Select —
  // it is the only way to type an arbitrary value when the fetched list is
  // short or empty (e.g. a provider with no api_key returns []). Remote-search
  // fields get the input via `onQuery`, which forces it inside Select.
  // Report the picked option's human label alongside its value, so the parent
  // can persist <field>_label. The label is looked up from the current option
  // list; a freeform/custom typed value (not in the list) falls back to the
  // value itself as its own label.
  const handlePick = (v) => {
    const opt = opts.find((o) => o.value === v);
    onChange(v, opt ? opt.label : String(v));
  };
  return <Select value={value} options={opts} onChange={handlePick} itemAction={itemAction} itemActionBusy={itemActionBusy}
    searchable={opts.length > 10} allowCustom={allowCustom} onQuery={remoteSearch ? onQuery : undefined} />;
}

// One property → one <Field> with the right widget.
function SchemaField({ name, node, root, value, labelValue, onLabelChange, onChange, optionsFor, itemActionFor, itemActionBusy }) {
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

  // `freeform` marks a dynamic select that also accepts arbitrary typed values
  // (e.g. an LLM model id missing from the provider's catalog).
  const freeform = !!(node.freeform || r.freeform);
  // `search: "remote"` marks a dynamic select whose catalog is searched
  // server-side: typing re-queries the provider instead of filtering locally.
  const remoteSearch = (node.search || r.search) === "remote";

  // A string array (e.g. wakeword `keywords`: list[str]) is edited as a single
  // comma/newline-separated text control whose STORED value stays a real array,
  // so pydantic list[str] validation passes on save (a plain text input would
  // write back a string and 422). Detect: resolved type "array" whose item type
  // resolves to "string" (or is unspecified — default to string-array handling).
  const itemsNode = r.items ? resolve(r.items, root) : null;
  const isStringArray = type === "array" && (!itemsNode || itemsNode.type == null || itemsNode.type === "string");

  let control;
  let hintSuffix = "";
  if (isStringArray) {
    // text <-> list mapping: render the array joined by ", "; on edit split the
    // typed text on commas and newlines, trim, drop empties, and store the array.
    const arr = Array.isArray(value) ? value : [];
    hintSuffix = "Comma-separated; one entry per item.";
    control = (
      <div className="z-inp mono">
        <input
          value={arr.join(", ")}
          onChange={(e) => set(e.target.value.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean))}
        />
      </div>
    );
  } else if (dynamic && optionsFor) {
    const itemAction = itemActionFor ? itemActionFor(name) : null;
    // Persist the picked option's human label into the <name>_label companion
    // field (when the field has one) so it renders immediately next load; the
    // value write keeps its existing behavior. No companion -> label ignored.
    const setWithLabel = (v, lbl) => {
      set(v);                                   // writes <name> = value (existing behavior)
      if (onLabelChange && lbl !== undefined) onLabelChange(lbl);  // writes <name>_label = label
    };
    control = <DynamicSelect value={value} currentLabel={labelValue} onChange={setWithLabel}
      load={(q) => optionsFor(name, q)} itemAction={itemAction || undefined}
      itemActionBusy={itemActionBusy} allowCustom={freeform} remoteSearch={remoteSearch} />;
  } else if (segOptions) {
    control = <ScaleSeg
      options={segOptions} value={value} onChange={set}
      poles={poles ? { left: poles[0], right: poles[1] } : undefined}
      readout={readout}
    />;
  } else if (enums && type !== "boolean") {
    // Numeric enums (e.g. mic.channel = Literal[0,1]) render as a Seg/Select too,
    // not a Stepper — an explicit enum always means "pick one of these".
    // Optional `enumLabels` (value -> display string) decorates each Select option
    // (e.g. append a model's RAM footprint) while the STORED value stays the bare
    // enum entry. Only the Select path honors it; Seg (<=3) renders bare values.
    const enumLabels = node.enumLabels || r.enumLabels;
    const selectOptions = enumLabels
      ? enums.map((v) => ({ value: v, label: enumLabels[v] ?? String(v) }))
      : enums;
    control = enums.length <= 3
      ? <Seg full options={enums} value={value} onChange={set} />
      : <Select value={value} options={selectOptions} onChange={set} />;
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
  } else if (r.secret === true) {
    // Masking is EXPLICIT: only fields the backend tags with json_schema_extra
    // {"secret": true} (api_key/token/password/psk/...) render as a masked reveal
    // input. We never guess from the field name, so e.g. wakeword `keywords` or
    // `keyboard_layout` stay plain text.
    control = <KeyInput value={value} onChange={set} />;
  } else {
    control = (
      <div className="z-inp mono">
        <input value={value ?? ""} onChange={(e) => set(e.target.value)} />
      </div>
    );
  }

  // Boolean and numeric steppers (both integer and fractional/float) render nicely as a
  // "row" field (label left, control right). Excluded: sliders (float fields with a
  // min/max range and widget "slider" stay full-width stacked) and numeric enums, which
  // render as a Seg/Select (handled above) and lay out like the other enum selects.
  const row = !segOptions && (type === "boolean" || ((type === "integer" || type === "number") && !enums && !(r.minimum != null && r.maximum != null && widget === "slider")));
  // Append any widget-specific format note (e.g. the array editor's comma hint)
  // to the schema description so the user sees the expected input format.
  const fullHint = [hint, hintSuffix].filter(Boolean).join(" ") || undefined;
  return (
    <Field label={label} hint={fullHint} row={row}>
      {control}
    </Field>
  );
}

export default function SchemaForm({ schema, values, onChange, optionsFor, root, skip = [], itemActionFor, itemActionBusy }) {
  const doc = root || schema;
  const props = (schema && schema.properties) || {};
  const skipSet = new Set(skip);
  return (
    <>
      {Object.entries(props).map(([name, node]) => {
        if (skipSet.has(name)) return null;
        const r = resolve(node, doc);
        if (node.hidden || (r && r.hidden)) return null; // hidden companion fields (e.g. *_label)
        const labelField = name + "_label";
        const hasLabel = Object.prototype.hasOwnProperty.call(props, labelField);
        return (
          <SchemaField
            key={name}
            name={name}
            node={node}
            root={doc}
            value={values ? values[name] : undefined}
            labelValue={hasLabel && values ? values[labelField] : undefined}
            onLabelChange={hasLabel ? (lbl) => onChange(labelField, lbl) : undefined}
            onChange={onChange}
            optionsFor={optionsFor}
            itemActionFor={itemActionFor}
            itemActionBusy={itemActionBusy}
          />
        );
      })}
    </>
  );
}
