// Catalog editor for the offline-NLU (simple-nlu) provider.
//
// The smart home advertises its devices and actions by technical names over MCP.
// Instead of asking the operator to hand-edit the raw `aliases`/`actions`
// textareas, this editor discovers every enum slot live from the tool catalog and
// offers one chip-input per unique enum VALUE: devices/scenes get spoken NAMES
// (→ `aliases`), state/command slots get VERBS (→ `actions`). Numeric slots
// (brightness/temperature) have no dictionary and render as read-only info cards.
//
// The source of truth stays `draft.aliases` / `draft.actions`: everything visible
// is DERIVED from those two fields each render, and every edit re-serializes the
// whole field through onChange — so this editor and the SchemaForm textareas stay
// in sync. Only transient view state (slot-type overrides, per-group collapse,
// search, active-source set, filter) lives in local React state.
//
// Ported from the standalone designer mockup (entities.jsx). Product adaptations:
// the demo state switch and the "new device" badge / "New" filter were dropped
// (no backend signal for them); the page chrome (PageHeader / Save bar) is gone
// because this renders as a section inside the LLM stage page.
import React, { useState, useRef, useEffect, useMemo } from "react";
import { Ic } from "./icons.jsx";
import { getTools } from "../api.js";
import {
  parseActions,
  parseAliasText,
  serializeAliasText,
  serializeActions,
  enumSlotsWithSource,
  numberSlotsFromTools,
  classifySlotKind,
  groupSlots,
  splitWords,
  joinWords,
} from "../nluAliases.js";

// ── chip input ─────────────────────────────────────────────
// ×-removable tags + a trailing text input. Comma or Enter commits the typed
// draft (splitting on commas, de-duping against existing words); Backspace on an
// empty input removes the last chip. `words` is the controlled array; `onChange`
// receives the next array. The empty state styles the box as a dashed "to fill".
function Chips({ words, tone, onChange, placeholder }) {
  const [draft, setDraft] = useState("");
  const ref = useRef(null);
  const commit = (raw) => {
    const parts = splitWords(raw);
    if (parts.length) onChange([...words, ...parts.filter((p) => !words.includes(p))]);
    setDraft("");
  };
  const key = (e) => {
    if ((e.key === "Enter" || e.key === ",") && draft.trim()) {
      e.preventDefault();
      commit(draft);
    } else if (e.key === "Backspace" && !draft && words.length) {
      onChange(words.slice(0, -1));
    }
  };
  const empty = words.length === 0;
  return (
    <div className={"e-chips" + (empty ? " empty" : "")} onClick={() => ref.current && ref.current.focus()}>
      {words.map((w, i) => (
        <span key={i} className={"e-chip " + tone}>
          {w}
          <span
            className="x"
            onClick={(e) => {
              e.stopPropagation();
              onChange(words.filter((_, j) => j !== i));
            }}
          >
            ×
          </span>
        </span>
      ))}
      <input
        ref={ref}
        value={draft}
        placeholder={empty ? placeholder : ""}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={key}
        onBlur={() => draft.trim() && commit(draft)}
      />
    </div>
  );
}

// ── type override dropdown ─────────────────────────────────
// A quiet per-group control that flips its classification entity↔action. The
// heuristic in classifySlotKind is right by default; this is the rare manual fix.
// `kind` is "entity" | "action"; onChange("entity"|"action") applies it.
const TYPE_LABEL = { entity: "Device", action: "Action" };
function TypeOverride({ kind, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const h = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("pointerdown", h);
    return () => document.removeEventListener("pointerdown", h);
  }, []);
  return (
    <div className="e-typ" ref={ref}>
      <button
        className="e-typ-btn"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        title="Slot type — rare manual override"
      >
        <span className="tg">type:</span>
        <b>{TYPE_LABEL[kind]}</b>
        <svg width="9" height="9" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.7">
          <path d="M2 4l3.5 3.5L9 4" />
        </svg>
      </button>
      {open && (
        <div className="e-typ-menu" onClick={(e) => e.stopPropagation()}>
          <div className="hd">What fills this slot</div>
          {[
            ["entity", "lex", "Object names: «свет в зале, люстра»"],
            ["action", "bolt", "Command verbs: «включи, зажги, вруби»"],
          ].map(([t, ic, d]) => (
            <div
              key={t}
              className={"e-typ-opt" + (kind === t ? " on" : "")}
              onClick={() => {
                onChange(t);
                setOpen(false);
              }}
            >
              <div className="ico">
                <Ic n={ic} w={14} />
              </div>
              <div className="tx">
                <b>{TYPE_LABEL[t]}</b>
                <span>{d}</span>
              </div>
            </div>
          ))}
          <div className="e-typ-note">
            The type is usually detected automatically from the slot name. Override only if it was detected wrong — it
            applies to the whole slot.
          </div>
        </div>
      )}
    </div>
  );
}

// ── one group block (collapsible) ──────────────────────────
// `group` is a display group from groupSlots: {key, kind, slotName, tools, values}.
// `wordsFor(value)` returns the current word array for a value; `setWords(value,
// words)` persists it. `vis` is the query-filtered subset of values to render.
function GroupBlock({ group, scene, vis, open, onToggle, wordsFor, setWords, onRetype }) {
  const isAction = group.kind === "action";
  const tone = scene ? "scene" : isAction ? "cmd" : "dev";
  const ph = isAction ? "command verbs, comma-separated" : "Russian names, comma-separated";
  const filled = group.values.filter((v) => wordsFor(v).length).length;
  const total = group.values.length;
  const missing = group.values.filter((v) => !wordsFor(v).length);
  // A partially-filled ACTION slot silently breaks the command (the backend only
  // treats a slot as a state slot when EVERY enum value is an action name).
  const partial = isAction && missing.length > 0 && filled > 0;
  // Action groups merge identical value sets across tools; show where they apply.
  const usedIn = isAction && group.tools.length > 1 ? group.tools : null;

  return (
    <div className="e-slot">
      <div className="e-slot-h" onClick={onToggle}>
        <svg
          className={"e-chev" + (open ? " open" : "")}
          width="11"
          height="11"
          viewBox="0 0 11 11"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.9"
        >
          <path d="M4 2l4 3.5L4 9" />
        </svg>
        <span className="e-slot-id">
          {group.tools[0]}
          <s> · {group.slotName}</s>
        </span>
        {usedIn && (
          <span className="e-used">
            used in:
            {usedIn.map((u) => (
              <i key={u}>{u}</i>
            ))}
          </span>
        )}
        <div className="e-slot-r">
          {partial ? (
            <span className="e-miss">
              <s />
              {missing.length} missing verbs
            </span>
          ) : (
            <span className="e-slot-meta" style={{ fontFamily: "var(--mono)" }}>
              {filled}/{total}
            </span>
          )}
          <TypeOverride kind={group.kind} onChange={onRetype} />
        </div>
      </div>
      {open && (
        <div className="e-rows">
          {partial && (
            <div className="e-warn">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
              </svg>
              <span>
                This action slot is incomplete. The command won't fire until every value has verbs — still empty:{" "}
                {missing
                  .map((m) => <code key={m}>{m}</code>)
                  .reduce((a, b) => [a, " ", b])}
                .
              </span>
            </div>
          )}
          {vis.map((v) => {
            const words = wordsFor(v);
            const isEmpty = words.length === 0;
            return (
              <div className="e-row" key={v}>
                <div className={"e-rid" + (isEmpty ? " is-empty" : "")}>
                  <span className={"mk " + (isEmpty ? "empty" : "full")} />
                  <code title={v}>{v}</code>
                </div>
                <Chips words={words} tone={tone} placeholder={ph} onChange={(w) => setWords(v, w)} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── zone (devices/scenes OR commands) ──────────────────────
function Zone({ kind, title, sub, icon, groups, total, children }) {
  if (!groups.length) return null;
  return (
    <div className={"e-zone " + kind}>
      <div className="e-zone-h">
        <div className="e-zicon">
          <Ic n={icon} w={18} />
        </div>
        <div className="tt">
          <b>{title}</b>
          <span>{sub}</span>
        </div>
        <span className="e-zcount">
          {/* filled / total values, computed by the caller via `total` = {filled,all} */}
          {total.filled}
          <s>/{total.all} values</s>
        </span>
      </div>
      <div className="e-zbody">{children}</div>
    </div>
  );
}

// ── numeric info block ─────────────────────────────────────
// brightness/temperature: no dictionary, set by a number in the spoken command.
// A `string`-typed numeric slot is off-capable ("number / off").
function NumBlock({ slots, showServer }) {
  if (!slots.length) return null;
  return (
    <div className="e-zbody" style={{ marginTop: 14 }}>
      {slots.map((s) => {
        const offCapable = s.type === "string";
        return (
          <div className="e-num" key={`${s.server}.${s.tool}.${s.slot}`}>
            <div className="ic">
              <Ic n="hash" w={15} />
            </div>
            <div className="tx">
              <b>
                {s.tool}
                <s> · {s.slot}</s>
              </b>
              <span>
                Set by a number in the command itself{offCapable ? " (or “off”)" : ""} — e.g. «поставь ночник на 30». No
                dictionary needed.{showServer && s.server ? "  ·  " + s.server : ""}
              </span>
            </div>
            <span className="tag">{offCapable ? "number / off" : "number"}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── sources bar ────────────────────────────────────────────
// Only sources that own ≥1 enum slot. Offline sources are disabled/excluded.
function Sources({ servers, counts, active, online, onToggle }) {
  return (
    <div className="e-src">
      <div className="e-src-lb">
        <Ic n="mcp" w={14} />
        Sources
      </div>
      <div className="e-srvs">
        {servers.map((id) => {
          const offline = !online.has(id);
          const on = active.has(id) && !offline;
          return (
            <div
              key={id}
              className={"e-srv" + (offline ? " disabled" : on ? " on" : " off")}
              onClick={() => !offline && onToggle(id)}
              title={
                offline
                  ? "Server offline — its tools aren't advertised"
                  : on
                  ? "Active — entities shown"
                  : "Hidden from the list"
              }
            >
              <span className="e-cb">
                {on && (
                  <svg
                    width="11"
                    height="11"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M3 8l3.5 3.5L13 4" />
                  </svg>
                )}
              </span>
              <span className="nm">{id}</span>
              <span className={"dot " + (offline ? "off" : "ok")} />
              {offline ? <span className="ofl">offline</span> : <span className="ct">{counts[id] || 0}</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── loading skeleton ───────────────────────────────────────
function Loading() {
  return (
    <div className="e-zbody" style={{ marginTop: 20 }}>
      {Array.from({ length: 7 }).map((_, i) => (
        <div
          key={i}
          style={{
            display: "grid",
            gridTemplateColumns: "212px 1fr",
            gap: 16,
            padding: "14px 16px",
            borderTop: i ? "1px solid var(--line2)" : "none",
          }}
        >
          <div className="e-sk" style={{ width: 120 + (i % 3) * 30, marginTop: 6 }} />
          <div className="e-sk" style={{ height: 38, borderRadius: 6, width: 60 + ((i * 13) % 38) + "%" }} />
        </div>
      ))}
    </div>
  );
}

// ── offline / empty card ───────────────────────────────────
function Offline({ onRetry }) {
  return (
    <div className="z-card" style={{ marginTop: 20 }}>
      <div className="z-empty">
        <div className="ic">
          <Ic n="mcp" w={22} />
        </div>
        <b>No devices found</b>
        The tool list didn't arrive — the smart-home server (MCP) looks offline.
        <br />
        Check the connection and retry.
        <div style={{ marginTop: 16, display: "flex", gap: 9, justifyContent: "center" }}>
          <button className="z-btn p" onClick={onRetry}>
            <Ic n="restart" w={13} />
            Retry
          </button>
        </div>
      </div>
    </div>
  );
}

// ── editor ─────────────────────────────────────────────────
export default function NluCatalogEditor({ draft, onChange }) {
  // Live catalog fetch state. `nonce` is bumped by Retry to refetch.
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [nonce, setNonce] = useState(0);

  // Transient view state (never persisted): slot-type overrides keyed by GROUP key,
  // per-group collapse (group key → bool, true = collapsed), search, active source
  // set, and the All/Empty filter.
  const [overrides, setOverrides] = useState({});
  const [collapsed, setCollapsed] = useState({});
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(null); // Set of active source ids, or null = "all online-with-slots"
  const [filter, setFilter] = useState("all"); // "all" | "empty"

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setFailed(false);
    (async () => {
      try {
        const data = await getTools();
        if (alive) setSources((data && data.sources) || []);
      } catch {
        if (alive) {
          setSources([]);
          setFailed(true);
        }
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [nonce]);

  // Derive the field maps from props each render — no duplicate text state.
  const { map: nameToVerbs, extraLines: actionExtras } = parseActions(draft.actions || "");
  const actionNames = Object.keys(nameToVerbs);
  const { map: aliasMap, extraLines: aliasExtras } = parseAliasText(draft.aliases || "");

  // Online source ids: the backend reports `online: false` only for a source that
  // advertised zero tools (so it owns no slots and never reaches the bar anyway);
  // we still honour the flag for the offline styling. Absent flag defaults online.
  const onlineIds = useMemo(
    () => new Set((sources || []).filter((s) => s.online !== false).map((s) => s.id)),
    [sources]
  );
  const allEnumSlots = useMemo(() => enumSlotsWithSource(sources), [sources]);
  const allNumberSlots = useMemo(() => numberSlotsFromTools(sources), [sources]);
  // Source ids that own at least one enum slot, in first-seen order.
  const slotServers = useMemo(() => {
    const out = [];
    for (const s of allEnumSlots) if (s.server != null && !out.includes(s.server)) out.push(s.server);
    return out;
  }, [allEnumSlots]);

  // Default active set: every online source that owns slots. `active === null`
  // means "use the default" so a fresh catalog auto-selects without an effect.
  const effectiveActive = useMemo(() => {
    if (active) return active;
    return new Set(slotServers.filter((id) => onlineIds.has(id)));
  }, [active, slotServers, onlineIds]);
  const toggleSource = (id) => {
    setActive((prev) => {
      const base = prev || new Set(slotServers.filter((s) => onlineIds.has(s)));
      const next = new Set(base);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };
  const srvOn = (server) => effectiveActive.has(server);

  // classifySlotKind, but resolving a GROUP-keyed override. A slot can belong to
  // either an entity group (key `tool.slot`) or an action group (key
  // sorted(values).join); we honour an override on EITHER key so a flip in one
  // direction is seen when computing the group in the other.
  const kindOf = (slot) => {
    const entityKey = `${slot.tool}.${slot.slot}`;
    const actionKey = [...slot.values].sort().join(" ");
    const override = overrides[entityKey] ?? overrides[actionKey];
    return classifySlotKind(slot.slot, slot.values, actionNames, override);
  };

  // Build the display groups. Split by kind for SERIALIZATION over ALL groups
  // (independent of the active-source view filter, so toggling a source off never
  // drops its aliases/verbs from the field), and a separate VISIBLE subset for
  // rendering + counts. The serializer accepts these directly — each group carries
  // a `values` array, the shape serializeAliasText/serializeActions expect.
  const groups = useMemo(() => groupSlots(allEnumSlots, kindOf), [allEnumSlots, overrides, draft.actions]); // eslint-disable-line react-hooks/exhaustive-deps
  const allEntityGroups = groups.filter((g) => g.kind === "entity");
  const allActionGroups = groups.filter((g) => g.kind === "action");
  const visibleGroups = groups.filter((g) => g.servers.some(srvOn));
  const entityGroups = visibleGroups.filter((g) => g.kind === "entity");
  const actionGroups = visibleGroups.filter((g) => g.kind === "action");
  const numberSlots = allNumberSlots.filter((s) => srvOn(s.server));

  // catalogValues: every enum value across ALL slots — keeps an action value out
  // of `aliases` (and vice versa) when serializing.
  const catalogValues = useMemo(() => new Set(allEnumSlots.flatMap((s) => s.values)), [allEnumSlots]);

  // ── word accessors over the derived maps ──
  const entityWords = (value) => splitWords(aliasMap[value] || "");
  const actionWords = (value) => splitWords(nameToVerbs[value] || "");

  // ── persistence: re-serialize the WHOLE field on every edit ──
  const setEntityWords = (value, words) => {
    const text = joinWords(words); // empty array -> "" -> value dropped from aliases
    onChange("aliases", serializeAliasText(allEntityGroups, { ...aliasMap, [value]: text }, aliasExtras, catalogValues));
  };
  const setActionWords = (value, words) => {
    const text = joinWords(words);
    onChange("actions", serializeActions(allActionGroups, { ...nameToVerbs, [value]: text }, actionExtras));
  };

  // ── flip a group's kind (entity↔action) ──
  // Set the override on the group's CURRENT key AND strip the group's values from
  // the OLD field, so a value never lives in both `aliases` and `actions`.
  const retypeGroup = (group, next) => {
    if (next === group.kind) return;
    if (group.kind === "entity") {
      // Was entity → strip from aliases, re-serialize without these values.
      const cleaned = { ...aliasMap };
      let changed = false;
      for (const v of group.values)
        if (v in cleaned) {
          delete cleaned[v];
          changed = true;
        }
      if (changed)
        onChange(
          "aliases",
          serializeAliasText(
            allEntityGroups.filter((g) => g.key !== group.key),
            cleaned,
            aliasExtras,
            catalogValues
          )
        );
    } else {
      // Was action → strip from actions.
      const cleaned = { ...nameToVerbs };
      let changed = false;
      for (const v of group.values)
        if (v in cleaned) {
          delete cleaned[v];
          changed = true;
        }
      if (changed)
        onChange(
          "actions",
          serializeActions(
            allActionGroups.filter((g) => g.key !== group.key),
            cleaned,
            actionExtras
          )
        );
    }
    setOverrides((prev) => ({ ...prev, [group.key]: next }));
  };

  // ── search + filter ──
  const q = query.trim().toLowerCase();
  const matchesQuery = (group) => {
    if (!q) return group.values;
    const wordsFor = group.kind === "action" ? actionWords : entityWords;
    return group.values.filter(
      (v) => v.toLowerCase().includes(q) || wordsFor(v).some((w) => w.toLowerCase().includes(q))
    );
  };
  // A group passes the filter when it has ≥1 visible value (after search) and,
  // for the Empty filter, ≥1 value with no words.
  const filterValues = (group, vis) => {
    if (filter === "empty") {
      const wordsFor = group.kind === "action" ? actionWords : entityWords;
      return vis.filter((v) => wordsFor(v).length === 0);
    }
    return vis;
  };

  // Render a list of groups into a zone body.
  const renderGroups = (gs) =>
    gs
      .map((g) => {
        const vis = filterValues(g, matchesQuery(g));
        if (vis.length === 0) return null;
        const open = !collapsed[g.key];
        const wordsFor = g.kind === "action" ? actionWords : entityWords;
        const setWords = g.kind === "action" ? setActionWords : setEntityWords;
        return (
          <GroupBlock
            key={g.key}
            group={g}
            scene={g.slotName === "scene"}
            vis={vis}
            open={open}
            onToggle={() => setCollapsed((prev) => ({ ...prev, [g.key]: !prev[g.key] }))}
            wordsFor={wordsFor}
            setWords={setWords}
            onRetype={(next) => retypeGroup(g, next)}
          />
        );
      })
      .filter(Boolean);

  // ── progress + counts ──
  const entityValues = entityGroups.flatMap((g) => g.values);
  const actionValues = actionGroups.flatMap((g) => g.values);
  const allValues = [...entityValues, ...actionValues];
  const totalVals = allValues.length;
  const filledEntity = entityValues.filter((v) => entityWords(v).length).length;
  const filledAction = actionValues.filter((v) => actionWords(v).length).length;
  const filled = filledEntity + filledAction;
  const devPct = totalVals ? (filledEntity / totalVals) * 100 : 0;
  const cmdPct = totalVals ? (filledAction / totalVals) * 100 : 0;
  const emptyCount =
    entityValues.filter((v) => entityWords(v).length === 0).length +
    actionValues.filter((v) => actionWords(v).length === 0).length;

  // Per-source enum value counts for the Sources bar.
  const counts = {};
  for (const s of allEnumSlots) counts[s.server] = (counts[s.server] || 0) + s.values.length;

  // Whether the Sources bar shows at all (hide when only one source owns slots).
  const showSourcesBar = slotServers.length > 1;
  const showServerTag = effectiveActive.size > 1;

  // ── states ──
  if (loading) {
    return (
      <div className="z-card" style={{ padding: 16 }}>
        <Loading />
      </div>
    );
  }
  // Failure OR zero sources with slots → offline card.
  if (failed || slotServers.length === 0) {
    return <Offline onRetry={() => setNonce((n) => n + 1)} />;
  }

  const renderedEntities = renderGroups(entityGroups);
  const renderedActions = renderGroups(actionGroups);
  // The numeric block only shows under the "all" filter, so it only counts toward
  // "something is on screen" then. Nothing visible at all → the empty card.
  const numVisible = filter === "all" && numberSlots.length > 0;
  const nothing = renderedEntities.length === 0 && renderedActions.length === 0 && !numVisible;

  return (
    <div className="z-card" style={{ padding: 16, background: "transparent", border: "none", boxShadow: "none" }}>
      {showSourcesBar && (
        <Sources
          servers={slotServers}
          counts={counts}
          active={effectiveActive}
          online={onlineIds}
          onToggle={toggleSource}
        />
      )}

      {/* sticky toolbar: progress + search + filters */}
      <div className="e-bar">
        <div className="e-bar-in">
          <div className="e-prog">
            <div className="lbl">
              <b>
                {filled}
                <s>/{totalVals}</s>
              </b>
              <em>values filled</em>
            </div>
            <div className="e-track">
              <i style={{ width: devPct + "%" }} />
              <i className="cmd" style={{ width: cmdPct + "%" }} />
            </div>
          </div>
          <div className="e-bar-sep" />
          <div className="e-search">
            <Ic n="search" w={14} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by id or Russian word…"
            />
          </div>
          <div className="e-filters">
            {[
              ["all", "All", null],
              ["empty", "Empty", emptyCount],
            ].map(([k, label, ct]) => (
              <div
                key={k}
                className={"e-fchip" + (k === "empty" ? " warnm" : "") + (filter === k ? " on" : "")}
                onClick={() => setFilter(k)}
              >
                {label}
                {ct != null && <span className="ct">{ct}</span>}
              </div>
            ))}
          </div>
        </div>
      </div>

      <Zone
        kind="dev"
        icon="lex"
        title="Devices & scenes"
        sub="what you call these things out loud"
        groups={entityGroups}
        total={{ filled: filledEntity, all: entityValues.length }}
      >
        {renderedEntities}
      </Zone>
      <Zone
        kind="cmd"
        icon="bolt"
        title="Commands"
        sub="the verbs you say — identical actions shown once"
        groups={actionGroups}
        total={{ filled: filledAction, all: actionValues.length }}
      >
        {renderedActions}
      </Zone>
      {filter === "all" && <NumBlock slots={numberSlots} showServer={showServerTag} />}

      {nothing && (
        <div className="z-card" style={{ marginTop: 20 }}>
          <div className="z-empty">
            <b>Nothing to show</b>
            {effectiveActive.size === 0
              ? "No sources selected — enable an MCP server above."
              : "No slots match the current filter."}
          </div>
        </div>
      )}
    </div>
  );
}
