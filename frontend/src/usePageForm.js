// Editing state for a page whose cards and provider tabs save together: each
// part keeps its own draft, and ONE Save sends the patches of every dirty part
// merged into ONE PATCH. Drafts live here, not in the forms that render them,
// so a provider tab switch (which remounts its form) keeps unsaved edits.
//
// A part is { key, values, toPatch(draft) }. An edit remembers the values it
// started from; once they change upstream (another save or a reload refreshed
// the catalog), the part shows the new values again. While a save is in flight
// the page's inputs are disabled (FormPage) and edits are ignored, so a
// successful save drops the edits of every part it sent: they show the
// refreshed values. A failed save keeps them for another try.
import { useState } from "react";

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

// Combine two config patches the way PATCH /api/config applies them: objects
// merge, everything else (lists included) replaces.
export function mergePatch(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) out[k] = isObject(v) && isObject(out[k]) ? mergePatch(out[k], v) : v;
  return out;
}

export function usePageForm(parts, patch) {
  const [edits, setEdits] = useState({}); // key -> { base, draft }
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);

  const state = {};
  for (const p of parts) {
    const values = p.values || {};
    const base = JSON.stringify(values);
    const e = edits[p.key];
    const draft = e && e.base === base ? e.draft : values;
    state[p.key] = { ...p, values, base, draft, dirty: JSON.stringify(draft) !== base };
  }
  const dirtyParts = Object.values(state).filter((p) => p.dirty);

  const draftOf = (key) => state[key].draft;
  const isDirty = (key) => state[key].dirty;
  // SchemaForm-style (field, value) setter for one part.
  const onChangeOf = (key) => (field, value) => {
    if (saving) return;
    const { base, values } = state[key];
    setEdits((prev) => {
      const e = prev[key];
      const cur = e && e.base === base ? e.draft : values;
      return { ...prev, [key]: { base, draft: { ...cur, [field]: value } } };
    });
  };

  const save = async () => {
    const body = dirtyParts.reduce((acc, p) => mergePatch(acc, p.toPatch(p.draft)), {});
    setSaving(true);
    setErr(null);
    try {
      await patch(body);
      setEdits((prev) => {
        const next = { ...prev };
        for (const p of dirtyParts) delete next[p.key];
        return next;
      });
    } catch (e) {
      setErr(e);
    } finally {
      setSaving(false);
    }
  };

  return { draftOf, onChangeOf, isDirty, dirty: dirtyParts.length > 0, saving, err, save };
}
