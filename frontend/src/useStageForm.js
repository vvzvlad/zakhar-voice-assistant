// Reusable editing state for a schema-driven form: holds a draft, tracks dirty
// fields, saves via a patch builder, and surfaces 422 validation errors.
import { useEffect, useMemo, useState, useCallback } from "react";

export function useStageForm(initialValues, buildPatch, patch) {
  const [draft, setDraft] = useState(initialValues || {});
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);

  // Re-seed when the upstream values change (e.g. after a successful save/reload
  // or a provider switch). JSON compare keeps it cheap and avoids clobbering edits
  // only when the source truly changed.
  const sig = JSON.stringify(initialValues || {});
  useEffect(() => {
    setDraft(initialValues || {});
    setErr(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const onChange = useCallback((field, value) => {
    setDraft((d) => ({ ...d, [field]: value }));
  }, []);

  const dirty = useMemo(
    () => JSON.stringify(draft) !== sig,
    [draft, sig]
  );

  const save = useCallback(async () => {
    setSaving(true);
    setErr(null);
    try {
      await patch(buildPatch(draft));
    } catch (e) {
      setErr(e);
    } finally {
      setSaving(false);
    }
  }, [draft, buildPatch, patch]);

  return { draft, onChange, dirty, saving, err, save, setErr };
}

// Format an ApiError (422 carries pydantic detail[]) into short lines for display.
export function errorLines(err) {
  if (!err) return [];
  if (Array.isArray(err.detail) && err.detail.length) {
    return err.detail.map((d) => {
      const loc = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
      return loc ? `${loc}: ${d.msg}` : d.msg;
    });
  }
  return [err.message || String(err)];
}
