// Unit tests for the run-toast pure helpers (src/toastModel.js).
import { describe, it, expect } from "vitest";
import { shouldNotify, toastFromRun, pushToast, QUIET_PAGES } from "../toastModel.js";

const finalized = { id: 7, device: "kitchen", result: "ok", stt_text: "turn on the light", t_total: 1234 };
const liveRun = { id: null, live: 1, device: "kitchen", stt_text: "", t_total: null };

describe("shouldNotify", () => {
  it("is true for live in-progress snapshots on non-quiet pages", () => {
    expect(shouldNotify({ ...finalized, id: null, live: true }, "stt")).toBe(true);
    expect(shouldNotify({ ...finalized, live: 1 }, "stt")).toBe(true);
  });

  it("is true for rows without an id (not finalized yet)", () => {
    expect(shouldNotify({ ...finalized, id: undefined }, "stt")).toBe(true);
    expect(shouldNotify({ ...finalized, id: null }, "stt")).toBe(true);
  });

  it("is false on the pages that already show live runs", () => {
    expect(shouldNotify(finalized, "dashboard")).toBe(false);
    expect(shouldNotify(finalized, "log")).toBe(false);
    expect(shouldNotify(liveRun, "dashboard")).toBe(false);
    expect(shouldNotify(liveRun, "log")).toBe(false);
  });

  it("is true for a finalized row on any other page", () => {
    expect(shouldNotify(finalized, "stt")).toBe(true);
  });

  it("is false for a missing run", () => {
    expect(shouldNotify(null, "stt")).toBe(false);
  });

  it("quiet pages set covers exactly dashboard and log", () => {
    expect([...QUIET_PAGES].sort()).toEqual(["dashboard", "log"]);
  });
});

describe("toastFromRun", () => {
  it("maps a known result via RESULT_META (ok -> OK/good) and keys by id", () => {
    const t = toastFromRun(finalized);
    expect(t).toEqual({
      key: 7, id: 7, live: false, device: "kitchen", label: "OK", tone: "good",
      text: "turn on the light", totalMs: 1234,
    });
  });

  it("keys a live row as 'live:<device>' with a null id", () => {
    const t = toastFromRun(liveRun);
    expect(t.key).toBe("live:kitchen");
    expect(t.id).toBeNull();
    expect(t.live).toBe(true);
  });

  it("labels a live row 'Running' with a muted tone", () => {
    const t = toastFromRun(liveRun);
    expect(t.label).toBe("Running");
    expect(t.tone).toBe("muted");
  });

  it("substitutes '(in progress)' when a live row has no stt_text yet", () => {
    expect(toastFromRun(liveRun).text).toBe("(in progress)");
    expect(toastFromRun({ ...liveRun, stt_text: "turn on" }).text).toBe("turn on");
  });

  it("falls back to a muted tone with the raw label for unknown results", () => {
    const t = toastFromRun({ ...finalized, result: "weird" });
    expect(t.label).toBe("weird");
    expect(t.tone).toBe("muted");
  });

  it("substitutes '(silence)' when a finalized row has no stt_text", () => {
    expect(toastFromRun({ ...finalized, stt_text: "" }).text).toBe("(silence)");
    expect(toastFromRun({ ...finalized, stt_text: undefined }).text).toBe("(silence)");
  });

  it("substitutes '—' when device is missing", () => {
    expect(toastFromRun({ ...finalized, device: undefined }).device).toBe("—");
  });

  it("keeps totalMs null when t_total is missing", () => {
    expect(toastFromRun({ ...finalized, t_total: undefined }).totalMs).toBeNull();
    expect(toastFromRun({ ...finalized, t_total: null }).totalMs).toBeNull();
  });

  it("preserves a zero t_total (not coerced to null)", () => {
    expect(toastFromRun({ ...finalized, t_total: 0 }).totalMs).toBe(0);
  });
});

describe("pushToast", () => {
  const mk = (key) => ({ key, device: "d" + key });

  it("prepends the new toast", () => {
    const out = pushToast([mk(1)], mk(2));
    expect(out.map((t) => t.key)).toEqual([2, 1]);
  });

  it("dedupes by key, replacing the old entry with the new one", () => {
    const out = pushToast([mk(1), mk(2)], { key: 2, device: "fresh" });
    expect(out.map((t) => t.key)).toEqual([2, 1]);
    expect(out[0].device).toBe("fresh");
  });

  it("upserts a live toast in place by its 'live:<device>' key", () => {
    const live1 = toastFromRun(liveRun);
    const live2 = toastFromRun({ ...liveRun, stt_text: "turn on" });
    const out = pushToast([live1, mk(1)], live2);
    expect(out.map((t) => t.key)).toEqual(["live:kitchen", 1]);
    expect(out[0].text).toBe("turn on");
  });

  it("finalized toast removes the same-device live toast", () => {
    const live = toastFromRun(liveRun);
    const done = toastFromRun(finalized);
    const out = pushToast([live, mk(1)], done);
    expect(out.map((t) => t.key)).toEqual([7, 1]);
  });

  it("finalized toast keeps live toasts of other devices", () => {
    const otherLive = toastFromRun({ ...liveRun, device: "office" });
    const done = toastFromRun(finalized);
    const out = pushToast([otherLive], done);
    expect(out.map((t) => t.key)).toEqual([7, "live:office"]);
  });

  it("finalized toast without a device replaces the same deviceless live toast", () => {
    const live = toastFromRun({ ...liveRun, device: undefined });
    const done = toastFromRun({ ...finalized, device: undefined });
    const out = pushToast([live], done);
    expect(out.map((t) => t.key)).toEqual([7]);
  });

  it("caps the stack at 4, dropping the oldest", () => {
    const out = pushToast([mk(4), mk(3), mk(2), mk(1)], mk(5));
    expect(out.map((t) => t.key)).toEqual([5, 4, 3, 2]);
  });

  it("respects a custom cap", () => {
    const out = pushToast([mk(2), mk(1)], mk(3), 2);
    expect(out.map((t) => t.key)).toEqual([3, 2]);
  });
});
