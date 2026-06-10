// Unit tests for the run-toast pure helpers (src/toastModel.js).
import { describe, it, expect } from "vitest";
import { shouldNotify, toastFromRun, pushToast, QUIET_PAGES } from "../toastModel.js";

const finalized = { id: 7, device: "kitchen", result: "ok", stt_text: "turn on the light", t_total: 1234 };

describe("shouldNotify", () => {
  it("is false for live in-progress snapshots", () => {
    expect(shouldNotify({ ...finalized, id: null, live: true }, "stt")).toBe(false);
    expect(shouldNotify({ ...finalized, live: 1 }, "stt")).toBe(false);
  });

  it("is false for rows without an id (not finalized yet)", () => {
    expect(shouldNotify({ ...finalized, id: undefined }, "stt")).toBe(false);
    expect(shouldNotify({ ...finalized, id: null }, "stt")).toBe(false);
  });

  it("is false on the pages that already show live runs", () => {
    expect(shouldNotify(finalized, "dashboard")).toBe(false);
    expect(shouldNotify(finalized, "log")).toBe(false);
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
  it("maps a known result via RESULT_META (ok -> OK/good)", () => {
    const t = toastFromRun(finalized);
    expect(t).toEqual({
      id: 7, device: "kitchen", label: "OK", tone: "good",
      text: "turn on the light", totalMs: 1234,
    });
  });

  it("falls back to a muted tone with the raw label for unknown results", () => {
    const t = toastFromRun({ ...finalized, result: "weird" });
    expect(t.label).toBe("weird");
    expect(t.tone).toBe("muted");
  });

  it("uses an em dash when the result is missing", () => {
    expect(toastFromRun({ ...finalized, result: undefined }).label).toBe("—");
  });

  it("substitutes '(silence)' when stt_text is missing", () => {
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
  const mk = (id) => ({ id, device: "d" + id });

  it("prepends the new toast", () => {
    const out = pushToast([mk(1)], mk(2));
    expect(out.map((t) => t.id)).toEqual([2, 1]);
  });

  it("dedupes by id, replacing the old entry with the new one", () => {
    const out = pushToast([mk(1), mk(2)], { id: 2, device: "fresh" });
    expect(out.map((t) => t.id)).toEqual([2, 1]);
    expect(out[0].device).toBe("fresh");
  });

  it("caps the stack at 4, dropping the oldest", () => {
    const out = pushToast([mk(4), mk(3), mk(2), mk(1)], mk(5));
    expect(out.map((t) => t.id)).toEqual([5, 4, 3, 2]);
  });

  it("respects a custom cap", () => {
    const out = pushToast([mk(2), mk(1)], mk(3), 2);
    expect(out.map((t) => t.id)).toEqual([3, 2]);
  });
});
