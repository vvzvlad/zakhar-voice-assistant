// Unit tests for the backend-row -> UI-run adapter (src/runsModel.js).
import { describe, it, expect } from "vitest";
import { mapRun, totalMs, fmtSec, statusMeta, applyStreamedRun, pageWindow } from "../runsModel.js";

describe("mapRun", () => {
  it("returns null for a null row (pass-through)", () => {
    expect(mapRun(null)).toBeNull();
  });

  it("coerces every t_* timing to 0 when missing/null", () => {
    const out = mapRun({ id: 1, ts: 0, t_vad: 10, t_stt: null, t_llm: 20 });
    // t_stt is null -> 0; t_tts/t_stress absent -> 0; provided ones kept.
    expect(out.t).toEqual({ vad: 10, stt: 0, llm: 20, stress: 0, tts: 0 });
  });

  it("maps stt_text/llm_text into stt/llm", () => {
    const out = mapRun({ id: 1, ts: 0, stt_text: "hi", llm_text: "hello" });
    expect(out.stt).toBe("hi");
    expect(out.llm).toBe("hello");
  });

  it("maps stress_text into stress (accent-stage output)", () => {
    const out = mapRun({ id: 1, ts: 0, stress_text: "прив+ет", t_stress: 5 });
    expect(out.stress).toBe("прив+ет");
    // The accent timing stays under r.t.stress.
    expect(out.t.stress).toBe(5);
  });

  it("leaves audio null unless audio_bytes is present", () => {
    expect(mapRun({ id: 1, ts: 0 }).audio).toBeNull();
    const withAudio = mapRun({ id: 1, ts: 0, audio_bytes: 1000, audio_ms: 500, audio_fmt: "wav" });
    expect(withAudio.audio).toEqual({ ms: 500, bytes: 1000, fmt: "wav" });
  });

  it("leaves error null unless error_stage is present", () => {
    expect(mapRun({ id: 1, ts: 0 }).error).toBeNull();
    const withErr = mapRun({ id: 1, ts: 0, error_stage: "stt", error_text: "boom" });
    expect(withErr.error).toEqual({ stage: "stt", text: "boom" });
  });

  it("keys a finalized row by id and marks it not-live", () => {
    const out = mapRun({ id: 42, ts: 0 });
    expect(out.key).toBe(42);
    expect(out.live).toBe(false);
  });

  it("keys a live row by device and marks it live (no id)", () => {
    const out = mapRun({ live: 1, device: "kitchen", ts: 0 });
    expect(out.key).toBe("live:kitchen");
    expect(out.live).toBe(true);
    expect(out.id == null).toBe(true);
  });
});

describe("statusMeta", () => {
  it("returns a Running pill for a live row", () => {
    expect(statusMeta({ live: true })).toEqual({ label: "Running", tone: "muted" });
  });

  it("maps a finalized result to its RESULT_META entry", () => {
    expect(statusMeta({ result: "ok" })).toEqual({ label: "OK", tone: "good" });
  });
});

describe("applyStreamedRun", () => {
  it("prepends a matching live partial, replacing a prior live row for the same device", () => {
    const prev = [{ key: "live:kitchen", device: "kitchen", stt: "old" }];
    const next = { key: "live:kitchen", live: true, device: "kitchen", stt: "new" };
    const out = applyStreamedRun(prev, next, true, 100);
    expect(out).toEqual([next]);
  });

  it("finalized (match) drops the superseded live row and dedups same id, then prepends", () => {
    const prev = [
      { key: "live:kitchen", device: "kitchen" },
      { key: 7, id: 7, stt: "stale" },
      { key: 3, id: 3 },
    ];
    const fin = { key: 7, id: 7, live: false, device: "kitchen", stt: "fresh" };
    const out = applyStreamedRun(prev, fin, true, 100);
    expect(out).toEqual([fin, { key: 3, id: 3 }]);
  });

  it("finalized with match=false still removes the superseded live row but does not insert itself", () => {
    const prev = [
      { key: "live:kitchen", device: "kitchen" },
      { key: 3, id: 3 },
    ];
    const fin = { key: 9, id: 9, live: false, device: "kitchen" };
    const out = applyStreamedRun(prev, fin, false, 100);
    expect(out).toEqual([{ key: 3, id: 3 }]);
  });

  it("returns the previous list unchanged for a non-matching live partial", () => {
    const prev = [{ key: 3, id: 3 }];
    const live = { key: "live:bath", live: true, device: "bath" };
    const out = applyStreamedRun(prev, live, false, 100);
    expect(out).toBe(prev);
  });

  it("respects the cap argument", () => {
    const prev = [{ key: 1, id: 1 }, { key: 2, id: 2 }, { key: 3, id: 3 }];
    const fin = { key: 4, id: 4, live: false, device: "x" };
    const out = applyStreamedRun(prev, fin, true, 2);
    expect(out).toEqual([fin, { key: 1, id: 1 }]);
  });
});

describe("totalMs", () => {
  it("prefers the backend t_total over the per-stage sum", () => {
    expect(totalMs({ t_total: 999, t: { vad: 1, stt: 2 } })).toBe(999);
  });

  it("keeps t_total:0 (does NOT fall back to the stage sum)", () => {
    // 0 is a valid authoritative total; only null/undefined trigger the fallback.
    expect(totalMs({ t_total: 0, t: { vad: 1, stt: 2 } })).toBe(0);
  });

  it("falls back to summing r.t when t_total is absent", () => {
    expect(totalMs({ t: { vad: 1, stt: 2, llm: 3 } })).toBe(6);
  });
});

describe("pageWindow", () => {
  it("returns the full 1..n list for small totals (<= 7), no ellipsis", () => {
    expect(pageWindow(1, 1)).toEqual([1]);
    expect(pageWindow(3, 5)).toEqual([1, 2, 3, 4, 5]);
    expect(pageWindow(4, 7)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("collapses gaps to a single '…' for large totals", () => {
    // Current 7 of 13: first, a gap, the 6/7/8 window, a gap, then last.
    expect(pageWindow(7, 13)).toEqual([1, "…", 6, 7, 8, "…", 13]);
  });

  it("renders the first/last edges without a stray ellipsis next to consecutive pages", () => {
    // At page 1 the leading 1,2 are consecutive -> no leading "…".
    expect(pageWindow(1, 13)).toEqual([1, 2, "…", 13]);
    // At the last page the trailing 12,13 are consecutive -> no trailing "…".
    expect(pageWindow(13, 13)).toEqual([1, "…", 12, 13]);
  });
});

describe("fmtSec", () => {
  it("renders an em dash for null / undefined / NaN", () => {
    expect(fmtSec(null)).toBe("—");
    expect(fmtSec(undefined)).toBe("—");
    expect(fmtSec(NaN)).toBe("—");
  });

  it("formats ms as seconds with two decimals (1234 -> '1.23s')", () => {
    expect(fmtSec(1234)).toBe("1.23s");
  });

  it("formats 0 as '0.00s' (not an em dash)", () => {
    expect(fmtSec(0)).toBe("0.00s");
  });
});
