// Unit tests for the waterfall segment builder (src/components/primitives.jsx:
// `segsFor` + `total`). These are pure functions; no DOM is needed.
import { describe, it, expect } from "vitest";
import { segsFor, total, fillerMarkerPct } from "../components/primitives.jsx";

describe("total", () => {
  it("returns 0 for an empty timings object", () => {
    expect(total({})).toBe(0);
  });

  it("sums all per-stage timings", () => {
    expect(total({ vad: 1, stt: 2, llm: 3, tts: 4 })).toBe(10);
  });
});

describe("segsFor", () => {
  it("renders a single 'no speech' segment for an empty result, using t_total", () => {
    const segs = segsFor({ result: "empty", t_total: 2500, t: {} });
    expect(segs).toHaveLength(1);
    expect(segs[0].label).toBe("no speech · 2.50s");
    expect(segs[0].pct).toBe(100);
  });

  it("for an empty result without t_total, falls back to the stage sum", () => {
    const segs = segsFor({ result: "empty", t_total: null, t: { vad: 1000, stt: 500 } });
    expect(segs[0].label).toBe("no speech · 1.50s");
  });

  it("emits one segment per non-zero stage in STAGE_ORDER", () => {
    const segs = segsFor({ result: "ok", t: { vad: 50, stt: 50, llm: 0, tts: 0 } });
    // llm/tts are 0 -> dropped; vad then stt order preserved.
    expect(segs).toHaveLength(2);
    expect(segs.map((s) => s.label)).toEqual(["vad 50", "stt 50"]);
  });

  it("labels a segment below the 15% threshold with the bare number only", () => {
    // vad = 10/100 = 10% (<15) -> "10"; stt = 90/100 = 90% -> "stt 90".
    const segs = segsFor({ result: "ok", t: { vad: 10, stt: 90, llm: 0, tts: 0 } });
    expect(segs[0].label).toBe("10");
    expect(segs[1].label).toBe("stt 90");
  });

  it("appends a hatched 'fail' segment for an error result", () => {
    const segs = segsFor({ result: "error", t: { vad: 100, stt: 0, llm: 0, tts: 0 } });
    const last = segs[segs.length - 1];
    expect(last.label).toBe("fail");
    expect(last.pct).toBe(24);
  });

  it("does NOT append a fail segment for a non-error result", () => {
    const segs = segsFor({ result: "ok", t: { vad: 100, stt: 0, llm: 0, tts: 0 } });
    expect(segs.some((s) => s.label === "fail")).toBe(false);
  });

  it("appends a 'rejected' segment for a wake-word-rejected result", () => {
    const segs = segsFor({ result: "rejected", t: { vad: 100, wakeword: 20, stt: 0, llm: 0, tts: 0 } });
    const last = segs[segs.length - 1];
    expect(last.label).toBe("rejected");
    expect(last.pct).toBe(24);
    // The wakeword stage timing still renders its own segment before the marker.
    expect(segs.some((s) => s.label === "20" || s.label === "wakeword 20")).toBe(true);
  });
});

describe("fillerMarkerPct", () => {
  it("returns null when t_filler is null", () => {
    expect(fillerMarkerPct({ t: { vad: 1000, stt: 500, llm: 8000, tts: 500 }, t_filler: null, filler_text: "щас гляну" })).toBe(null);
  });

  it("returns null when filler_text is empty", () => {
    expect(fillerMarkerPct({ t: { vad: 1000, stt: 500, llm: 8000, tts: 500 }, t_filler: 1500, filler_text: "" })).toBe(null);
  });

  it("returns null when total(t) is 0", () => {
    expect(fillerMarkerPct({ t: { vad: 0, stt: 0, llm: 0, tts: 0 }, t_filler: 1500, filler_text: "щас гляну" })).toBe(null);
  });

  it("computes the marker offset as (vad + wakeword + t_filler) / total * 100", () => {
    // No wakeword key here, so wakeword=0: (1000 + 0 + 1500) / 10000 * 100 = 25
    expect(fillerMarkerPct({ t: { vad: 1000, stt: 500, llm: 8000, tts: 500 }, t_filler: 1500, filler_text: "щас гляну" })).toBe(25);
  });

  it("includes the wakeword segment in the marker offset", () => {
    // (1000 + 200 + 1500) / 11000 * 100 = 24.5454...
    expect(fillerMarkerPct({ t: { vad: 1000, wakeword: 200, stt: 500, llm: 8500, tts: 800 }, t_filler: 1500, filler_text: "щас гляну" })).toBeCloseTo(24.5454, 3);
  });

  it("clamps to 100 when the computed value exceeds 100", () => {
    // (1000 + 20000) / 10000 * 100 = 210 -> clamped to 100
    expect(fillerMarkerPct({ t: { vad: 1000, stt: 500, llm: 8000, tts: 500 }, t_filler: 20000, filler_text: "щас гляну" })).toBe(100);
  });
});
