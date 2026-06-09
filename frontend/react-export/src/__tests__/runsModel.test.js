// Unit tests for the backend-row -> UI-run adapter (src/runsModel.js).
import { describe, it, expect } from "vitest";
import { mapRun, totalMs, fmtSec } from "../runsModel.js";

describe("mapRun", () => {
  it("returns null for a null row (pass-through)", () => {
    expect(mapRun(null)).toBeNull();
  });

  it("coerces every t_* timing to 0 when missing/null", () => {
    const out = mapRun({ id: 1, ts: 0, t_vad: 10, t_stt: null, t_llm: 20 });
    // t_stt is null -> 0; t_tts/t_ruaccent absent -> 0; provided ones kept.
    expect(out.t).toEqual({ vad: 10, stt: 0, llm: 20, ruaccent: 0, tts: 0 });
  });

  it("maps stt_text/llm_text into stt/llm", () => {
    const out = mapRun({ id: 1, ts: 0, stt_text: "hi", llm_text: "hello" });
    expect(out.stt).toBe("hi");
    expect(out.llm).toBe("hello");
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
