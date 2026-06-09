// Unit tests for errorLines (src/useStageForm.js): turns a pydantic 422 ApiError
// into short display lines. errorLines is already exported from the module.
import { describe, it, expect } from "vitest";
import { errorLines } from "../useStageForm.js";

describe("errorLines", () => {
  it("returns [] for a null error", () => {
    expect(errorLines(null)).toEqual([]);
  });

  it("formats pydantic detail[] as 'loc: msg' with the 'body' prefix stripped", () => {
    const err = {
      detail: [
        { loc: ["body", "silence_ms"], msg: "must be >= 0" },
        { loc: ["body", "vad", "min_speech_ms"], msg: "invalid" },
      ],
    };
    expect(errorLines(err)).toEqual([
      "silence_ms: must be >= 0",
      "vad.min_speech_ms: invalid",
    ]);
  });

  it("uses just the msg when the loc reduces to empty after stripping 'body'", () => {
    const err = { detail: [{ loc: ["body"], msg: "root level error" }] };
    expect(errorLines(err)).toEqual(["root level error"]);
  });

  it("falls back to [err.message] when detail is empty", () => {
    const err = { message: "boom", detail: [] };
    expect(errorLines(err)).toEqual(["boom"]);
  });

  it("falls back to [err.message] when detail is absent", () => {
    const err = { message: "network down" };
    expect(errorLines(err)).toEqual(["network down"]);
  });
});
