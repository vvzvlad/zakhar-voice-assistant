// Unit tests for matchPreset (src/vadPresets.js), extracted from pages/stages.jsx
// (R-FE-1). Returns the preset name whose silence_ms+min_speech_ms match, else null.
import { describe, it, expect } from "vitest";
import { matchPreset, VAD_PRESETS } from "../vadPresets.js";

describe("matchPreset", () => {
  it("returns the preset name on an exact silence_ms + min_speech_ms match", () => {
    expect(matchPreset({ silence_ms: 500, min_speech_ms: 150 })).toBe("Fast");
    expect(matchPreset({ silence_ms: 800, min_speech_ms: 200 })).toBe("Balanced");
    expect(matchPreset({ silence_ms: 1200, min_speech_ms: 300 })).toBe("Patient");
  });

  it("ignores extra fields on the draft (only the two macro fields matter)", () => {
    expect(matchPreset({ silence_ms: 500, min_speech_ms: 150, aggressiveness: 2 })).toBe("Fast");
  });

  it("returns null when one of the two fields mismatches", () => {
    expect(matchPreset({ silence_ms: 500, min_speech_ms: 999 })).toBeNull();
    expect(matchPreset({ silence_ms: 999, min_speech_ms: 150 })).toBeNull();
  });

  it("returns null when only unrelated extra fields are present (no macro match)", () => {
    expect(matchPreset({ aggressiveness: 2 })).toBeNull();
    expect(matchPreset({})).toBeNull();
  });

  it("stays consistent with the exported VAD_PRESETS table", () => {
    // Guards against the table and matcher drifting apart.
    for (const [name, p] of Object.entries(VAD_PRESETS)) {
      expect(matchPreset(p)).toBe(name);
    }
  });
});
