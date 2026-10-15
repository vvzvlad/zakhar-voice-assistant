// @vitest-environment jsdom
// Unit tests for providerOf (dashboard service map). Pins:
//   - vad maps the selected id to the human provider label,
//   - vad falls back to the raw id when the providers list has no match,
//   - a catalog without the vad category (old backend) yields "—" without throwing,
//   - other stages intentionally keep the raw id even when a label exists,
//   - tts shows its first ENABLED variant (no selected provider).
import { describe, it, expect } from "vitest";
import { providerOf } from "../pages/dashboard.jsx";

// Stage objects shaped like the real STAGES entries (src/stageMeta.js).
const VAD_STAGE = { key: "vad", name: "VAD", role: "Voice capture", cat: "vad" };
const WAKEWORD_STAGE = { key: "wakeword", name: "Wakeword", role: "Wake-word verify", cat: "wakeword" };
const TTS_STAGE = { key: "tts", name: "TTS", role: "Text → speech", cat: "tts" };

describe("providerOf", () => {
  it("returns the human label for the vad stage when the provider entry has one", () => {
    const catalog = {
      categories: [
        {
          id: "vad",
          selected: "webrtc",
          providers: [{ id: "webrtc", label: "WebRTC VAD" }],
        },
      ],
    };
    expect(providerOf(VAD_STAGE, catalog)).toBe("WebRTC VAD");
  });

  it("falls back to the raw selected id for vad when no providers entry matches", () => {
    const catalog = {
      categories: [
        {
          id: "vad",
          selected: "mystery",
          providers: [{ id: "webrtc", label: "WebRTC VAD" }],
        },
      ],
    };
    expect(providerOf(VAD_STAGE, catalog)).toBe("mystery");
  });

  it('returns "—" without throwing when the catalog has no vad category (old backend)', () => {
    const catalog = {
      categories: [
        { id: "wakeword", selected: "vosk", providers: [{ id: "vosk", label: "Vosk" }] },
      ],
    };
    expect(() => providerOf(VAD_STAGE, catalog)).not.toThrow();
    expect(providerOf(VAD_STAGE, catalog)).toBe("—");
  });

  it("returns the raw id (NOT the label) for a non-vad stage even when a label exists", () => {
    const catalog = {
      categories: [
        {
          id: "wakeword",
          selected: "vosk",
          providers: [{ id: "vosk", label: "Vosk verifier" }],
        },
      ],
    };
    // Asymmetry by design: only vad shows the human label.
    expect(providerOf(WAKEWORD_STAGE, catalog)).toBe("vosk");
  });

  it('returns the first enabled tts variant, or "—" when none is enabled', () => {
    const tts = (variants) => ({ categories: [{ id: "tts", variants, providers: [] }] });
    expect(providerOf(TTS_STAGE, tts([
      { provider: "fish", enabled: false }, { provider: "piper", enabled: true },
    ]))).toBe("piper");
    expect(providerOf(TTS_STAGE, tts([{ provider: "fish", enabled: false }]))).toBe("—");
  });
});
