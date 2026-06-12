// @vitest-environment jsdom
// Unit tests for providerOf (dashboard service map). Pins:
//   - vad maps the selected id to the human provider label,
//   - vad falls back to the raw id when the providers list has no match,
//   - a catalog without the vad category (old backend) yields "—" without throwing,
//   - stt/llm/tts intentionally keep the raw id even when a label exists.
import { describe, it, expect } from "vitest";
import { providerOf } from "../pages/dashboard.jsx";

// Stage objects shaped like the real STAGES entries (src/stageMeta.js).
const VAD_STAGE = { key: "vad", name: "VAD", role: "Voice capture", cat: "vad" };
const STT_STAGE = { key: "stt", name: "STT", role: "Speech → text", cat: "stt" };

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
        { id: "stt", selected: "whisper", providers: [{ id: "whisper", label: "Whisper" }] },
      ],
    };
    expect(() => providerOf(VAD_STAGE, catalog)).not.toThrow();
    expect(providerOf(VAD_STAGE, catalog)).toBe("—");
  });

  it("returns the raw id (NOT the label) for a non-vad stage even when a label exists", () => {
    const catalog = {
      categories: [
        {
          id: "stt",
          selected: "whisper",
          providers: [{ id: "whisper", label: "OpenAI Whisper" }],
        },
      ],
    };
    // Asymmetry by design: only vad shows the human label.
    expect(providerOf(STT_STAGE, catalog)).toBe("whisper");
  });
});
