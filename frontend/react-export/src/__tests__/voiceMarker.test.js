// Unit tests for the JS mirror of the preferred-voice marker parser
// (src/voiceMarker.js). Must stay consistent with tests/test_voice_marker.py.
import { describe, it, expect } from "vitest";
import { parseVoiceMarker, voiceLabel } from "../voiceMarker.js";

describe("parseVoiceMarker", () => {
  it("parses a single field", () => {
    expect(parseVoiceMarker("hi <<<<<VOICE provider=yandex voice=zahar>>>>> bye"))
      .toEqual({ provider: "yandex", fields: { voice: "zahar" } });
  });

  it("parses multiple fields", () => {
    expect(parseVoiceMarker("<<<<<VOICE provider=fishaudio model=s2-pro reference_id=3b5c9f>>>>>"))
      .toEqual({ provider: "fishaudio", fields: { model: "s2-pro", reference_id: "3b5c9f" } });
  });

  it("parses a provider-only marker", () => {
    expect(parseVoiceMarker("<<<<<VOICE provider=teratts>>>>>"))
      .toEqual({ provider: "teratts", fields: {} });
  });

  it("returns null when the provider is missing", () => {
    expect(parseVoiceMarker("<<<<<VOICE voice=zahar>>>>>")).toBeNull();
  });

  it("returns null when there is no marker / empty / null", () => {
    expect(parseVoiceMarker("a normal prompt")).toBeNull();
    expect(parseVoiceMarker("")).toBeNull();
    expect(parseVoiceMarker(null)).toBeNull();
    expect(parseVoiceMarker(undefined)).toBeNull();
  });

  it("ignores tokens without '='", () => {
    expect(parseVoiceMarker("<<<<<VOICE provider=yandex garbage voice=zahar>>>>>"))
      .toEqual({ provider: "yandex", fields: { voice: "zahar" } });
  });

  it("does not cross a newline", () => {
    expect(parseVoiceMarker("<<<<<VOICE provider=yandex\nfoo >>>>>")).toBeNull();
  });
});

describe("voiceLabel", () => {
  it("joins provider and field values with '/'", () => {
    expect(voiceLabel({ provider: "yandex", fields: { voice: "zahar" } })).toBe("yandex/zahar");
    expect(voiceLabel({ provider: "fishaudio", fields: { model: "s2-pro", reference_id: "3b5c9f" } }))
      .toBe("fishaudio/s2-pro/3b5c9f");
    expect(voiceLabel({ provider: "teratts", fields: {} })).toBe("teratts");
  });

  it("returns an empty string for null", () => {
    expect(voiceLabel(null)).toBe("");
  });
});
