// Unit tests for the JS mirror of the preferred-voice marker parser
// (src/voiceMarker.js). Must stay consistent with tests/test_voice_marker.py.
import { describe, it, expect } from "vitest";
import { parseVoiceMarker, voiceName, differsFromVoice, buildVoiceMarker } from "../voiceMarker.js";

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
    expect(parseVoiceMarker("<<<<<VOICE provider=piper>>>>>"))
      .toEqual({ provider: "piper", fields: {} });
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

describe("voiceName", () => {
  const fish = { provider: "fishaudio", fields: { reference_id: "9b64e", model: "s2-pro", speed: "0.9" } };

  it("takes the name the TTS form keeps for the same voice id", () => {
    expect(voiceName(fish, { reference_id: "9b64e", reference_id_label: "Anna" }, "Other")).toBe("Anna");
  });

  it("names another voice by the options lookup, else by its id", () => {
    expect(voiceName(fish, { reference_id: "aaaa", reference_id_label: "Anna" }, "Boris")).toBe("Boris");
    expect(voiceName(fish, { reference_id: "aaaa", reference_id_label: "Anna" })).toBe("9b64e");
  });

  it("names a plain voice field by its value, a voice file by its basename, none by the provider", () => {
    expect(voiceName({ provider: "yandex", fields: { voice: "zahar" } })).toBe("zahar");
    expect(voiceName({ provider: "piper", fields: { voice_path: "models/ru_RU-ruslan-medium.onnx" } }))
      .toBe("ru_RU-ruslan-medium.onnx");
    expect(voiceName({ provider: "piper", fields: {} })).toBe("piper");
    expect(voiceName(null)).toBe("");
  });
});

describe("differsFromVoice", () => {
  const marker = { provider: "yandex", fields: { voice: "zahar", speed: "1.2" } };

  it("is false when the current provider holds every marker field", () => {
    expect(differsFromVoice(marker, "yandex", { voice: "zahar", speed: 1.2, role: "neutral" })).toBe(false);
  });

  it("is true for another provider or a differing field", () => {
    expect(differsFromVoice(marker, "piper", { voice: "zahar", speed: 1.2 })).toBe(true);
    expect(differsFromVoice(marker, "yandex", { voice: "alena", speed: 1.2 })).toBe(true);
    expect(differsFromVoice(marker, "yandex", { voice: "zahar" })).toBe(true);
  });

  it("is false without a marker", () => {
    expect(differsFromVoice(null, "yandex", {})).toBe(false);
  });
});

describe("buildVoiceMarker", () => {
  it("builds a marker from provider + fields", () => {
    expect(buildVoiceMarker("yandex", { voice: "zahar", speed: 1.2 }))
      .toBe("<<<<<VOICE provider=yandex voice=zahar speed=1.2>>>>>");
  });

  it("never emits secret fields", () => {
    const m = buildVoiceMarker("yandex", {
      voice: "zahar", api_key: "sk-secret", token: "tok", password: "pw",
    });
    expect(m).toBe("<<<<<VOICE provider=yandex voice=zahar>>>>>");
    expect(m).not.toContain("api_key");
    expect(m).not.toContain("token");
    expect(m).not.toContain("password");
  });

  it("skips empty, whitespace-bearing, and '>'-containing values", () => {
    expect(buildVoiceMarker("fishaudio", { model: "s2-pro", reference_id: "", note: "a b", bad: "x>>>>>y" }))
      .toBe("<<<<<VOICE provider=fishaudio model=s2-pro>>>>>");
  });

  it("keeps falsey-but-valid values (0 / false), not dropped as falsey", () => {
    expect(buildVoiceMarker("yandex", { speed: 0, flag: false }))
      .toBe("<<<<<VOICE provider=yandex speed=0 flag=false>>>>>");
  });

  it("skips '<field>_label' display-companion fields (not voice identity)", () => {
    // A space-less label exercises the endsWith("_label") skip specifically: a
    // label with whitespace would already be dropped by the whitespace guard.
    const m = buildVoiceMarker("fishaudio", { reference_id: "abc123", reference_id_label: "Nice" });
    expect(m).toContain("reference_id=abc123");
    expect(m).not.toContain("reference_id_label");
  });

  it("yields a provider-only marker when nothing emittable remains", () => {
    expect(buildVoiceMarker("piper", { base_url: "" }))
      .toBe("<<<<<VOICE provider=piper>>>>>");
  });

  it("returns an empty string for an empty provider", () => {
    expect(buildVoiceMarker("")).toBe("");
  });

  it("round-trips through parseVoiceMarker", () => {
    expect(parseVoiceMarker(buildVoiceMarker("fishaudio", { model: "s2-pro", reference_id: "abc" })))
      .toEqual({ provider: "fishaudio", fields: { model: "s2-pro", reference_id: "abc" } });
  });
});
