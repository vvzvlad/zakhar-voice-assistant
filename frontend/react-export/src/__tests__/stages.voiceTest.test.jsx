// @vitest-environment jsdom
// Integration tests (jsdom) for the universal TTS voice-test card: rendered ONLY
// on the TTS stage page, it synthesizes the typed phrase with the CURRENT
// (possibly unsaved) provider draft via the /api/tts/test endpoint and plays the
// returned blob in the browser. The phrase persists in localStorage across
// reloads, prefilled with a default phrase.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, act } from "@testing-library/react";
import { TTS, STT, LLM } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";
import * as api from "../api.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getOptions: vi.fn(async () => ({ options: [] })),
  getChimes: vi.fn(async () => ({ options: [] })),
  playChime: vi.fn(async () => ({ played: ["speaker"], offline: [] })),
  testTtsVoice: vi.fn(async () => new Blob(["mp3"], { type: "audio/mpeg" })),
}));

const TEXT_KEY = "zakhar.tts.test_text";
const DEFAULT_PHRASE = "Привет! Это проверка голоса: раз, два, три.";
const PHRASE_PLACEHOLDER = "Phrase to synthesize";

// Minimal catalog mirroring /api/catalog: all three provider stages so the
// STT/LLM pages render too (they must NOT show the voice-test card).
function makeCatalog() {
  const schema = { type: "object", properties: { base_url: { type: "string" } } };
  const cat = (id, pid) => ({
    id,
    selected: pid,
    providers: [{ id: pid, label: pid, schema, values: { base_url: "http://initial" } }],
  });
  return {
    categories: [cat("stt", "groq"), cat("llm", "openrouter"), cat("tts", "teratts")],
    core: { schema: {}, values: {} },
  };
}

// Audio stub: records created instances; play() resolves immediately by default;
// pause() is a spy so Stop / unmount behavior is observable; listeners can be
// fired manually to simulate "ended" / "error".
class FakeAudio {
  constructor(url) {
    this.url = url;
    this.listeners = {};
    this.pause = vi.fn();
    FakeAudio.instances.push(this);
  }
  addEventListener(ev, cb) { this.listeners[ev] = cb; }
  play() { return FakeAudio.playImpl(); }
}

// Functional in-memory localStorage: the ambient global in this environment is
// Node's experimental (nonfunctional without --localstorage-file) webstorage,
// which shadows jsdom's — stub a real one so persistence is testable.
function makeLocalStorage() {
  let store = {};
  return {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
    clear: () => { store = {}; },
  };
}

let storage;
beforeEach(() => {
  vi.clearAllMocks();
  storage = makeLocalStorage();
  vi.stubGlobal("localStorage", storage);
  useAppData.mockReturnValue({ catalog: makeCatalog(), config: {}, patch: vi.fn(async () => ({})) });
  FakeAudio.instances = [];
  FakeAudio.playImpl = () => Promise.resolve();
  vi.stubGlobal("Audio", FakeAudio);
  // jsdom has no object-URL implementation — stub both ends.
  URL.createObjectURL = vi.fn(() => "blob:voice-test");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("VoiceTestCard placement", () => {
  it("renders on the TTS stage only — not on STT or LLM", () => {
    render(<TTS />);
    expect(screen.getByRole("button", { name: "Test voice" })).toBeInTheDocument();
    cleanup();

    render(<STT />);
    expect(screen.queryByText("Test voice")).toBeNull();
    cleanup();

    render(<LLM />);
    expect(screen.queryByText("Test voice")).toBeNull();
  });
});

describe("test phrase persistence (localStorage)", () => {
  it("prefills with the default phrase when nothing is stored", () => {
    render(<TTS />);
    expect(screen.getByPlaceholderText(PHRASE_PLACEHOLDER)).toHaveValue(DEFAULT_PHRASE);
  });

  it("initializes from the stored value", () => {
    storage.setItem(TEXT_KEY, "Сохранённая фраза");
    render(<TTS />);
    expect(screen.getByPlaceholderText(PHRASE_PLACEHOLDER)).toHaveValue("Сохранённая фраза");
  });

  it("writes every edit through to localStorage", () => {
    render(<TTS />);
    const input = screen.getByPlaceholderText(PHRASE_PLACEHOLDER);
    fireEvent.change(input, { target: { value: "Новая фраза" } });
    expect(input).toHaveValue("Новая фраза");
    expect(storage.getItem(TEXT_KEY)).toBe("Новая фраза");
  });
});

describe("voice test action", () => {
  it("calls the API with the selected provider, the CURRENT (unsaved) draft and the text, then plays the blob", async () => {
    render(<TTS />);
    // Edit the provider form WITHOUT saving: the test must use the live draft.
    fireEvent.change(screen.getByDisplayValue("http://initial"), {
      target: { value: "http://unsaved-draft" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));

    await waitFor(() => expect(api.testTtsVoice).toHaveBeenCalledWith(
      "teratts", { base_url: "http://unsaved-draft" }, DEFAULT_PHRASE,
    ));
    // The blob is turned into an object URL and played in the browser.
    await waitFor(() => expect(FakeAudio.instances).toHaveLength(1));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(FakeAudio.instances[0].url).toBe("blob:voice-test");
    // The object URL is revoked when playback ends (act: the handler sets state).
    act(() => FakeAudio.instances[0].listeners.ended());
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:voice-test");
  });

  it("shows the API error message (red line) on failure", async () => {
    api.testTtsVoice.mockRejectedValueOnce(new Error("text too long (max 500 chars)"));
    render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    await waitFor(() =>
      expect(screen.getByText("text too long (max 500 chars)")).toBeInTheDocument());
    // No audio was created for a failed request.
    expect(FakeAudio.instances).toHaveLength(0);
  });

  it("disables the button with 'Synthesizing…' while the request is in flight", async () => {
    let resolveCall;
    api.testTtsVoice.mockReturnValueOnce(new Promise((res) => { resolveCall = res; }));
    render(<TTS />);
    const button = screen.getByRole("button", { name: "Test voice" });
    fireEvent.click(button);
    // Disabled until the synthesized clip STARTS playing — repeat clicks impossible.
    await waitFor(() => expect(button).toBeDisabled());
    expect(button).toHaveTextContent("Synthesizing…");
    // Once playback starts the SAME button becomes an enabled Stop button.
    resolveCall(new Blob(["mp3"], { type: "audio/mpeg" }));
    await waitFor(() => expect(button).toHaveTextContent("Stop"));
    expect(button).toBeEnabled();
  });

  it("turns into a danger-styled Stop button while playing", async () => {
    render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    const stopBtn = await screen.findByRole("button", { name: "Stop" });
    expect(stopBtn).toBeEnabled();
    expect(stopBtn).toHaveClass("z-btn", "d");
  });

  it("Stop pauses the audio, revokes the object URL and returns to idle", async () => {
    render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    const stopBtn = await screen.findByRole("button", { name: "Stop" });
    fireEvent.click(stopBtn);

    expect(FakeAudio.instances[0].pause).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:voice-test");
    const idle = screen.getByRole("button", { name: "Test voice" });
    expect(idle).toBeEnabled();
    expect(idle).toHaveClass("z-btn", "p");
  });

  it("'ended' reverts to idle and revokes the URL; a late 'ended' after Stop is harmless", async () => {
    render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    await screen.findByRole("button", { name: "Stop" });

    // Natural end of playback: back to idle, URL freed exactly once.
    act(() => FakeAudio.instances[0].listeners.ended());
    expect(screen.getByRole("button", { name: "Test voice" })).toBeEnabled();
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1);

    // Second run: manual Stop, then the browser still fires "ended" afterwards —
    // cleanup is idempotent, so no crash and no double revoke for this clip.
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2);
    expect(() => act(() => FakeAudio.instances[1].listeners.ended())).not.toThrow();
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2);
  });

  it("a late 'ended'/'error' from a superseded clip does not kill the next playback", async () => {
    // Distinguishable URLs per clip so revocations can be attributed.
    let n = 0;
    URL.createObjectURL.mockImplementation(() => `blob:clip-${n++}`);
    render(<TTS />);

    // Test #1 → Stop: clip0's URL is revoked, refs are cleared.
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:clip-0");

    // Test #2: clip1 (instances[1]) is now playing and owns the refs.
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    await screen.findByRole("button", { name: "Stop" });
    expect(FakeAudio.instances).toHaveLength(2);

    // The browser delivers clip0's queued "ended" (and even "error") AFTER
    // clip1 started: both must be no-ops for the shared refs.
    act(() => FakeAudio.instances[0].listeners.ended());
    act(() => FakeAudio.instances[0].listeners.error());

    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    expect(FakeAudio.instances[1].pause).not.toHaveBeenCalled();
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith("blob:clip-1");
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1); // only clip0, by Stop
    expect(screen.queryByText("Audio playback failed")).toBeNull();
  });

  it("playback 'error' shows a message and reverts to idle", async () => {
    render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    await screen.findByRole("button", { name: "Stop" });

    act(() => FakeAudio.instances[0].listeners.error());
    expect(screen.getByText("Audio playback failed")).toBeInTheDocument();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:voice-test");
    expect(screen.getByRole("button", { name: "Test voice" })).toBeEnabled();
  });

  it("unmount during playback pauses the audio and revokes the URL", async () => {
    const { unmount } = render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    await screen.findByRole("button", { name: "Stop" });

    unmount();
    expect(FakeAudio.instances[0].pause).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:voice-test");
  });

  it("disables the button when the text is blank", () => {
    render(<TTS />);
    fireEvent.change(screen.getByPlaceholderText(PHRASE_PLACEHOLDER), { target: { value: "   " } });
    expect(screen.getByRole("button", { name: "Test voice" })).toBeDisabled();
  });
});
