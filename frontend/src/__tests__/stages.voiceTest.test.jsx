// @vitest-environment jsdom
// Integration tests (jsdom) for the universal TTS voice-test card: rendered ONLY
// on the TTS stage page, it synthesizes the typed phrase with the CURRENT
// (possibly unsaved) provider draft via the /api/tts/test endpoint and plays the
// result through the streaming player. The phrase persists in localStorage across
// reloads, prefilled with a default phrase.
//
// The card is wired to two seams, both mocked here:
//   - api.streamTtsVoice(provider, settings, text, signal) -> a raw Response. The
//     fake Response is shape-only (the component never reads its body — that is
//     playAudioResponse's job, which is also mocked).
//   - streamAudio.playAudioResponse(resp, { onPlaying, onEnded, onError }) ->
//     { stop }. We capture the callbacks so each test can drive the terminal
//     events (onPlaying / onEnded / onError) and assert the resulting UI phase.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, act } from "@testing-library/react";
import { TTS, STT, LLM } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";
import * as api from "../api.js";
import { playAudioResponse } from "../streamAudio.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getOptions: vi.fn(async () => ({ options: [] })),
  getChimes: vi.fn(async () => ({ options: [] })),
  playChime: vi.fn(async () => ({ played: ["speaker"], offline: [] })),
  // Minimal fake Response: ok + a Content-Type header. The component hands this
  // straight to playAudioResponse (mocked), so no body/blob plumbing is needed.
  streamTtsVoice: vi.fn(async () => ({
    ok: true,
    headers: { get: () => "audio/mpeg" },
    body: {},
  })),
}));
vi.mock("../streamAudio.js", () => ({ playAudioResponse: vi.fn() }));

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

// Fake playAudioResponse controllers: each call records its captured callbacks and
// returns a { stop } spy, so tests can fire onPlaying/onEnded/onError and assert
// stop() was invoked on cleanup. controllers[i] mirrors the i-th playback.
let controllers;
function installPlayer() {
  controllers = [];
  playAudioResponse.mockImplementation((resp, cbs) => {
    const ctl = { stop: vi.fn(), cbs };
    controllers.push(ctl);
    return ctl;
  });
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
  installPlayer();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// Click "Test voice", wait for the request to resolve and a NEW playback to be
// wired, then fire onPlaying so the button reaches the "Stop" phase. Returns the
// freshly-created controller (works across repeated runs, since `controllers`
// accumulates one entry per playback).
async function startAndPlay() {
  const before = controllers.length;
  fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
  await waitFor(() => expect(controllers.length).toBe(before + 1));
  const ctl = controllers[before];
  act(() => ctl.cbs.onPlaying());
  await screen.findByRole("button", { name: "Stop" });
  return ctl;
}

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
  it("calls streamTtsVoice with provider, the CURRENT (unsaved) draft, text and an AbortSignal, then starts playback", async () => {
    render(<TTS />);
    // Edit the provider form WITHOUT saving: the test must use the live draft.
    fireEvent.change(screen.getByDisplayValue("http://initial"), {
      target: { value: "http://unsaved-draft" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));

    await waitFor(() => expect(api.streamTtsVoice).toHaveBeenCalledTimes(1));
    const [provider, draft, text, signal] = api.streamTtsVoice.mock.calls[0];
    expect(provider).toBe("teratts");
    expect(draft).toEqual({ base_url: "http://unsaved-draft" });
    expect(text).toBe(DEFAULT_PHRASE);
    expect(signal).toBeInstanceOf(AbortSignal);

    // The resolved Response is handed to the streaming player.
    await waitFor(() => expect(playAudioResponse).toHaveBeenCalledTimes(1));
    expect(playAudioResponse.mock.calls[0][0]).toMatchObject({ ok: true });
  });

  it("shows the API error message (red line) on failure and starts no playback", async () => {
    api.streamTtsVoice.mockRejectedValueOnce(new Error("text too long (max 500 chars)"));
    render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    await waitFor(() =>
      expect(screen.getByText("text too long (max 500 chars)")).toBeInTheDocument());
    expect(playAudioResponse).not.toHaveBeenCalled();
    // Back to the idle button.
    expect(screen.getByRole("button", { name: "Test voice" })).toBeEnabled();
  });

  it("an AbortError rejection (Stop/unmount race) shows NO message", async () => {
    const abort = new Error("aborted");
    abort.name = "AbortError";
    api.streamTtsVoice.mockRejectedValueOnce(abort);
    render(<TTS />);
    fireEvent.click(screen.getByRole("button", { name: "Test voice" }));
    // Let the rejected promise settle, then assert nothing went red.
    await waitFor(() => expect(api.streamTtsVoice).toHaveBeenCalled());
    await Promise.resolve();
    expect(screen.queryByText("aborted")).toBeNull();
    expect(playAudioResponse).not.toHaveBeenCalled();
  });

  it("disables the button with 'Synthesizing…' while the request is in flight", async () => {
    let resolveCall;
    api.streamTtsVoice.mockReturnValueOnce(new Promise((res) => { resolveCall = res; }));
    render(<TTS />);
    const button = screen.getByRole("button", { name: "Test voice" });
    fireEvent.click(button);
    // Disabled until the synthesized clip STARTS playing — repeat clicks impossible.
    await waitFor(() => expect(button).toBeDisabled());
    expect(button).toHaveTextContent("Synthesizing…");
    // Resolve the request, then the player reports playback started: the SAME
    // button becomes an enabled Stop button.
    resolveCall({ ok: true, headers: { get: () => "audio/mpeg" }, body: {} });
    await waitFor(() => expect(controllers).toHaveLength(1));
    act(() => controllers[0].cbs.onPlaying());
    await waitFor(() => expect(button).toHaveTextContent("Stop"));
    expect(button).toBeEnabled();
  });

  it("onPlaying turns the button into a danger-styled Stop button", async () => {
    render(<TTS />);
    const ctl = await startAndPlay();
    const stopBtn = screen.getByRole("button", { name: "Stop" });
    expect(stopBtn).toBeEnabled();
    expect(stopBtn).toHaveClass("z-btn", "d");
    expect(ctl.stop).not.toHaveBeenCalled();
  });

  it("Stop calls the controller's stop(), aborts the request and returns to idle", async () => {
    render(<TTS />);
    const ctl = await startAndPlay();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));

    expect(ctl.stop).toHaveBeenCalled();
    const idle = screen.getByRole("button", { name: "Test voice" });
    expect(idle).toBeEnabled();
    expect(idle).toHaveClass("z-btn", "p");
  });

  it("onEnded reverts to idle; a late onEnded after Stop is harmless", async () => {
    render(<TTS />);
    const ctl = await startAndPlay();

    // Natural end of playback: back to idle.
    act(() => ctl.cbs.onEnded());
    expect(screen.getByRole("button", { name: "Test voice" })).toBeEnabled();

    // Second run: manual Stop, then the player still delivers onEnded afterwards —
    // cleanup is idempotent, so no crash and the card stays idle.
    const ctl2 = await startAndPlay();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(ctl2.stop).toHaveBeenCalled();
    expect(() => act(() => ctl2.cbs.onEnded())).not.toThrow();
    expect(screen.getByRole("button", { name: "Test voice" })).toBeEnabled();
  });

  it("a late onEnded/onError from a superseded run does not touch the next playback", async () => {
    render(<TTS />);

    // Run #1 → Stop: controller[0] torn down, refs cleared.
    const ctl0 = await startAndPlay();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(ctl0.stop).toHaveBeenCalled();

    // Run #2: controller[1] is now active and owns the refs.
    const ctl1 = await startAndPlay();
    expect(controllers).toHaveLength(2);

    // The superseded run #1 delivers its queued onEnded (and even onError) AFTER
    // run #2 started playing: both must be no-ops for the shared state.
    act(() => ctl0.cbs.onEnded());
    act(() => ctl0.cbs.onError("Audio playback failed"));

    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    expect(ctl1.stop).not.toHaveBeenCalled();
    expect(screen.queryByText("Audio playback failed")).toBeNull();
  });

  it("onError shows a message and reverts to idle", async () => {
    render(<TTS />);
    const ctl = await startAndPlay();

    act(() => ctl.cbs.onError("Audio playback failed"));
    expect(screen.getByText("Audio playback failed")).toBeInTheDocument();
    expect(ctl.stop).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Test voice" })).toBeEnabled();
  });

  it("unmount during playback calls the controller's stop()", async () => {
    const { unmount } = render(<TTS />);
    const ctl = await startAndPlay();

    unmount();
    expect(ctl.stop).toHaveBeenCalled();
  });

  it("disables the button when the text is blank", () => {
    render(<TTS />);
    fireEvent.change(screen.getByPlaceholderText(PHRASE_PLACEHOLDER), { target: { value: "   " } });
    expect(screen.getByRole("button", { name: "Test voice" })).toBeDisabled();
  });
});
