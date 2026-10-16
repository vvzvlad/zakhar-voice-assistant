// @vitest-environment jsdom
// Integration tests (jsdom) for the chain parts of the Request Log: the Chain
// column, the chain filter forwarded to /api/runs, and the run detail listing
// the chain and TTS attempts in the order they ran, with why a handler declined.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import Log from "../pages/log.jsx";
import { getRuns, getRun } from "../api.js";

vi.mock("../api.js", () => ({
  getRuns: vi.fn(),
  getRun: vi.fn(),
  openRunsStream: vi.fn(() => () => {}),
  runAudioUrl: vi.fn(() => ""),
  downloadRunAudio: vi.fn(),
  runTtsAudioUrl: vi.fn(() => ""),
  downloadRunTtsAudio: vi.fn(),
}));

const ROW = { id: 7, ts: 0, device: "kitchen", chain: "smart", result: "ok", stt_text: "какая погода", llm_text: "Солнечно" };

beforeEach(() => {
  vi.clearAllMocks();
  getRuns.mockResolvedValue({ runs: [ROW, { ...ROW, id: 8, chain: null }], total: 2 });
});
afterEach(cleanup);

describe("Request log — chains", () => {
  it("shows a Chain column and forwards the chain filter to /api/runs", async () => {
    render(<Log />);
    await screen.findByText("Chain");
    expect(screen.getAllByTestId("run-chain").map((td) => td.textContent)).toEqual(["smart", "—"]);

    fireEvent.change(screen.getByPlaceholderText("Chain…"), { target: { value: " fast " } });
    await waitFor(() => expect(getRuns).toHaveBeenLastCalledWith({ limit: 100, offset: 0, chain: "fast" }));
  });

  it("lists the run's attempts in order with text, outcome, error and timings", async () => {
    getRun.mockResolvedValue({
      ...ROW,
      attempts: [
        { kind: "chain", chain: "fast", stt: "vosk", handler: "simple-nlu", text: "какая погода", outcome: "declined", error: null, detail: "no alias phrase matched", t_stt: 310, t_handler: 12 },
        { kind: "chain", chain: "smart", stt: "groq", handler: "openrouter", text: "какая погода", outcome: "answered", error: null, detail: null, t_stt: 800, t_handler: 4100 },
        { kind: "tts", provider: "yandex", outcome: "error", error: "timeout", t: 6000 },
        { kind: "tts", provider: "piper", outcome: "spoke", error: null, t: 900 },
      ],
    });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    await screen.findByText("Attempts");

    const items = screen.getAllByTestId("attempt");
    expect(items).toHaveLength(4);
    expect(items[0].textContent).toContain("fast · vosk → simple-nlu");
    expect(items[0].textContent).toContain("STT 0.31s · handler 0.01s");
    expect(items[0].textContent).toContain("declined");
    expect(items[0].textContent).toContain("какая погода");
    // Why the handler declined shows under its attempt; an answer has no detail.
    expect(screen.getAllByTestId("attempt-detail").map((d) => d.textContent)).toEqual(["no alias phrase matched"]);
    expect(items[1].textContent).toContain("answered");
    expect(items[2].textContent).toContain("TTS · yandex");
    expect(items[2].textContent).toContain("timeout");
    expect(items[2].textContent).toContain("6.00s");
    expect(items[3].textContent).toContain("spoke");
  });
});
