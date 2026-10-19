// @vitest-environment jsdom
// Integration tests (jsdom) for the chain parts of the Request Log: the Chain
// column, the chain filter forwarded to /api/runs, and the run detail listing
// the chain and TTS attempts in the order they ran, with why a handler declined.
// Also the device column, which shows a device's label and filters by it, the
// request of an announcement, which is no request rather than silence, and the
// detail of a duplicate, which is no verifier reject.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import Log from "../pages/log.jsx";
import { getRuns, getRun } from "../api.js";

vi.mock("../appData.jsx", () => ({ useDeviceList: () => [{ name: "kitchen", label: "Кухня" }] }));
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

  it("shows the device's label and filters by it as the device id", async () => {
    render(<Log />);
    expect(await screen.findAllByText("Кухня")).toHaveLength(2);
    fireEvent.change(screen.getByPlaceholderText("Device…"), { target: { value: "Кухня" } });
    await waitFor(() => expect(getRuns).toHaveBeenLastCalledWith({ limit: 100, offset: 0, device: "kitchen" }));
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

  it("reads a dropped follow-up phrase as not a continuation, with the check's probability", async () => {
    getRun.mockResolvedValue({
      ...ROW, result: "rejected", reason: "followup_reject", stt_text: "ну и погодка", llm_text: "",
      attempts: [
        { kind: "followup", provider: "jev", text: "ну и погодка", outcome: "rejected", probability: 0.12, t: 640 },
        { kind: "chain", chain: "smart", stt: "groq", handler: "openrouter", text: "ну и погодка", outcome: "rejected", error: null, detail: null, t_stt: 800, t_handler: 0 },
      ],
    });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    expect((await screen.findByText("Not a continuation (follow-up check).")).parentElement.textContent)
      .toContain("probability 0.12");
    expect(screen.queryByText("Wake-word verify")).toBeNull();
    const items = screen.getAllByTestId("attempt");
    expect(items[0].textContent).toContain("Follow-up check · jev");
    expect(items[0].textContent).toContain("probability 0.12 · 0.64s");
    expect(items[1].textContent).toContain("ну и погодка");
  });

  it("shows a dash, not (silence), as the request of an announcement", async () => {
    getRuns.mockResolvedValue({ runs: [
      { ...ROW, reason: "reminder", stt_text: null, llm_text: "молоко" },
      { ...ROW, id: 8, reason: "endpoint", stt_text: "", llm_text: "" },
    ], total: 2 });
    render(<Log />);
    await screen.findByText("молоко");
    const recognized = screen.getAllByRole("row").filter((tr) => tr.querySelector("td"))
      .map((tr) => tr.querySelectorAll("td")[3].textContent);
    expect(recognized).toEqual(["—", "(silence)"]);
  });

  it("shows a duplicate as dropped, without the verifier card or a silence caption", async () => {
    const dup = { ...ROW, result: "rejected", reason: "duplicate", stt_text: "", llm_text: "", wakeword_score: 1 };
    getRuns.mockResolvedValue({ runs: [dup], total: 1 });
    getRun.mockResolvedValue(dup);
    render(<Log />);
    fireEvent.click(await screen.findByText("Кухня"));
    const banner = await screen.findByText("Dropped as a duplicate:");
    expect(banner.parentElement.textContent).toBe("Dropped as a duplicate: another device kept the phrase");
    expect(screen.queryByText("Wake-word verify")).toBeNull();
    expect(screen.queryByText(/Rejected by wake-word verify/)).toBeNull();
    expect(screen.queryByText("— silence, no speech detected —")).toBeNull();
  });
});
