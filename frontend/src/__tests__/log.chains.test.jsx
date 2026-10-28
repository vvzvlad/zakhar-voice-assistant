// @vitest-environment jsdom
// Integration tests (jsdom) for the chain parts of the Request Log: the Chain
// column, the chain dropdown filter forwarded to /api/runs, and the run detail listing
// the chain and TTS attempts in the order they ran, with why a handler declined.
// Also the device column, which shows a device's label and filters by it from a
// dropdown, the Result dropdown, the request of an announcement, which is no request
// rather than silence, the detail of a duplicate, which is no verifier reject, and the
// detail's dated header, its copy button, its Metadata and what its Total spans.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import Log from "../pages/log.jsx";
import { getRuns, getRun } from "../api.js";
import { fmtTime } from "../runsModel.js";

vi.mock("../appData.jsx", () => ({
  useDeviceList: () => [{ name: "kitchen", label: "Кухня" }],
  useChainList: () => [{ name: "fast" }, { name: "smart" }],
  useProviderLabel: () => (cat, id) => ({ vosk: "Vosk (offline)", "simple-nlu": "Simple NLU (no LLM)" })[id] || id,
}));
vi.mock("../api.js", () => ({
  getRuns: vi.fn(),
  getRun: vi.fn(),
  runLink: vi.fn((id) => `http://panel:8201/api/runs/${id}`),
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

// Pick `option` in the filter dropdown captioned `label`, after checking its options.
function pick(label, options, option) {
  fireEvent.click(within(screen.getByTestId(`filter-${label}`)).getByRole("button"));
  expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(options);
  fireEvent.click(screen.getByRole("option", { name: option }));
}

describe("Request log — chains", () => {
  it("shows a Chain column and forwards the chain filter to /api/runs", async () => {
    render(<Log />);
    expect((await screen.findAllByTestId("run-chain")).map((td) => td.textContent)).toEqual(["smart", "—"]);

    pick("chain", ["All", "fast", "smart"], "fast");
    await waitFor(() => expect(getRuns).toHaveBeenLastCalledWith({ limit: 100, offset: 0, chain: "fast" }));
  });

  it("shows the device's label and filters by it as the device id", async () => {
    render(<Log />);
    expect(await screen.findAllByText("Кухня")).toHaveLength(2);
    pick("device", ["All", "Кухня"], "Кухня");
    await waitFor(() => expect(getRuns).toHaveBeenLastCalledWith({ limit: 100, offset: 0, device: "kitchen" }));
  });

  it("filters by result from a dropdown", async () => {
    render(<Log />);
    await screen.findAllByTestId("run-chain");
    pick("result", ["All", "Errors", "OK", "Rejected"], "Rejected");
    await waitFor(() => expect(getRuns).toHaveBeenLastCalledWith({ limit: 100, offset: 0, result: "rejected" }));
  });

  it("dates the detail's header, keeps the header's fields out of Metadata, shows the wake word score in percent and its time once, and says what Total spans", async () => {
    getRun.mockResolvedValue({ ...ROW, ts: 1760854865, reason: "endpoint", t_vad: 1500, t_wakeword: 230, wakeword_score: 0.9, t_stt: 400, t_total: 9000 });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    await screen.findByText("Metadata");
    expect(within(screen.getByRole("dialog")).getByRole("heading").textContent).toBe("Кухня · Oct 19 2025 06:21:05");
    expect([...document.querySelectorAll(".z-kv .k")].map((k) => k.textContent)).toEqual(["Score", "Verdict", "Chain"]);
    expect(document.querySelector(".z-kv .v").textContent).toBe("90%");
    expect(screen.getByText("Wake word verify", { selector: ".lbl" }).closest(".z-grow").querySelector(".ms").textContent).toBe("230 ms");
    expect(screen.queryByText("0.23s")).toBeNull();
    expect(screen.getByText("end of phrase → end of reply playback").nextSibling.textContent).toBe("9.00s");
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("shows only Metadata rows with a value, no section without one, and the reply's speech as TTS audio", async () => {
    // An announcement with its recording but no chain, model, tokens or reply speech: no Metadata.
    getRun.mockResolvedValueOnce({ ...ROW, chain: null, reason: "say", has_audio: 1 });
    getRun.mockResolvedValueOnce({ ...ROW, id: 8, audio_bytes: 12288, audio_fmt: "wav" });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    await screen.findByText("Utterance audio");
    expect(screen.queryByText("Metadata")).toBeNull();
    fireEvent.click(screen.getAllByText("Солнечно")[1]);
    await screen.findByText("Metadata");
    expect([...document.querySelectorAll(".z-kv")].map((kv) => kv.textContent)).toEqual(["Chainsmart", "TTS audio12 kB · wav"]);
  });

  it("keeps the model and tokens out of Metadata when the rounds header shows them", async () => {
    getRun.mockResolvedValue({ ...ROW, model: "haiku", tokens: 120, rounds: [{ round: 1, note: "", tokens: 120, calls: [], content: "Солнечно" }] });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    await screen.findByText("Metadata");
    expect([...document.querySelectorAll(".z-kv .k")].map((k) => k.textContent)).not.toContain("Model");
    expect(screen.getByText("haiku · 120 tokens")).toBeInTheDocument();
  });

  it("names the handler and how the run ended in words, not ids", async () => {
    getRun.mockResolvedValue({ ...ROW, reason: "endpoint", model: "simple-nlu/none", tokens: 0, rounds: [{ round: 1, note: "", tokens: 0, calls: [], content: "Солнечно" }] });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    await screen.findByText("Metadata");
    expect(screen.getByText("Simple NLU (no LLM) · 0 tokens")).toBeInTheDocument();
    expect(screen.getByText("id 7 · end of phrase")).toBeInTheDocument();
  });

  it("names the run of a timer, an alarm and a deferred command in words", async () => {
    for (const [reason, label] of [["timer", "timer"], ["alarm", "alarm"], ["command", "deferred command"]]) {
      getRun.mockResolvedValue({ ...ROW, reason });
      render(<Log />);
      fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
      expect(await screen.findByText(`id 7 · ${label}`)).toBeInTheDocument();
      cleanup();
    }
  });

  it("copies the device, time, id and the run's link, without how the run ended", async () => {
    const writeText = vi.fn().mockResolvedValue();
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    getRun.mockResolvedValue({ ...ROW, reason: "endpoint" });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    fireEvent.click(await screen.findByRole("button", { name: "Copy" }));
    await screen.findByRole("button", { name: "Copied" });
    expect(writeText).toHaveBeenCalledWith(`id 7, ${fmtTime(0, true)}\ndevice Кухня, http://panel:8201/api/runs/7`);
  });

  it("copies through a hidden text field over plain http, where the browser has no Clipboard API", async () => {
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
    let copied;
    document.execCommand = vi.fn(() => { copied = document.activeElement.value; return true; });
    try {
      getRun.mockResolvedValue({ ...ROW, reason: "endpoint" });
      render(<Log />);
      fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
      fireEvent.click(await screen.findByRole("button", { name: "Copy" }));
      await screen.findByRole("button", { name: "Copied" });
      expect(document.execCommand).toHaveBeenCalledWith("copy");
      expect(copied).toBe(`id 7, ${fmtTime(0, true)}\ndevice Кухня, http://panel:8201/api/runs/7`);
      expect(document.querySelector("textarea")).toBeNull();
    } finally {
      delete document.execCommand;
    }
  });

  it("names the failed stage with a capital letter", async () => {
    getRun.mockResolvedValue({ ...ROW, result: "error", error_stage: "pipeline", error_text: "boom" });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    expect(await screen.findByText("Pipeline error.")).toBeInTheDocument();
  });

  it("shows an LLM's upstream model id as it is", async () => {
    getRun.mockResolvedValue({ ...ROW, model: "anthropic/claude-haiku-4.5", tokens: 120, rounds: [{ round: 1, note: "", tokens: 120, calls: [], content: "Солнечно" }] });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    await screen.findByText("Metadata");
    expect(screen.getByText("anthropic/claude-haiku-4.5 · 120 tokens")).toBeInTheDocument();
  });

  it("lists the run's attempts in order with text, outcome, error and timings", async () => {
    getRun.mockResolvedValue({
      ...ROW,
      attempts: [
        { kind: "chain", chain: "fast", stt: "vosk", handler: "simple-nlu", text: "какая погода", outcome: "declined", error: null, detail: "no alias phrase matched", t_stt: 310, t_handler: 12 },
        { kind: "chain", chain: "smart", stt: "groq", handler: "openrouter", text: "какая погода", outcome: "answered", error: null, detail: null, t_stt: 800, t_handler: 4100 },
        { kind: "chain", chain: "smart", stt: "groq", handler: "openrouter", text: null, outcome: "stt_error", error: "429", detail: null, t_stt: 200, t_handler: 0 },
        { kind: "tts", provider: "yandex", outcome: "error", error: "timeout", t: 6000 },
        { kind: "tts", provider: "piper", outcome: "spoke", error: null, t: 900 },
      ],
    });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    await screen.findByText("Attempts");

    const items = screen.getAllByTestId("attempt");
    expect(items).toHaveLength(5);
    expect(items[0].textContent).toContain("fast · Vosk (offline) → Simple NLU (no LLM)");
    expect(items[0].textContent).toContain("STT 0.31s · handler 0.01s");
    expect(items[0].textContent).toContain("declined");
    expect(items[0].textContent).toContain("какая погода");
    // Why the handler declined shows under its attempt; an answer has no detail.
    expect(screen.getAllByTestId("attempt-detail").map((d) => d.textContent)).toEqual(["no alias phrase matched"]);
    expect(items[1].textContent).toContain("answered");
    // An outcome id that is not a word shows in words.
    expect(items[2].textContent).toContain("STT error");
    expect(items[2].textContent).not.toContain("stt_error");
    expect(items[3].textContent).toContain("TTS · yandex");
    expect(items[3].textContent).toContain("timeout");
    expect(items[3].textContent).toContain("6.00s");
    expect(items[4].textContent).toContain("spoke");
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
      .toContain("probability 12%");
    expect(screen.queryByText("Wake word verify")).toBeNull();
    const items = screen.getAllByTestId("attempt");
    expect(items[0].textContent).toContain("Follow-up check · jev");
    expect(items[0].textContent).toContain("probability 12% · 0.64s");
    expect(items[1].textContent).toContain("ну и погодка");
  });

  it("shows an answer's tool calls in the context as the calls, not an empty assistant line", async () => {
    getRun.mockResolvedValue({ ...ROW, request: { system_prompt: "p", user_text: "сколько времени", tools: [], context: [
      { role: "user", content: "включи лампу" },
      { role: "assistant", content: "", tool_calls: [{ id: "call_1", type: "function", function: { name: "set_light", arguments: '{"device_id": "table_light", "state": "on"}' } }] },
      { role: "tool", tool_call_id: "call_1", content: "applied" },
      { role: "assistant", content: "Готово." },
    ] } });
    render(<Log />);
    fireEvent.click((await screen.findAllByText("Солнечно"))[0]);
    const block = (await screen.findByText(/Context · 4 msg/)).closest(".z-tool");
    expect([...block.querySelectorAll(".z-code.text")].map((d) => d.textContent)).toEqual([
      "user: включи лампу", 'assistant: set_light({"device_id": "table_light", "state": "on"})', "tool: applied", "assistant: Готово."]);
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
    expect(screen.queryByText("Wake word verify")).toBeNull();
    expect(screen.queryByText(/Rejected by wake word verify/)).toBeNull();
    expect(screen.queryByText("— silence, no speech detected —")).toBeNull();
  });
});
