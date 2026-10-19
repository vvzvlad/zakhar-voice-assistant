// @vitest-environment jsdom
// Integration tests (jsdom) for the Pipeline pages Capture and Speech. They only
// choose what runs and switch stages on and off: every change is a draft until
// the Save in the page's save bar, which sends the whole page as ONE patch;
// "configure →" opens the Providers page (and provider tab) holding the settings; a
// switched-off TTS variant fades its name, not its toggle and buttons.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import { Capture, Speech } from "../pages/pipeline.jsx";
import { useAppData } from "../appData.jsx";
import { setNavigate, pendingTab, clearTab } from "../navStore.js";
import { faded } from "./faded.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));

const WAKE_HINT = "Verify the on-device wake trigger.";
const ACK_HINT = "Play a short confirmation chime.";
const STRESS_HINT = "Place Russian stress marks.";
const FOLLOWUP_HINT = "Keep listening after the answer.";
const onOff = (hint) => ({ type: "object", properties: { enabled: { type: "boolean", description: hint } } });

function makeCatalog({ engines = ["webrtc", "silero"] } = {}) {
  return {
    categories: [
      { id: "vad", selected: "webrtc", providers: engines.map((id) => ({ id, label: `${id} VAD`, values: {} })) },
      { id: "wakeword", selected: "vosk", providers: [{ id: "vosk", schema: onOff(WAKE_HINT), values: { enabled: true, keywords: ["захар"] } }] },
      { id: "stress", selected: "ruaccent", providers: [{ id: "ruaccent", schema: onOff(STRESS_HINT), values: { enabled: true, model_size: "turbo" } }] },
      { id: "followup", selected: "jev", providers: [{ id: "jev", schema: onOff(FOLLOWUP_HINT), values: { enabled: false, window_ms: 5000 } }] },
      {
        id: "tts",
        variants: [{ provider: "yandex", enabled: true }, { provider: "piper", enabled: false }],
        providers: [
          { id: "piper", label: "Piper", values: { voice_path: "models/ru_RU-ruslan-medium.onnx" } },
          { id: "yandex", label: "Yandex SpeechKit", values: { voice: "zahar", role: "neutral" } },
          { id: "fishaudio", label: "Fish Audio", values: { reference_id: "3b5c9f", reference_id_label: "Захар", model: "s2-pro" } },
        ],
      },
    ],
    core: { schema: { $defs: { AckConfig: onOff(ACK_HINT) } }, values: { ack: { enabled: true, sound_path: "" } } },
  };
}

let patch, go;
beforeEach(() => {
  vi.clearAllMocks();
  patch = vi.fn(async () => ({}));
  go = vi.fn();
  setNavigate(go);
  useAppData.mockReturnValue({ catalog: makeCatalog(), config: {}, patch });
});
afterEach(() => { cleanup(); clearTab("vad"); clearTab("tts"); });

const itemOf = (label) => screen.getByText(label, { selector: "b" }).closest(".z-f");
const configure = (label) => fireEvent.click(within(itemOf(label)).getByRole("button", { name: "configure →" }));
async function saved() {
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
  return patch.mock.calls[0][0];
}

describe("Capture page", () => {
  it("keeps the engine choice and toggles as a draft and saves them in one patch", async () => {
    render(<Capture />);
    fireEvent.click(screen.getByText("silero"));
    fireEvent.click(within(itemOf("Wake-word verify")).getByRole("switch"));
    fireEvent.click(within(itemOf("End-of-phrase chime")).getByRole("switch"));
    expect(patch).not.toHaveBeenCalled();
    expect(await saved()).toEqual({
      vad: { selected: "silero" },
      wakeword: { instances: { vosk: { enabled: false } } },
      core: { ack: { enabled: false } },
    });
  });

  it("explains each switch with the backend's description", () => {
    render(<Capture />);
    expect(within(itemOf("Wake-word verify")).getByText(WAKE_HINT)).toBeInTheDocument();
    expect(within(itemOf("End-of-phrase chime")).getByText(ACK_HINT)).toBeInTheDocument();
  });

  it("renders no engine selector for a single engine", () => {
    useAppData.mockReturnValue({ catalog: makeCatalog({ engines: ["webrtc"] }), config: {}, patch });
    render(<Capture />);
    expect(document.querySelector(".z-seg")).toBeNull();
    expect(within(itemOf("VAD engine")).getByText("webrtc VAD")).toBeInTheDocument();
  });

  it("configure links open Providers → VAD, on the saved engine's tab for the engine", () => {
    render(<Capture />);
    configure("VAD engine");
    expect(go).toHaveBeenLastCalledWith("vad");
    expect(pendingTab("vad")).toBe("webrtc");
    clearTab("vad");
    configure("Wake-word verify");
    expect(go).toHaveBeenLastCalledWith("vad");
    expect(pendingTab("vad")).toBeUndefined();
    expect(patch).not.toHaveBeenCalled();
  });

  it("an unsaved engine choice asks before leaving, then opens the saved engine's tab", () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    render(<Capture />);
    fireEvent.click(screen.getByText("silero"));
    configure("VAD engine");
    expect(confirm).toHaveBeenLastCalledWith("Discard unsaved changes?");
    expect(go).not.toHaveBeenCalled();
    expect(pendingTab("vad")).toBeUndefined();
    configure("VAD engine");
    expect(go).toHaveBeenLastCalledWith("vad");
    expect(pendingTab("vad")).toBe("webrtc");
    confirm.mockRestore();
  });
});

describe("Speech page", () => {
  const rows = () => screen.getAllByTestId("tts-variant");

  it("a switched-off variant fades its name, not its toggle and buttons", () => {
    render(<Speech />);
    const piper = rows()[1];
    expect(faded(within(piper).getByText("piper"))).toBe(true);
    for (const el of [within(piper).getByRole("switch"), ...within(piper).getAllByRole("button")]) expect(faded(el)).toBe(false);
  });

  it("saves variant edits and the stress and follow-up switches in one patch carrying the whole variant array", async () => {
    render(<Speech />);
    expect(rows().map((r) => r.querySelector("b").textContent)).toEqual(["yandex", "piper"]);
    fireEvent.click(within(rows()[0]).getByRole("button", { name: "Move down" }));   // piper, yandex
    fireEvent.click(within(rows()[0]).getByRole("switch"));                            // enable piper
    fireEvent.click(screen.getByRole("button", { name: "+ fishaudio" }));              // piper, yandex, fishaudio
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Remove" }));        // piper, fishaudio
    fireEvent.click(within(itemOf("Stress placement")).getByRole("switch"));
    fireEvent.click(within(itemOf("Follow-up")).getByRole("switch"));
    expect(within(itemOf("Follow-up")).getByText(FOLLOWUP_HINT)).toBeInTheDocument();
    expect(patch).not.toHaveBeenCalled();
    expect(await saved()).toEqual({
      tts: { variants: [{ provider: "piper", enabled: true }, { provider: "fishaudio", enabled: true }] },
      stress: { instances: { ruaccent: { enabled: false } } },
      followup: { instances: { jev: { enabled: true } } },
    });
  });

  it("names the voice each variant speaks with", () => {
    render(<Speech />);
    fireEvent.click(screen.getByRole("button", { name: "+ fishaudio" }));
    expect(rows().map((r) => r.querySelector(".z-fh").textContent)).toEqual([
      "Yandex SpeechKit · zahar", "Piper · ru_RU-ruslan-medium.onnx", "Fish Audio · Захар",
    ]);
  });

  it("configure → opens the variant's tab on Providers → TTS", () => {
    render(<Speech />);
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "configure →" }));
    expect(go).toHaveBeenLastCalledWith("tts");
    expect(pendingTab("tts")).toBe("piper");
    expect(patch).not.toHaveBeenCalled();
  });
});
