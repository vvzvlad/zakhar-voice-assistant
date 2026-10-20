// @vitest-environment jsdom
// Integration tests (jsdom) for the Pipeline pages Capture and Speech. They only
// choose what runs and switch stages on and off: every change is a draft until
// the Save in the page's save bar, which sends the whole page as ONE patch;
// "configure →" sits right after every item's name and opens the Providers page
// (and provider tab) scrolled to the item's card; providers are named by their
// catalog labels, short in switches and "+" buttons; a switched-off TTS variant
// fades its name, not its toggle and buttons.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import { Capture, Speech } from "../pages/pipeline.jsx";
import { useAppData } from "../appData.jsx";
import { setNavigate, pendingTab, clearTab, takeCard } from "../navStore.js";
import { faded } from "./faded.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));

const WAKE_HINT = "Verify the on-device wake trigger.";
const ACK_HINT = "Play a short confirmation chime.";
const STRESS_HINT = "Place Russian stress marks.";
const FOLLOWUP_HINT = "Keep listening after the answer.";
const onOff = (hint) => ({ type: "object", properties: { enabled: { type: "boolean", description: hint } } });

const ENGINE_LABELS = { webrtc: "WebRTC VAD", silero: "Silero VAD (neural)" };

function makeCatalog({ engines = ["webrtc", "silero"] } = {}) {
  return {
    categories: [
      { id: "vad", selected: "webrtc", providers: engines.map((id) => ({ id, label: ENGINE_LABELS[id], values: {} })) },
      { id: "wakeword", selected: "vosk", providers: [{ id: "vosk", schema: onOff(WAKE_HINT), values: { enabled: true, keywords: ["захар"] } }] },
      { id: "stress", selected: "ruaccent", providers: [{ id: "ruaccent", schema: onOff(STRESS_HINT), values: { enabled: true, model_size: "turbo" } }] },
      { id: "followup", selected: "jev", providers: [{ id: "jev", schema: onOff(FOLLOWUP_HINT), values: { enabled: false, window_ms: 5000 } }] },
      {
        id: "tts",
        variants: [{ provider: "yandex", enabled: true }, { provider: "piper", enabled: false }],
        providers: [
          { id: "piper", label: "Piper (offline)", values: { voice_path: "models/ru_RU-ruslan-medium.onnx" } },
          { id: "yandex", label: "Yandex SpeechKit", values: { voice: "zahar", role: "neutral" } },
          { id: "fishaudio", label: "Fish Audio", values: { reference_id: "3b5c9f", reference_id_label: "Захар", model: "s2-pro" } },
          { id: "silero", label: "Silero (offline)", values: {} },
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
afterEach(() => { cleanup(); clearTab("vad"); clearTab("tts"); takeCard(); });

const itemOf = (label) => screen.getByText(label, { selector: "b" }).closest(".z-f");
const configureOf = (el) => within(el).getByRole("button", { name: "configure →" });
const configure = (label) => fireEvent.click(configureOf(itemOf(label)));
async function saved() {
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
  return patch.mock.calls[0][0];
}

describe("Capture page", () => {
  it("keeps the engine choice and toggles as a draft and saves them in one patch", async () => {
    render(<Capture />);
    fireEvent.click(screen.getByText("Silero VAD"));
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
    expect(within(itemOf("VAD engine")).getByText("WebRTC VAD")).toBeInTheDocument();
  });

  it("names the engines in the switch by their labels without the trailing note", () => {
    render(<Capture />);
    expect(Array.from(document.querySelectorAll(".z-seg button")).map((b) => b.textContent)).toEqual(["WebRTC VAD", "Silero VAD"]);
  });

  it("configure links open Providers → VAD at the item's card, on the saved engine's tab for the engine", () => {
    render(<Capture />);
    configure("VAD engine");
    expect(go).toHaveBeenLastCalledWith("vad");
    expect(pendingTab("vad")).toBe("webrtc");
    expect(takeCard()).toBe("card-vad");
    clearTab("vad");
    configure("Wake-word verify");
    expect(go).toHaveBeenLastCalledWith("vad");
    expect(pendingTab("vad")).toBeUndefined();
    expect(takeCard()).toBe("card-wakeword");
    configure("End-of-phrase chime");
    expect(takeCard()).toBe("card-ack");
    expect(patch).not.toHaveBeenCalled();
  });

  it("an unsaved engine choice asks before leaving, then opens the saved engine's tab", () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    render(<Capture />);
    fireEvent.click(screen.getByText("Silero VAD"));
    configure("VAD engine");
    expect(confirm).toHaveBeenLastCalledWith("Discard unsaved changes?");
    expect(go).not.toHaveBeenCalled();
    expect(pendingTab("vad")).toBeUndefined();
    expect(takeCard()).toBeNull();
    configure("VAD engine");
    expect(go).toHaveBeenLastCalledWith("vad");
    expect(pendingTab("vad")).toBe("webrtc");
    confirm.mockRestore();
  });
});

describe("Speech page", () => {
  const rows = () => screen.getAllByTestId("tts-variant");
  const names = () => rows().map((r) => r.querySelector("b").lastChild.textContent);

  it("a switched-off variant fades its name, not its toggle and buttons", () => {
    render(<Speech />);
    const piper = rows()[1];
    expect(faded(within(piper).getByText("Piper (offline)"))).toBe(true);
    for (const el of [within(piper).getByRole("switch"), ...within(piper).getAllByRole("button")]) expect(faded(el)).toBe(false);
  });

  it("saves variant edits and the stress and follow-up switches in one patch carrying the whole variant array", async () => {
    render(<Speech />);
    expect(names()).toEqual(["Yandex SpeechKit", "Piper (offline)"]);
    fireEvent.click(within(rows()[0]).getByRole("button", { name: "Move down" }));   // piper, yandex
    fireEvent.click(within(rows()[0]).getByRole("switch"));                            // enable piper
    fireEvent.click(screen.getByRole("button", { name: "+ Fish Audio" }));             // piper, yandex, fishaudio
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

  it("offers the missing providers as \"+ <short label>\" buttons", () => {
    render(<Speech />);
    expect(screen.getAllByRole("button", { name: /^\+ / }).map((b) => b.textContent)).toEqual(["+ Fish Audio", "+ Silero"]);
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Remove" }));
    expect(screen.getByRole("button", { name: "+ Piper" })).toBeInTheDocument();
  });

  it("names the voice each variant speaks with", () => {
    render(<Speech />);
    fireEvent.click(screen.getByRole("button", { name: "+ Fish Audio" }));
    expect(names()).toEqual(["Yandex SpeechKit", "Piper (offline)", "Fish Audio"]);
    expect(rows().map((r) => r.querySelector(".z-fh").textContent)).toEqual(["zahar", "ru_RU-ruslan-medium.onnx", "Захар"]);
  });

  it("Remove is the panel's danger button, the other variant actions its plain small buttons", () => {
    render(<Speech />);
    const row = rows()[0];
    expect(within(row).getByRole("button", { name: "Remove" })).toHaveClass("z-btn", "d", "sm");
    for (const name of ["Move up", "Move down", "configure →"]) expect(within(row).getByRole("button", { name })).toHaveClass("z-btn", "g", "sm");
    expect(screen.getByRole("button", { name: "+ Fish Audio" })).toHaveClass("z-btn", "g", "sm");
  });

  it("puts configure → right after the name on variants and on Stress/Follow-up alike", () => {
    render(<Speech />);
    for (const el of [...rows(), itemOf("Stress placement"), itemOf("Follow-up")]) {
      expect(configureOf(el).previousElementSibling.tagName).toBe("B");
    }
  });

  it("configure → opens Providers → TTS at the card: the variant's tab, stress placement, follow-up", () => {
    render(<Speech />);
    fireEvent.click(configureOf(rows()[1]));
    expect(go).toHaveBeenLastCalledWith("tts");
    expect(pendingTab("tts")).toBe("piper");
    expect(takeCard()).toBe("card-tts");
    configure("Stress placement");
    expect(takeCard()).toBe("card-stress");
    configure("Follow-up");
    expect(takeCard()).toBe("card-followup");
    expect(patch).not.toHaveBeenCalled();
  });
});
