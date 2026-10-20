// @vitest-environment jsdom
// Integration tests (jsdom) for the Providers → VAD page. Pins:
//   - the engine settings render WITH a vad category and the core cards either way,
//   - the wake-word and chime cards leave `enabled` out (the Capture page switches them),
//   - the page's one Save sends every dirty card and engine tab as ONE patch, the two
//     core.vad cards staying disjoint,
//   - an engine tab switch sends no patch and keeps the other tab's unsaved edits,
//   - engine tabs named by label, the Capture page's engine marked "in use",
//   - the engine, wake-word and chime sections carry the ids configure links scroll to,
//   - the VAD auto gain sits in the microphone card, not with the thresholds,
//   - the chime field names its sound by file name, listed or not.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import { VAD } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";
import * as api from "../api.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getOptions: vi.fn(async () => ({ options: [] })),
  getChimes: vi.fn(async () => ({ options: [] })),
  playChime: vi.fn(async () => ({ played: ["speaker"], offline: [] })),
}));

afterEach(cleanup);

const ENGINES = [
  {
    id: "webrtc", label: "WebRTC VAD",
    schema: { type: "object", properties: { aggressiveness: { type: "integer", minimum: 0, maximum: 3 } } },
    values: { aggressiveness: 2 },
  },
  {
    id: "silero", label: "Silero VAD",
    schema: { type: "object", properties: { threshold: { type: "number", minimum: 0, maximum: 1 } } },
    values: { threshold: 0.5 },
  },
];

// Minimal-but-realistic catalog mirroring /api/catalog: vad and wakeword
// categories plus core.schema $defs (VadConfig / AckConfig) and their values.
function makeCatalog({ withVadCategory = true } = {}) {
  const wakeword = {
    id: "wakeword", selected: "vosk",
    providers: [{
      id: "vosk", label: "Vosk wake-word",
      schema: { type: "object", properties: { enabled: { type: "boolean" }, keywords: { type: "array", items: { type: "string" } } } },
      values: { enabled: true, keywords: ["захар"] },
    }],
  };
  return {
    categories: withVadCategory
      ? [{ id: "vad", selected: "webrtc", providers: ENGINES }, wakeword]
      : [wakeword],
    core: {
      schema: {
        $defs: {
          VadConfig: {
            type: "object",
            properties: {
              silence_ms: { type: "integer", minimum: 100, maximum: 5000 },
              mic_channel: { type: "integer", enum: [0, 1] },
              mic_auto_gain: { type: "boolean", title: "VAD auto gain" },
              mic_normalize: { type: "boolean" },
              mic_highpass: { type: "boolean" },
            },
          },
          AckConfig: {
            type: "object",
            properties: {
              enabled: { type: "boolean" },
              sound_path: { type: "string", options: "dynamic" },
            },
          },
        },
      },
      values: {
        vad: { silence_ms: 800, mic_channel: 0, mic_auto_gain: false, mic_normalize: true, mic_highpass: false },
        ack: { enabled: true, sound_path: "assets/chimes/Chime_SFX_001.wav" },
      },
    },
  };
}

let patch;
beforeEach(() => {
  vi.clearAllMocks();
  patch = vi.fn(async () => ({}));
});

function renderPage(opts) {
  useAppData.mockReturnValue({ catalog: makeCatalog(opts), config: {}, patch });
  return render(<VAD />);
}

// Flush the async DynamicSelect option load (getChimes) so no state update
// lands outside the test body.
async function settle() {
  await waitFor(() => expect(api.getChimes).toHaveBeenCalled());
}

const cardByTitle = (title) => screen.getByText(title, { selector: "b" }).closest(".z-card");
const tab = (label) => within(document.querySelector(".z-seg")).getByText(label);

function setStepper(card, value) {
  const input = card.querySelector(".z-stepper input");
  fireEvent.change(input, { target: { value } });
  fireEvent.blur(input);
}

async function saved() {
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
  return patch.mock.calls[0][0];
}

describe("VAD page — layout", () => {
  it("shows the selected engine's settings next to the core cards", async () => {
    renderPage();
    expect(tab("Silero VAD")).toBeInTheDocument(); // engine tabs
    expect(cardByTitle("WebRTC VAD")).toBeTruthy();
    expect(screen.getByText("End-pointing thresholds")).toBeInTheDocument();
    await settle();
  });

  it("marks the engine the Capture page selected as in use", async () => {
    renderPage();
    expect(screen.getAllByTestId("in-use").map((m) => m.parentElement)).toEqual([tab("WebRTC VAD")]);
    await settle();
  });

  it("gives the engine, wake-word and chime sections the ids configure links scroll to", async () => {
    renderPage();
    expect(document.getElementById("card-vad")).toContainElement(cardByTitle("WebRTC VAD"));
    expect(document.getElementById("card-wakeword")).toBe(cardByTitle("Wake word verify"));
    expect(document.getElementById("card-ack")).toBe(cardByTitle("End-of-phrase chime"));
    await settle();
  });

  it("says which engine the Capture page selected", async () => {
    renderPage();
    expect(screen.getByText("Selected on Capture")).toBeInTheDocument();
    fireEvent.click(tab("Silero VAD"));
    expect(screen.getByText("Not selected")).toBeInTheDocument();
    await settle();
  });

  it("renders the core cards on an old backend without the vad category", async () => {
    renderPage({ withVadCategory: false });
    expect(screen.getByText("End-pointing thresholds")).toBeInTheDocument();
    expect(screen.queryByText("WebRTC VAD")).toBeNull();
    await settle();
  });

  it("puts the VAD auto gain into the microphone card", async () => {
    renderPage();
    expect(within(cardByTitle("Microphone input & conditioning")).getByText("VAD auto gain")).toBeInTheDocument();
    expect(within(cardByTitle("End-pointing thresholds")).queryByText("VAD auto gain")).toBeNull();
    await settle();
  });

  it("names the chime sound by its file name although the list lacks it", async () => {
    renderPage();
    await settle();
    expect(within(cardByTitle("End-of-phrase chime")).getByRole("button", { name: "Chime_SFX_001.wav" })).toBeInTheDocument();
  });

  it("leaves `enabled` out of the wake-word and chime cards", async () => {
    renderPage();
    expect(within(cardByTitle("Wake word verify")).queryByRole("switch")).toBeNull();
    expect(within(cardByTitle("End-of-phrase chime")).queryByRole("switch")).toBeNull();
    await settle();
  });
});

describe("VAD page — one Save", () => {
  it("saving only the thresholds patches core.vad without any mic_* keys", async () => {
    renderPage();
    await settle();
    setStepper(cardByTitle("End-pointing thresholds"), "900");
    expect(await saved()).toEqual({ core: { vad: { silence_ms: 900 } } });
  });

  it("sends every dirty card and engine tab in one patch", async () => {
    renderPage();
    await settle();
    setStepper(cardByTitle("End-pointing thresholds"), "900");
    fireEvent.click(within(cardByTitle("Microphone input & conditioning")).getAllByRole("switch")[0]);
    fireEvent.change(within(cardByTitle("Wake word verify")).getByDisplayValue("захар"), { target: { value: "алиса" } });
    setStepper(cardByTitle("WebRTC VAD"), "3");
    fireEvent.click(tab("Silero VAD"));
    setStepper(cardByTitle("Silero VAD"), "0.7");
    expect(patch).not.toHaveBeenCalled();

    expect(await saved()).toEqual({
      core: { vad: { silence_ms: 900, mic_channel: 0, mic_auto_gain: true, mic_normalize: true, mic_highpass: false } },
      vad: { instances: { webrtc: { aggressiveness: 3 }, silero: { threshold: 0.7 } } },
      wakeword: { instances: { vosk: { keywords: ["алиса"] } } },
    });
  });

  it("an engine tab switch keeps the other tab's unsaved edit and marks that tab", async () => {
    renderPage();
    await settle();
    setStepper(cardByTitle("WebRTC VAD"), "3");
    fireEvent.click(tab("Silero VAD"));
    expect(within(tab("WebRTC VAD")).getByTestId("unsaved-dot")).toBeInTheDocument();
    fireEvent.click(tab("WebRTC VAD"));
    expect(cardByTitle("WebRTC VAD").querySelector(".z-stepper input")).toHaveValue("3");
    expect(patch).not.toHaveBeenCalled();
  });
});
