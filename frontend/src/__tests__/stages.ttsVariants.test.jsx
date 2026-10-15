// @vitest-environment jsdom
// Integration tests (jsdom) for the TTS page variant list: reorder / toggle /
// add / remove edit one draft that is saved as ONE patch carrying the whole
// tts.variants array (PATCH replaces lists), and "Configure" only switches the
// settings tab below without patching.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import { TTS } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getOptions: vi.fn(async () => ({ options: [] })),
  getChimes: vi.fn(async () => ({ options: [] })),
  playChime: vi.fn(async () => ({ played: [], offline: [] })),
  streamTtsVoice: vi.fn(),
}));
vi.mock("../streamAudio.js", () => ({ playAudioResponse: vi.fn() }));

function makeCatalog() {
  const prov = (id, label) => ({
    id, label, schema: { type: "object", properties: { voice: { type: "string" } } }, values: { voice: `${id}-voice` },
  });
  return {
    categories: [{
      id: "tts",
      variants: [{ provider: "yandex", enabled: true }, { provider: "piper", enabled: false }],
      providers: [prov("piper", "Piper"), prov("yandex", "Yandex SpeechKit"), prov("fish", "Fish Audio")],
    }],
    core: { schema: {}, values: {} },
  };
}

let patch;
beforeEach(() => {
  vi.clearAllMocks();
  patch = vi.fn(async () => ({}));
  useAppData.mockReturnValue({ catalog: makeCatalog(), config: {}, patch });
});
afterEach(cleanup);

const rows = () => screen.getAllByTestId("tts-variant");
const tabs = () => document.querySelector(".z-seg");

describe("TTS page — variants", () => {
  it("saves reorder, toggle, add and remove as one patch of the whole variant array", async () => {
    render(<TTS />);
    expect(rows().map((r) => r.querySelector("b").textContent)).toEqual(["yandex", "piper"]);

    fireEvent.click(within(rows()[0]).getByRole("button", { name: "Move down" }));   // piper, yandex
    fireEvent.click(within(rows()[0]).getByRole("switch"));                            // enable piper
    fireEvent.click(screen.getByRole("button", { name: "+ fish" }));                   // piper, yandex, fish
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Remove" }));        // piper, fish
    expect(patch).not.toHaveBeenCalled();

    fireEvent.click(within(screen.getByText("Variants").closest(".z-card")).getByText("Save changes"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    expect(patch.mock.calls[0][0]).toEqual({
      tts: { variants: [{ provider: "piper", enabled: true }, { provider: "fish", enabled: true }] },
    });
  });

  it("opens on the first enabled variant and Configure switches the settings tab without a patch", () => {
    render(<TTS />);
    expect(within(tabs()).getByText("yandex")).toHaveClass("on");
    expect(screen.getByDisplayValue("yandex-voice")).toBeInTheDocument();

    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Configure" }));
    expect(within(tabs()).getByText("piper")).toHaveClass("on");
    expect(screen.getByDisplayValue("piper-voice")).toBeInTheDocument();
    expect(patch).not.toHaveBeenCalled();
  });

  it("saving the settings form patches only that provider's instance", async () => {
    render(<TTS />);
    fireEvent.change(screen.getByDisplayValue("yandex-voice"), { target: { value: "alena" } });
    fireEvent.click(within(screen.getByText("Yandex SpeechKit", { selector: "b" }).closest(".z-card")).getByText("Save changes"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    expect(patch.mock.calls[0][0]).toEqual({ tts: { instances: { yandex: { voice: "alena" } } } });
  });
});
