// @vitest-environment jsdom
// Integration tests (jsdom) for the Providers → TTS page: provider tabs (opened
// on the first enabled variant or the tab a configure link asked for), and the
// Stress placement section without its `enabled` switch (the Speech page holds
// it). The page's one Save sends the dirty tabs and the stress settings as ONE patch.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import { TTS } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";
import { navTab, setNavigate } from "../navStore.js";

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
    categories: [
      {
        id: "tts",
        variants: [{ provider: "yandex", enabled: true }, { provider: "piper", enabled: false }],
        providers: [prov("piper", "Piper"), prov("yandex", "Yandex SpeechKit"), prov("fish", "Fish Audio")],
      },
      {
        id: "stress", selected: "ruaccent",
        providers: [{
          id: "ruaccent", label: "RuAccent",
          schema: { type: "object", properties: { enabled: { type: "boolean" }, stress_format: { type: "string", enum: ["plus", "acute", "uppercase"] } } },
          values: { enabled: true, stress_format: "plus" },
        }],
      },
    ],
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

const tabs = () => document.querySelector(".z-seg");
const stressCard = () => screen.getByText("Stress placement").closest(".z-card");

async function saved() {
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
  return patch.mock.calls[0][0];
}

describe("TTS page — provider tabs", () => {
  it("opens on the first enabled variant", () => {
    render(<TTS />);
    expect(within(tabs()).getByText("yandex")).toHaveClass("on");
    expect(screen.getByDisplayValue("yandex-voice")).toBeInTheDocument();
  });

  it("says where each provider is used on the Speech page", () => {
    render(<TTS />);
    expect(screen.getByText("Variant 1 on Speech")).toBeInTheDocument();
    fireEvent.click(within(tabs()).getByText("piper"));
    expect(screen.getByText("Variant 2 on Speech (off)")).toBeInTheDocument();
    fireEvent.click(within(tabs()).getByText("fish"));
    expect(screen.getByText("Not in variants")).toBeInTheDocument();
  });

  it("opens the tab a configure link asked for", () => {
    setNavigate(() => {});
    navTab("tts", "piper");
    render(<TTS />);
    expect(within(tabs()).getByText("piper")).toHaveClass("on");
    expect(screen.getByDisplayValue("piper-voice")).toBeInTheDocument();
  });

  it("saving the settings form patches only that provider's instance", async () => {
    render(<TTS />);
    fireEvent.change(screen.getByDisplayValue("yandex-voice"), { target: { value: "alena" } });
    expect(await saved()).toEqual({ tts: { instances: { yandex: { voice: "alena" } } } });
  });
});

describe("TTS page — stress placement", () => {
  it("shows the stress settings without the on/off switch", () => {
    render(<TTS />);
    expect(within(stressCard()).getByText("acute")).toBeInTheDocument();
    expect(within(stressCard()).queryByRole("switch")).toBeNull();
  });

  it("saves a stress edit together with a TTS tab edit in one patch", async () => {
    render(<TTS />);
    fireEvent.change(screen.getByDisplayValue("yandex-voice"), { target: { value: "alena" } });
    fireEvent.click(within(stressCard()).getByText("acute"));
    expect(await saved()).toEqual({
      tts: { instances: { yandex: { voice: "alena" } } },
      stress: { instances: { ruaccent: { stress_format: "acute" } } },
    });
  });
});
