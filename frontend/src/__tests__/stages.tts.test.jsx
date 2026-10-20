// @vitest-environment jsdom
// Integration tests (jsdom) for the Providers → TTS page: provider tabs (named by
// label, in the Speech variants' order, the enabled variants marked "in use",
// opened on the first enabled variant or the tab a configure link asked for), the
// cards a configure link scrolls to, and the Stress placement section without its
// `enabled` switch (the Speech page holds it). The page's one Save sends the dirty
// tabs and the stress settings as ONE patch.
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
      {
        id: "followup", selected: "jev",
        providers: [{
          id: "jev", label: "Jev (decision model)",
          schema: { type: "object", properties: { enabled: { type: "boolean" }, window_ms: { type: "integer", title: "Window" } } },
          values: { enabled: false, window_ms: 5000 },
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
const tabLabels = () => Array.from(tabs().querySelectorAll("button")).map((b) => b.firstChild.textContent);
const stressCard = () => screen.getByText("Stress placement").closest(".z-card");
const followupCard = () => screen.getByText("Follow-up").closest(".z-card");

async function saved() {
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
  return patch.mock.calls[0][0];
}

describe("TTS page — provider tabs", () => {
  it("opens on the first enabled variant", () => {
    render(<TTS />);
    expect(within(tabs()).getByText("Yandex SpeechKit")).toHaveClass("on");
    expect(screen.getByDisplayValue("yandex-voice")).toBeInTheDocument();
  });

  it("orders the tabs like the variants, the rest after, and marks the enabled variants in use", () => {
    render(<TTS />);
    expect(tabLabels()).toEqual(["Yandex SpeechKit", "Piper", "Fish Audio"]);
    expect(screen.getAllByTestId("in-use").map((m) => m.parentElement.firstChild.textContent)).toEqual(["Yandex SpeechKit"]);
  });

  it("gives the provider, stress and follow-up sections the ids configure links scroll to", () => {
    render(<TTS />);
    expect(document.getElementById("card-tts")).toContainElement(tabs());
    expect(document.getElementById("card-stress")).toBe(stressCard());
    expect(document.getElementById("card-followup")).toBe(followupCard());
  });

  it("says what the voice test and the voice marker do once, in the card header", () => {
    render(<TTS />);
    for (const title of ["Test voice", "Prompt voice marker"]) {
      const card = screen.getByText(title, { selector: ".z-card-h b" }).closest(".z-card");
      expect(card.querySelector(".z-card-h .sub")).toBeTruthy();
      expect(card.querySelector(".z-foot .z-fh")).toBeNull();
    }
  });

  it("says where each provider is used on the Speech page", () => {
    render(<TTS />);
    expect(screen.getByText("Variant 1 on Speech")).toBeInTheDocument();
    fireEvent.click(within(tabs()).getByText("Piper"));
    expect(screen.getByText("Variant 2 on Speech (off)")).toBeInTheDocument();
    fireEvent.click(within(tabs()).getByText("Fish Audio"));
    expect(screen.getByText("Not in variants")).toBeInTheDocument();
  });

  it("opens the tab a configure link asked for", () => {
    setNavigate(() => {});
    navTab("tts", "piper");
    render(<TTS />);
    expect(within(tabs()).getByText("Piper")).toHaveClass("on");
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

describe("TTS page — follow-up", () => {
  it("shows the follow-up settings without the on/off switch and saves them", async () => {
    render(<TTS />);
    expect(within(followupCard()).queryByRole("switch")).toBeNull();
    const window = within(followupCard()).getByDisplayValue("5000");
    fireEvent.change(window, { target: { value: "4000" } });
    fireEvent.blur(window);
    expect(await saved()).toEqual({ followup: { instances: { jev: { window_ms: 4000 } } } });
  });
});
