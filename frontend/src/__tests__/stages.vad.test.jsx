// @vitest-environment jsdom
// Integration tests (jsdom) for the VAD page. Pins:
//   - the `vadCat &&` guard: ProviderCard renders WITH a vad category and the
//     End-pointing card renders either way (new + old backend),
//   - an old backend (no vad category) must not throw inside ProviderCard,
//   - the three forms on the page patch DISJOINT subsets of core.vad,
//   - the Provider selector patches {vad:{selected}} only (no `instances`), and
//     re-selecting the current provider does not patch at all.
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

// Minimal-but-realistic catalog mirroring /api/catalog: a vad category with
// provider schema/values plus core.schema $defs (VadConfig / AckConfig) and
// core.values for the vad and ack sub-sections.
function makeCatalog({ withVadCategory = true, providers } = {}) {
  const providerSchema = {
    type: "object",
    properties: { aggressiveness: { type: "integer", minimum: 0, maximum: 3 } },
  };
  const vadProviders = providers || [
    { id: "webrtc", label: "WebRTC VAD", schema: providerSchema, values: { aggressiveness: 2 } },
  ];
  return {
    categories: withVadCategory
      ? [{ id: "vad", selected: "webrtc", providers: vadProviders }]
      : [],
    core: {
      schema: {
        $defs: {
          VadConfig: {
            type: "object",
            properties: {
              silence_ms: { type: "integer", minimum: 100, maximum: 5000 },
              mic_channel: { type: "integer", enum: [0, 1] },
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
        vad: { silence_ms: 800, mic_channel: 0, mic_normalize: true, mic_highpass: false },
        ack: { enabled: true, sound_path: "" },
      },
    },
  };
}

let patch;
beforeEach(() => {
  vi.clearAllMocks();
  patch = vi.fn(async () => ({}));
});

function mockData(catalog) {
  useAppData.mockReturnValue({ catalog, config: {}, patch });
}

// Flush the async DynamicSelect option load (getChimes) so no state update
// lands outside the test body.
async function settle() {
  await waitFor(() => expect(api.getChimes).toHaveBeenCalled());
}

// Locate the .z-card wrapper of a card by its title text.
function cardByTitle(title) {
  const card = screen.getByText(title).closest(".z-card");
  expect(card).toBeTruthy();
  return card;
}

describe("VAD page — backend catalog shapes", () => {
  it("renders BOTH the ProviderCard and the End-pointing card with a new backend (vad category present)", async () => {
    mockData(makeCatalog());
    render(<VAD />);
    // ProviderCard: card title carries the provider label, plus the Provider selector.
    expect(screen.getAllByText("WebRTC VAD").length).toBeGreaterThan(0);
    expect(screen.getByText("Provider")).toBeInTheDocument();
    // End-pointing card renders alongside (guard must not be inverted).
    expect(screen.getByText("End-pointing thresholds")).toBeInTheDocument();
    await settle();
  });

  it("renders without throwing on an old backend (no vad category): End-pointing card present, no ProviderCard", async () => {
    mockData(makeCatalog({ withVadCategory: false }));
    expect(() => render(<VAD />)).not.toThrow();
    expect(screen.getByText("End-pointing thresholds")).toBeInTheDocument();
    // No ProviderCard at all: no Provider selector, no provider label anywhere.
    expect(screen.queryByText("Provider")).toBeNull();
    expect(screen.queryByText("WebRTC VAD")).toBeNull();
    await settle();
  });
});

describe("VAD page — disjoint core.vad saves", () => {
  it("saving the End-pointing card patches core.vad WITHOUT any mic_* keys", async () => {
    mockData(makeCatalog());
    render(<VAD />);
    await settle();

    const card = cardByTitle("End-pointing thresholds");
    // silence_ms renders as a Stepper; commit a new value via change + blur.
    const input = card.querySelector(".z-stepper input");
    expect(input).toBeTruthy();
    fireEvent.change(input, { target: { value: "900" } });
    fireEvent.blur(input);

    fireEvent.click(within(card).getByText("Save changes"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    const payload = patch.mock.calls[0][0];
    expect(payload).toEqual({ core: { vad: { silence_ms: 900 } } });
    expect(Object.keys(payload.core.vad).some((k) => k.startsWith("mic_"))).toBe(false);
  });

  it("saving the Microphone card patches core.vad with ONLY the mic_* draft (no silence_ms)", async () => {
    mockData(makeCatalog());
    render(<VAD />);
    await settle();

    const card = cardByTitle("Microphone input & conditioning");
    // The mic card's toggles are mic_normalize then mic_highpass; flip the first.
    const switches = within(card).getAllByRole("switch");
    expect(switches.length).toBeGreaterThan(0);
    fireEvent.click(switches[0]);

    fireEvent.click(within(card).getByText("Save changes"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    const payload = patch.mock.calls[0][0];
    expect(payload).toEqual({
      core: { vad: { mic_channel: 0, mic_normalize: false, mic_highpass: false } },
    });
    expect(payload.core.vad).not.toHaveProperty("silence_ms");
  });
});

describe("VAD page — ProviderCard provider switch", () => {
  const twoProviders = [
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

  it("selecting another provider patches exactly {vad:{selected:newId}} with no `instances`", async () => {
    mockData(makeCatalog({ providers: twoProviders }));
    render(<VAD />);
    await settle();

    fireEvent.click(screen.getByText("silero"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    const payload = patch.mock.calls[0][0];
    expect(payload).toEqual({ vad: { selected: "silero" } });
    expect(payload.vad).not.toHaveProperty("instances");
  });

  it("re-selecting the CURRENT provider does not call patch", async () => {
    mockData(makeCatalog({ providers: twoProviders }));
    render(<VAD />);
    await settle();

    fireEvent.click(screen.getByText("webrtc"));
    // Give a microtask tick so an accidental async patch would surface.
    await Promise.resolve();
    expect(patch).not.toHaveBeenCalled();
  });
});
