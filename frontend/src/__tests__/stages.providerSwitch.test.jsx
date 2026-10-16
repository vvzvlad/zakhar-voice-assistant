// @vitest-environment jsdom
// Integration tests (jsdom) for the provider-tab pages (STT / LLM). Pins:
//   - switching a tab only changes the view: it sends NO patch,
//   - the tab's form remounts, so a dynamic select refetches for the NEW provider
//     (openrouter / groq both expose `model`; without the remount the dropdown
//     would keep the previous provider's options),
//   - saving patches only that provider's instance, never routing; edits on
//     several tabs survive the switch and go out in one patch,
//   - the "used in chains" line, the offline command engine kept off the LLM tabs,
//   - a "configure" link from another page opens its provider's tab.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { LLM, STT } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";
import { navTab, setNavigate } from "../navStore.js";
import * as api from "../api.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  // Different model catalogs per plugin so a stale list is detectable.
  getOptions: vi.fn(async (cat, plugin) => (
    plugin === "groq"
      ? { options: ["llama-3-70b", "mixtral-8x7b"] }
      : { options: ["openrouter/auto", "anthropic/claude"] }
  )),
  getChimes: vi.fn(async () => ({ options: [] })),
  playChime: vi.fn(async () => ({ played: ["speaker"], offline: [] })),
}));

afterEach(cleanup);

// Catalog mirroring /api/catalog: llm and stt categories WITHOUT `selected`; the
// two llm providers share the same field set (a dynamic `model` select).
function makeCatalog() {
  const schema = { type: "object", properties: { model: { type: "string", options: "dynamic" } } };
  return {
    categories: [
      {
        id: "llm",
        providers: [
          { id: "openrouter", label: "OpenRouter", schema, values: { model: "openrouter/auto" } },
          { id: "groq", label: "Groq", schema, values: { model: "llama-3-70b" } },
          { id: "simple-nlu", label: "Commands", schema: { type: "object", properties: {} }, values: {}, has_vocabulary: true },
        ],
      },
      {
        id: "stt",
        providers: [
          { id: "vosk", label: "Vosk", schema: { type: "object", properties: {} }, values: {}, supports_vocabulary: true },
          { id: "groq", label: "Groq Whisper", schema: { type: "object", properties: {} }, values: {} },
        ],
      },
    ],
    core: { schema: {}, values: {} },
  };
}

const config = {
  core: {
    chains: [
      { name: "fast", enabled: true, stt: "vosk", handler: "simple-nlu", narrow: true, parallel: false },
      { name: "smart", enabled: true, stt: "groq", handler: "openrouter", narrow: false, parallel: true },
    ],
  },
};

let patch;
beforeEach(() => {
  vi.clearAllMocks();
  patch = vi.fn(async () => ({}));
  useAppData.mockReturnValue({ catalog: makeCatalog(), config, patch });
});

describe("LLM page — provider tabs", () => {
  it("switching a tab sends no patch and refetches the model options for the new provider", async () => {
    render(<LLM />);
    // The first enabled chain uses simple-nlu, which is not an LLM tab, so the
    // page opens on the first LLM provider.
    await waitFor(() => expect(api.getOptions).toHaveBeenCalledWith("llm", "openrouter", "model"));

    fireEvent.click(screen.getByText("groq"));
    await waitFor(() => expect(api.getOptions).toHaveBeenCalledWith("llm", "groq", "model"));
    await Promise.resolve();
    expect(patch).not.toHaveBeenCalled();

    // The remounted dropdown lists groq's catalog, not openrouter's.
    fireEvent.click(screen.getByRole("button", { name: /llama-3-70b/ }));
    const listbox = await screen.findByRole("listbox");
    await waitFor(() => expect(screen.getByText("mixtral-8x7b")).toBeInTheDocument());
    expect(listbox.textContent).not.toContain("anthropic/claude");
  });

  it("keeps the offline command engine off the tabs and shows which chains use the provider", async () => {
    render(<LLM />);
    expect(screen.queryByText("simple-nlu")).toBeNull();
    expect(screen.getByText("Used in chains: smart")).toBeInTheDocument();
    fireEvent.click(screen.getByText("groq"));
    expect(screen.getByText("Not used in any chain")).toBeInTheDocument();
    await waitFor(() => expect(api.getOptions).toHaveBeenCalledWith("llm", "groq", "model"));
  });

  it("saving patches only the shown provider's instance", async () => {
    render(<LLM />);
    fireEvent.click(screen.getByText("groq"));
    fireEvent.click(screen.getByRole("button", { name: /llama-3-70b/ }));
    fireEvent.click(await screen.findByText("mixtral-8x7b"));
    fireEvent.click(screen.getByText("Save changes"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    expect(patch.mock.calls[0][0]).toEqual({ llm: { instances: { groq: { model: "mixtral-8x7b" } } } });
  });

  it("keeps an edit on another tab and saves both tabs in one patch", async () => {
    render(<LLM />);
    fireEvent.click(screen.getByRole("button", { name: /openrouter\/auto/ }));
    fireEvent.click(await screen.findByText("anthropic/claude"));
    fireEvent.click(screen.getByText("groq"));
    expect(screen.getByText("openrouter").querySelector("[data-testid=unsaved-dot]")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /llama-3-70b/ }));
    fireEvent.click(await screen.findByText("mixtral-8x7b"));
    fireEvent.click(screen.getByText("Save changes"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    expect(patch.mock.calls[0][0]).toEqual({ llm: { instances: {
      openrouter: { model: "anthropic/claude" }, groq: { model: "mixtral-8x7b" },
    } } });
  });
});

describe("STT page — provider tabs", () => {
  it("opens on the first enabled chain's STT", () => {
    render(<STT />);
    expect(screen.getByText("Used in chains: fast")).toBeInTheDocument();
    expect(screen.getByText("vosk")).toHaveClass("on");
  });

  it("opens the tab a configure link asked for", () => {
    setNavigate(() => {});
    navTab("stt", "groq");
    render(<STT />);
    expect(screen.getByText("groq")).toHaveClass("on");
    expect(screen.getByText("Used in chains: smart")).toBeInTheDocument();
    // The request is consumed: a later plain visit opens the default tab.
    cleanup();
    render(<STT />);
    expect(screen.getByText("vosk")).toHaveClass("on");
  });
});
