// @vitest-environment jsdom
// Integration tests (jsdom) for the provider stage pages (LLM). Pins the
// key={selected} on <SchemaForm> in ProviderStage: dynamic selects fetch their
// option list once on mount, and providers often share an identical field set
// (openrouter / groq both expose `model`), so WITHOUT the key a provider switch
// is reconciled in place and the dropdown keeps showing the PREVIOUS provider's
// options. The key forces a remount, so every DynamicSelect refetches for the
// newly selected provider.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { LLM } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";
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

// Minimal-but-realistic catalog mirroring /api/catalog: an llm category with two
// providers whose schemas have the SAME field set (a dynamic `model` select).
function makeCatalog(selected = "openrouter") {
  const schema = {
    type: "object",
    properties: { model: { type: "string", options: "dynamic" } },
  };
  return {
    categories: [{
      id: "llm",
      selected,
      providers: [
        { id: "openrouter", label: "OpenRouter", schema, values: { model: "openrouter/auto" } },
        { id: "groq", label: "Groq", schema, values: { model: "llama-3-70b" } },
      ],
    }],
    core: { schema: {}, values: {} },
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

describe("LLM page — dynamic options refetch on provider switch", () => {
  it("refetches the model options for the NEW plugin and shows its catalog in the dropdown", async () => {
    mockData(makeCatalog("openrouter"));
    const { rerender } = render(<LLM />);

    // Initial mount fetches the options for the currently selected provider.
    await waitFor(() => expect(api.getOptions).toHaveBeenCalledWith("llm", "openrouter", "model"));

    // Switch the provider: the Selector patches {llm:{selected:"groq"}} …
    fireEvent.click(screen.getByText("groq"));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    expect(patch.mock.calls[0][0]).toEqual({ llm: { selected: "groq" } });

    // … and the app context delivers the updated catalog (patch → refetch in
    // the real appData provider); mirror that by updating the mocked context.
    mockData(makeCatalog("groq"));
    rerender(<LLM />);

    // The remounted DynamicSelect must refetch with the NEW plugin id.
    await waitFor(() => expect(api.getOptions).toHaveBeenCalledWith("llm", "groq", "model"));

    // Open the model dropdown: it must list groq's catalog, not openrouter's.
    // (DynamicSelect deliberately keeps the mount-time current value selectable
    // even when the fetched list omits it, so we only assert that the OLD
    // provider's fetched catalog entries are gone — not the carried-over value.)
    fireEvent.click(screen.getByRole("button", { name: /llama-3-70b/ }));
    const listbox = await screen.findByRole("listbox");
    await waitFor(() => expect(screen.getByText("mixtral-8x7b")).toBeInTheDocument());
    expect(listbox.textContent).not.toContain("anthropic/claude");
  });
});
