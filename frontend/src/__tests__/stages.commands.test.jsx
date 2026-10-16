// @vitest-environment jsdom
// Integration tests (jsdom) for the Providers → Commands page: the vocabulary
// editor comes first, the engine settings card below it starts collapsed, and
// the page's one Save sends the editor's and the engine card's edits together.
// The editor itself is stubbed (it has its own tests); only its wiring is pinned.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { Commands } from "../pages/stages.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({ getOptions: vi.fn(async () => ({ options: [] })) }));
vi.mock("../components/NluCatalogEditor.jsx", () => ({
  default: ({ draft, onChange }) => <input aria-label="Vocabulary" value={draft.aliases}
    onChange={(e) => onChange("aliases", e.target.value)} />,
}));

const catalog = {
  categories: [{
    id: "llm",
    providers: [{
      id: "simple-nlu", label: "Commands",
      schema: { type: "object", properties: {
        aliases: { type: "string" }, actions: { type: "string" }, fuzzy: { type: "boolean" },
      } },
      values: { aliases: "лампа", actions: "", fuzzy: true },
    }],
  }],
  core: { schema: {}, values: {} },
};

let patch;
beforeEach(() => {
  vi.clearAllMocks();
  patch = vi.fn(async () => ({}));
  useAppData.mockReturnValue({ catalog, config: { core: { chains: [] } }, patch });
});
afterEach(cleanup);

const engineHeader = () => screen.getByRole("button", { name: /Engine settings/ });

describe("Commands page", () => {
  it("puts the vocabulary editor first and the engine settings collapsed below it", () => {
    render(<Commands />);
    const editor = screen.getByLabelText("Vocabulary");
    expect(editor.compareDocumentPosition(engineHeader()) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(engineHeader()).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("switch")).toBeNull();
  });

  it("saves the editor's and the engine card's edits in one patch", async () => {
    render(<Commands />);
    fireEvent.change(screen.getByLabelText("Vocabulary"), { target: { value: "люстра" } });
    fireEvent.click(engineHeader());
    fireEvent.click(screen.getByRole("switch"));
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    expect(patch.mock.calls[0][0]).toEqual({
      llm: { instances: { "simple-nlu": { aliases: "люстра", actions: "", fuzzy: false } } },
    });
  });
});
