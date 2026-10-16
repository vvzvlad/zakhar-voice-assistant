// @vitest-environment jsdom
// Integration tests (jsdom) for the Logic page (chains). Every edit (add / reorder /
// delete / field change) changes one draft, and Save sends ONE patch carrying
// the whole core.chains array (PATCH replaces lists). Also pins which per-chain
// controls appear (narrow checkbox, parallel toggle, the last-chain warning),
// the server error banner and the "configure" links.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import Chains from "../pages/chains.jsx";
import { useAppData } from "../appData.jsx";
import { setNavigate, pendingTab, clearTab } from "../navStore.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));

const catalog = {
  categories: [
    { id: "stt", providers: [{ id: "vosk", supports_vocabulary: true }, { id: "groq" }] },
    { id: "llm", providers: [{ id: "simple-nlu", has_vocabulary: true }, { id: "openrouter" }] },
  ],
  core: { schema: {}, values: {} },
};
const FAST = { name: "fast", enabled: true, stt: "vosk", handler: "simple-nlu", narrow: true, parallel: false };
const SMART = { name: "smart", enabled: true, stt: "groq", handler: "openrouter", narrow: false, parallel: true };

let patch;
beforeEach(() => {
  vi.clearAllMocks();
  patch = vi.fn(async () => ({}));
  useAppData.mockReturnValue({ catalog, config: { core: { chains: [FAST, SMART] } }, patch });
});
afterEach(cleanup);

const cards = () => screen.getAllByTestId("chain-card");
async function saved() {
  fireEvent.click(screen.getByText("Save changes"));
  await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
  return patch.mock.calls[0][0];
}
function pick(card, current, next) {
  fireEvent.click(within(card).getByRole("button", { name: current }));
  fireEvent.click(screen.getByRole("option", { name: next }));
}

describe("Chains page — patch shape", () => {
  it("Add chain appends a new chain and Save patches the whole list", async () => {
    render(<Chains />);
    fireEvent.click(screen.getByRole("button", { name: /Add chain/ }));
    expect(cards()).toHaveLength(3);
    expect(patch).not.toHaveBeenCalled();
    expect(await saved()).toEqual({ core: { chains: [FAST, SMART, {
      name: "chain-3", enabled: true, stt: "vosk", handler: "simple-nlu", narrow: true, parallel: false,
    }] } });
  });

  it("Move down reorders the list", async () => {
    render(<Chains />);
    expect(within(cards()[0]).getByRole("button", { name: "Move up" })).toBeDisabled();
    fireEvent.click(within(cards()[0]).getByRole("button", { name: "Move down" }));
    expect(screen.getAllByLabelText("Chain name").map((i) => i.value)).toEqual(["smart", "fast"]);
    expect(await saved()).toEqual({ core: { chains: [SMART, FAST] } });
  });

  it("Delete removes the chain", async () => {
    render(<Chains />);
    fireEvent.click(within(cards()[0]).getByRole("button", { name: "Delete" }));
    expect(await saved()).toEqual({ core: { chains: [SMART] } });
  });

  it("edits name, providers and toggles", async () => {
    render(<Chains />);
    fireEvent.change(within(cards()[1]).getByLabelText("Chain name"), { target: { value: "cloud" } });
    pick(cards()[1], "groq", "vosk");
    const [enabled, parallel] = within(cards()[1]).getAllByRole("switch");
    fireEvent.click(parallel);
    fireEvent.click(enabled);
    expect(await saved()).toEqual({ core: { chains: [FAST, {
      ...SMART, name: "cloud", stt: "vosk", enabled: false, parallel: false,
    }] } });
  });

  it("shows server validation errors like the other save bars", async () => {
    // The shape the backend sends for a duplicate name: a ValueError message, empty detail.
    patch.mockRejectedValueOnce(Object.assign(
      new Error("core.chains: duplicate chain names: ['fast']"), { detail: [] }));
    render(<Chains />);
    fireEvent.change(within(cards()[1]).getByLabelText("Chain name"), { target: { value: "fast" } });
    fireEvent.click(screen.getByText("Save changes"));
    expect(await screen.findByText(/core\.chains: duplicate chain names/)).toBeInTheDocument();
  });
});

describe("Chains page — per-chain controls", () => {
  it("shows the narrow checkbox only for an STT with vocabulary support and a handler with a vocabulary", async () => {
    render(<Chains />);
    expect(within(cards()[0]).getByLabelText("Narrow STT to handler vocabulary")).toBeChecked();
    expect(within(cards()[1]).queryByLabelText("Narrow STT to handler vocabulary")).toBeNull();
    // groq STT + simple-nlu: the STT cannot narrow.
    pick(cards()[1], "openrouter", "simple-nlu");
    expect(within(cards()[1]).queryByLabelText("Narrow STT to handler vocabulary")).toBeNull();
    // A handler without a vocabulary hides it too; the flag itself is kept
    // (the backend ignores it for a pair that cannot narrow).
    pick(cards()[0], "simple-nlu", "openrouter");
    expect(within(cards()[0]).queryByLabelText("Narrow STT to handler vocabulary")).toBeNull();
    const body = await saved();
    expect(body.core.chains[0]).toEqual({ ...FAST, handler: "openrouter" });
  });

  it("unchecking narrow saves narrow:false", async () => {
    render(<Chains />);
    fireEvent.click(within(cards()[0]).getByLabelText("Narrow STT to handler vocabulary"));
    expect((await saved()).core.chains[0]).toEqual({ ...FAST, narrow: false });
  });

  it("offers Recognize in parallel on every chain but the first enabled one", () => {
    render(<Chains />);
    expect(within(cards()[0]).queryByText("Recognize in parallel")).toBeNull();
    expect(within(cards()[1]).getByText("Recognize in parallel")).toBeInTheDocument();
    // With the first chain disabled, the second one is the first to run.
    fireEvent.click(within(cards()[0]).getAllByRole("switch")[0]);
    expect(within(cards()[1]).queryByText("Recognize in parallel")).toBeNull();
  });

  it("warns when the vocabulary handler is the last enabled chain", () => {
    render(<Chains />);
    expect(screen.queryByText(/phrases outside its vocabulary get «не понял»/)).toBeNull();

    fireEvent.click(within(cards()[1]).getAllByRole("switch")[0]); // disable "smart"
    expect(within(cards()[0]).getByText(/phrases outside its vocabulary get «не понял»/)).toBeInTheDocument();
  });
});

describe("Chains page — configure links", () => {
  it("opens the STT page on the chain's provider tab, and the Commands page for the command engine", () => {
    const go = vi.fn();
    setNavigate(go);
    render(<Chains />);
    fireEvent.click(within(cards()[1]).getAllByRole("button", { name: "configure →" })[0]);
    expect(go).toHaveBeenLastCalledWith("stt");
    expect(pendingTab("stt")).toBe("groq");
    clearTab("stt");

    fireEvent.click(within(cards()[0]).getAllByRole("button", { name: "configure →" })[1]);
    expect(go).toHaveBeenLastCalledWith("commands");
    expect(patch).not.toHaveBeenCalled();
  });
});
