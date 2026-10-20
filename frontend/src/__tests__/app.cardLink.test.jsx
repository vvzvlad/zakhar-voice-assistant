// @vitest-environment jsdom
// A cross-page "configure" link to one card (navStore.navCard + App.jsx): the
// target page opens on the requested provider tab and App scrolls the card into
// view once the page has rendered; a plain visit afterwards scrolls nowhere.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import App from "../App.jsx";
import { useAppData } from "../appData.jsx";
import { navCard, pendingTab, clearTab } from "../navStore.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../pages/index.js", () => ({ PAGES: {
  capture: () => <button onClick={() => navCard("vad", "card-wakeword", "silero")}>configure wake-word</button>,
  vad: () => <><div id="card-vad">engine</div><div id="card-wakeword">wake-word</div></>,
} }));

let scrolled;
beforeEach(() => {
  useAppData.mockReturnValue({ loading: false, config: {}, system: {}, connected: true, subscribeRuns: () => () => {} });
  window.history.replaceState(null, "", "/capture");
  scrolled = [];
  Element.prototype.scrollIntoView = vi.fn(function scrollIntoView(opts) { scrolled.push([this.id, opts]); });
});
afterEach(() => { cleanup(); clearTab("vad"); delete Element.prototype.scrollIntoView; });

describe("configure link to a card", () => {
  it("opens the page on the tab and scrolls the card to the top, once", () => {
    render(<App />);
    fireEvent.click(screen.getByText("configure wake-word"));
    expect(screen.getByText("wake-word")).toBeInTheDocument();
    expect(pendingTab("vad")).toBe("silero");
    expect(scrolled).toEqual([["card-wakeword", { block: "start" }]]);

    fireEvent.click(screen.getByRole("button", { name: "Capture" }));
    fireEvent.click(screen.getByRole("button", { name: "VAD" }));
    expect(scrolled).toHaveLength(1);
  });
});
