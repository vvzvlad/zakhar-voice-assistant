// @vitest-environment jsdom
// Leaving a page with unsaved changes (App.jsx + navStore.js): a sidebar click and
// the browser's Back ask first and stay on "cancel"; closing the tab gets the
// browser's prompt only while there is something unsaved.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import App from "../App.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
// Capture holds unsaved changes; Logic has none.
vi.mock("../pages/index.js", async () => {
  const { FormPage } = await import("../components/primitives.jsx");
  const form = { dirty: true, saving: false, err: null, save: () => {} };
  return { PAGES: {
    capture: () => <FormPage form={form}><p>Capture page</p></FormPage>,
    logic: () => <p>Logic page</p>,
  } };
});

let confirm;
beforeEach(() => {
  useAppData.mockReturnValue({ loading: false, config: {}, system: {}, connected: true, subscribeRuns: () => () => {} });
  window.history.replaceState(null, "", "/capture");
  confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
});
afterEach(() => { cleanup(); confirm.mockRestore(); });

const navItem = (label) => screen.getByRole("button", { name: label });
const unloadPrevented = () => {
  const e = new Event("beforeunload", { cancelable: true });
  window.dispatchEvent(e);
  return e.defaultPrevented;
};

describe("leaving a page with unsaved changes", () => {
  it("a sidebar click asks, stays on cancel and leaves on discard", () => {
    render(<App />);
    fireEvent.click(navItem("Capture"));      // its own item: nothing to ask
    expect(confirm).not.toHaveBeenCalled();
    fireEvent.click(navItem("Logic"));
    expect(confirm).toHaveBeenCalledWith("Discard unsaved changes?");
    expect(screen.getByText("Capture page")).toBeInTheDocument();
    expect(unloadPrevented()).toBe(true);
    fireEvent.click(navItem("Logic"));
    expect(screen.getByText("Logic page")).toBeInTheDocument();
    expect(window.location.pathname).toBe("/logic");
    expect(unloadPrevented()).toBe(false);
  });

  it("the browser's Back asks too and restores the address on cancel", () => {
    render(<App />);
    const back = () => {
      window.history.pushState(null, "", "/logic");
      fireEvent(window, new PopStateEvent("popstate"));
    };
    back();
    expect(screen.getByText("Capture page")).toBeInTheDocument();
    expect(window.location.pathname).toBe("/capture");
    back();
    expect(screen.getByText("Logic page")).toBeInTheDocument();
  });
});
