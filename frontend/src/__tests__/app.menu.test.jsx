// @vitest-environment jsdom
// The narrow-screen menu (App.jsx, Sidebar, Topbar): the top bar's menu button opens the
// sidebar over the page with a scrim; picking an item navigates and closes it, and the
// scrim or Escape closes it too. The narrow-screen styles themselves are not under test.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import App from "../App.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../pages/index.js", () => ({ PAGES: {
  capture: () => <p>Capture page</p>,
  logic: () => <p>Logic page</p>,
} }));

beforeEach(() => {
  useAppData.mockReturnValue({ loading: false, config: {}, system: {}, connected: true, subscribeRuns: () => () => {} });
  window.history.replaceState(null, "", "/capture");
});
afterEach(cleanup);

const open = () => fireEvent.click(screen.getByRole("button", { name: "Menu" }));
const isOpen = (container) => container.querySelector(".z-side").classList.contains("open") && !!screen.queryByTestId("menu-scrim");

describe("narrow-screen menu", () => {
  it("opens from the top bar and closes on a picked item, the scrim and Escape", () => {
    const { container } = render(<App />);
    expect(isOpen(container)).toBe(false);
    open();
    expect(isOpen(container)).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Logic" }));
    expect(screen.getByText("Logic page")).toBeInTheDocument();
    expect(isOpen(container)).toBe(false);
    expect(screen.queryByTestId("menu-scrim")).toBeNull();

    open();
    fireEvent.click(screen.getByTestId("menu-scrim"));
    expect(container.querySelector(".z-side")).not.toHaveClass("open");

    open();
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(container.querySelector(".z-side")).not.toHaveClass("open");
  });
});
