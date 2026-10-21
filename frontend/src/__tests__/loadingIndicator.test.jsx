// @vitest-environment jsdom
// Render-level tests for the Topbar's global loading badge: it appears whenever the
// system heartbeat reports a non-empty `reloading` list (a backend model load is in
// flight) and is absent otherwise. Mirrors the appData mocking used by the other
// page tests (vi.mock("../appData.jsx") + useAppData.mockReturnValue).
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { Topbar } from "../components/Topbar.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

describe("Topbar loading badge", () => {
  it("shows the loading badge when a backend category is reloading", () => {
    useAppData.mockReturnValue({
      system: { reloading: ["stress"], uptime_seconds: 5, version: "9.9" },
      connected: true,
    });
    render(<Topbar active="dashboard" />);
    // The badge and its tooltip use the stage's name (Stress) for the stress category.
    const badge = screen.getByText(/Loading/);
    expect(badge.textContent).toContain("Stress");
    expect(badge).toHaveAttribute("title", "Loading: Stress");
  });

  it("names the tools category Tools", () => {
    useAppData.mockReturnValue({
      system: { reloading: ["tools"], uptime_seconds: 5, version: "9.9" },
      connected: true,
    });
    render(<Topbar active="dashboard" />);
    expect(screen.getByText(/Loading/)).toHaveAttribute("title", "Loading: Tools");
  });

  it("hides the loading badge when nothing is reloading", () => {
    useAppData.mockReturnValue({
      system: { reloading: [], uptime_seconds: 5, version: "9.9" },
      connected: true,
    });
    render(<Topbar active="dashboard" />);
    expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
  });
});
