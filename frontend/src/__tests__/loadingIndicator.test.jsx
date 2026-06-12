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
    // The badge text uses the stage's short name (Accents) for the stress category.
    expect(screen.getByText(/Loading/)).toBeInTheDocument();
    expect(screen.getByText(/Accents/)).toBeInTheDocument();
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
