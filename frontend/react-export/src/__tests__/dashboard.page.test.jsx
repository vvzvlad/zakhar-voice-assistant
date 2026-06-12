// @vitest-environment jsdom
// Render-level tests for the Dashboard service map. Even though providerOf has
// unit tests, this pins the actual render path: the vad node shows the human
// provider label with a new backend, and an old backend (no vad category)
// degrades to "—" without crashing the page.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Dashboard from "../pages/dashboard.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getMetrics: vi.fn(async () => ({})),
  getRuns: vi.fn(async () => ({ runs: [] })),
  openRunsStream: vi.fn(() => () => {}),
}));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

const config = { core: { vad: { silence_ms: 800 }, mcp_servers: [] } };

function catalogWithVad() {
  return {
    categories: [
      {
        id: "vad",
        selected: "webrtc",
        providers: [{ id: "webrtc", label: "WebRTC VAD" }],
      },
    ],
  };
}

// Render and wait for the runs fetch to settle (the empty-state appears),
// so no state update lands outside the test body.
async function renderDashboard() {
  const utils = render(<Dashboard />);
  await screen.findByText("No runs");
  return utils;
}

describe("Dashboard service map", () => {
  it("shows the vad provider label on the vad service node (new backend)", async () => {
    useAppData.mockReturnValue({ catalog: catalogWithVad(), config });
    const { container } = await renderDashboard();
    const provs = [...container.querySelectorAll(".z-svc .prov")];
    expect(provs).toHaveLength(5); // vad, stt, llm, ruaccent, tts
    expect(provs[0].textContent).toBe("WebRTC VAD");
  });

  it('shows "—" and still renders the page when the catalog lacks a vad category (old backend)', async () => {
    useAppData.mockReturnValue({ catalog: { categories: [] }, config });
    const { container } = await renderDashboard();
    expect(screen.getByText("Pipeline overview")).toBeInTheDocument();
    const provs = [...container.querySelectorAll(".z-svc .prov")];
    expect(provs).toHaveLength(5);
    expect(provs[0].textContent).toBe("—");
  });
});
