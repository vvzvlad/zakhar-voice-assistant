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
import { getMetrics } from "../api.js";

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
    expect(provs).toHaveLength(5); // vad, wakeword, chains, stress, tts
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

  it("shows one Chains row per enabled chain with its share of served requests and average time", async () => {
    getMetrics.mockResolvedValueOnce({
      per_stage_avg_ms: {},
      per_chain: { fast: { served: 3, avg_ms: 400 }, smart: { served: 1, avg_ms: 9000 } },
    });
    const chains = [
      { name: "fast", enabled: true, stt: "vosk", handler: "simple-nlu" },
      { name: "off", enabled: false, stt: "groq", handler: "openrouter" },
      { name: "smart", enabled: true, stt: "groq", handler: "openrouter" },
    ];
    useAppData.mockReturnValue({ catalog: catalogWithVad(), config: { core: { ...config.core, chains } } });
    await renderDashboard();
    const rows = screen.getAllByTestId("chain-row").map((r) => r.textContent);
    expect(rows).toEqual(["vosk → simple-nlu75%0.40s", "groq → openrouter25%9.00s"]);
    expect(screen.getByText("2 enabled")).toBeInTheDocument();
  });

  it("shows the TTS node's first enabled variant and the number of fallbacks", async () => {
    const tts = {
      id: "tts",
      variants: [
        { provider: "fish", enabled: false },
        { provider: "yandex", enabled: true },
        { provider: "piper", enabled: true },
      ],
      providers: [
        { id: "yandex", label: "Yandex", values: { voice: "alena" } },
        { id: "piper", label: "Piper", values: {} },
      ],
    };
    useAppData.mockReturnValue({ catalog: { categories: [...catalogWithVad().categories, tts] }, config });
    const { container } = await renderDashboard();
    const node = [...container.querySelectorAll(".z-svc")].pop();
    expect(node.querySelector(".prov").textContent).toBe("yandex");
    expect(node.querySelector(".mdl").textContent).toBe("voice · alena · +1 fallback");
  });

  it("surfaces rejected_24h as its own KPI tile", async () => {
    getMetrics.mockResolvedValueOnce({
      requests_24h: 5, p50_ms: 1200, p95_ms: 2000, error_rate: 0.2, rejected_24h: 3,
      per_stage_avg_ms: {},
    });
    useAppData.mockReturnValue({ catalog: catalogWithVad(), config });
    const { container } = await renderDashboard();
    // The KPI label is present and its value cell shows the reject count.
    expect(screen.getByText("Rejected · 24h")).toBeInTheDocument();
    const labels = [...container.querySelectorAll(".z-kpi .k")].map((e) => e.textContent);
    expect(labels).toContain("Rejected · 24h");
    const tile = [...container.querySelectorAll(".z-kpi")].find(
      (el) => el.querySelector(".k")?.textContent === "Rejected · 24h",
    );
    expect(tile.querySelector(".v").textContent).toBe("3");
  });
});
