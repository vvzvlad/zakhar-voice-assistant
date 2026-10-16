// @vitest-environment jsdom
// Render-level tests for the Dashboard: the Capture → Logic → Speech service map
// (what each tile shows, grey dots for what is off, where a click leads, an old
// backend without categories) and the KPI tiles.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, within } from "@testing-library/react";
import Dashboard from "../pages/dashboard.jsx";
import { useAppData } from "../appData.jsx";
import { getMetrics } from "../api.js";
import { nav } from "../navStore.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../navStore.js", () => ({ nav: vi.fn() }));
vi.mock("../api.js", () => ({
  getMetrics: vi.fn(async () => ({})),
  getRuns: vi.fn(async () => ({ runs: [] })),
  openRunsStream: vi.fn(() => () => {}),
}));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

const chains = [
  { name: "fast", enabled: true, stt: "vosk", handler: "simple-nlu" },
  { name: "off", enabled: false, stt: "groq", handler: "openrouter" },
  { name: "smart", enabled: true, stt: "groq", handler: "openrouter" },
];
const config = { core: { chains, mcp_servers: [{ name: "home" }] } };

const single = (id, selected, label, values) => ({ id, selected, providers: [{ id: selected, label, values }] });

function catalog({ wake = true, stress = true } = {}) {
  return {
    categories: [
      single("vad", "webrtc", "WebRTC VAD", {}),
      single("wakeword", "vosk", "Vosk", { enabled: wake }),
      single("stress", "ruaccent", "RuAccent", { enabled: stress }),
      {
        id: "tts",
        variants: [
          { provider: "yandex", enabled: false },
          { provider: "fishaudio", enabled: true },
          { provider: "piper", enabled: true },
        ],
        providers: [
          { id: "yandex", label: "Yandex", values: { voice: "alena" } },
          { id: "fishaudio", label: "Fish Audio", values: { reference_id: "9b64e", reference_id_label: "Anna" } },
          { id: "piper", label: "Piper", values: {} },
        ],
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

const tile = (page) => screen.getByTestId(`tile-${page}`);
const dots = (el) => [...el.querySelectorAll(".z-dot")].map((d) => d.className.replace("z-dot ", ""));

describe("Dashboard service map", () => {
  it("shows Capture, Logic and Speech with their providers, switches and averages", async () => {
    getMetrics.mockResolvedValueOnce({
      per_stage_avg_ms: { vad: 1200, tts: 600 },
      per_chain: { fast: { served: 3, avg_ms: 400 }, smart: { served: 1, avg_ms: 9000 } },
    });
    useAppData.mockReturnValue({ catalog: catalog(), config });
    await renderDashboard();

    const capture = tile("capture");
    expect(capture.querySelector(".prov").textContent).toBe("WebRTC VAD");
    expect(capture.textContent).toContain("wake-word verify on");
    expect(capture.textContent).toContain("1.20s avg");

    // Logic rows name the chain, not its stt → handler pair.
    expect(screen.getAllByTestId("chain-row").map((r) => r.textContent)).toEqual(["fast75%0.40s", "smart25%9.00s"]);
    expect(within(tile("logic")).getByText("2 enabled")).toBeInTheDocument();

    const speech = tile("speech");
    expect(speech.querySelector(".prov").textContent).toBe("Fish Audio");
    expect(within(speech).getByTestId("speech-voice").textContent).toBe("Anna · +1 fallback");
    expect(speech.textContent).toContain("stress on");
    expect(speech.textContent).toContain("0.60s avg");

    expect([capture, tile("logic"), speech].map(dots)).toEqual([["ok", "ok"], ["ok"], ["ok", "ok"]]);
  });

  it("greys the dot of whatever is off", async () => {
    useAppData.mockReturnValue({
      catalog: catalog({ wake: false, stress: false }),
      config: { core: { chains: chains.map((c) => ({ ...c, enabled: false })) } },
    });
    await renderDashboard();
    expect(tile("capture").textContent).toContain("wake-word verify off");
    expect(tile("speech").textContent).toContain("stress off");
    expect([tile("capture"), tile("logic"), tile("speech")].map(dots)).toEqual([["ok", "off"], ["off"], ["ok", "off"]]);
    expect(within(tile("logic")).getByText("no enabled chain")).toBeInTheDocument();
  });

  it("opens the tile's pipeline page on click, and Tools from the MCP chip", async () => {
    useAppData.mockReturnValue({ catalog: catalog(), config });
    await renderDashboard();
    for (const page of ["capture", "logic", "speech"]) fireEvent.click(tile(page));
    fireEvent.click(screen.getByText("◆ MCP"));
    expect(nav.mock.calls.map((c) => c[0])).toEqual(["capture", "logic", "speech", "tools"]);
  });

  it('shows "—" and still renders the page when the catalog has no categories (old backend)', async () => {
    useAppData.mockReturnValue({ catalog: { categories: [] }, config: { core: {} } });
    await renderDashboard();
    expect(screen.getByText("Pipeline overview")).toBeInTheDocument();
    expect(tile("capture").querySelector(".prov").textContent).toBe("—");
    expect(tile("speech").querySelector(".prov").textContent).toBe("—");
  });

  it("surfaces rejected_24h as its own KPI tile", async () => {
    getMetrics.mockResolvedValueOnce({
      requests_24h: 5, p50_ms: 1200, p95_ms: 2000, error_rate: 0.2, rejected_24h: 3,
      per_stage_avg_ms: {},
    });
    useAppData.mockReturnValue({ catalog: catalog(), config });
    const { container } = await renderDashboard();
    const tileEl = [...container.querySelectorAll(".z-kpi")].find(
      (el) => el.querySelector(".k")?.textContent === "Rejected · 24h",
    );
    expect(tileEl.querySelector(".v").textContent).toBe("3");
  });
});
