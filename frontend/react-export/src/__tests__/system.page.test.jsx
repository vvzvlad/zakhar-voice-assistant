// @vitest-environment jsdom
// Render-level tests for the System page's "MCP Server for other agents" card:
// the card renders the enabled toggle off the AgentMcpConfig schema, shows a
// short static capability summary, and the read-only endpoint hint is the
// panel's own origin plus /mcp (the endpoint is served by the panel itself, so
// it is always same-origin — no host/port math).
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import SystemPage from "../pages/system.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

// Minimal pydantic-shaped schema for the core.agent_mcp section (enabled only:
// the endpoint binds with the panel, so there is no host/port anymore).
const agentMcpSchema = {
  title: "AgentMcpConfig",
  type: "object",
  properties: {
    enabled: { type: "boolean", title: "Enabled", default: true },
  },
};

// Empty-properties stubs for the sibling cards so their schema-less fallback
// markup does not collide with the MCP card's field labels in getByText.
const emptySchema = { type: "object", properties: {} };

function appData(agentMcpValues) {
  return {
    patch: vi.fn(async () => {}),
    connected: true,
    system: { version: "1.0.0", uptime_seconds: 10, db_size_bytes: 2048, log_level: "INFO" },
    catalog: {
      core: {
        schema: {
          $defs: {
            AgentMcpConfig: agentMcpSchema,
            NetworkConfig: emptySchema,
            AudioConfig: emptySchema,
            RunsConfig: emptySchema,
          },
        },
        values: { agent_mcp: agentMcpValues, log_level: "INFO" },
      },
    },
  };
}

describe("System page — MCP Server for other agents card", () => {
  it("renders the card with the toggle, capability summary and endpoint hint", () => {
    useAppData.mockReturnValue(appData({ enabled: true }));
    render(<SystemPage />);
    expect(screen.getByText("MCP Server for other agents")).toBeInTheDocument();
    expect(screen.getByText("streamable-HTTP MCP endpoint on this panel's port")).toBeInTheDocument();
    // Schema-driven enabled toggle is rendered.
    expect(screen.getByText("Enabled")).toBeInTheDocument();
    // Short static capability summary for a connected agent.
    expect(screen.getByText(/A connected agent can:/)).toBeInTheDocument();
    expect(screen.getByText(/read the request\/reply log/)).toBeInTheDocument();
    // The endpoint is same-origin: the page's own origin + /mcp.
    expect(screen.getByText("Endpoint")).toBeInTheDocument();
    expect(screen.getByText(`${window.location.origin}/mcp`)).toBeInTheDocument();
  });

  it("shows the same same-origin endpoint regardless of saved values", () => {
    // Stale docs may still carry host/port keys; the hint must ignore them.
    useAppData.mockReturnValue(appData({ enabled: false, host: "10.0.0.5", port: 9300 }));
    render(<SystemPage />);
    expect(screen.getByText(`${window.location.origin}/mcp`)).toBeInTheDocument();
    expect(screen.queryByText(/10\.0\.0\.5/)).not.toBeInTheDocument();
  });

  it("renders the endpoint hint even when values are empty", () => {
    useAppData.mockReturnValue(appData({}));
    render(<SystemPage />);
    expect(screen.getByText(`${window.location.origin}/mcp`)).toBeInTheDocument();
  });
});
