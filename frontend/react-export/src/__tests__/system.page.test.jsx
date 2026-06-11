// @vitest-environment jsdom
// Render-level tests for the System page's Agent MCP card: the card renders off
// the AgentMcpConfig schema and the read-only endpoint hint is computed from the
// saved values (wildcard/empty bind host falls back to window.location.hostname).
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import SystemPage from "../pages/system.jsx";
import { useAppData } from "../appData.jsx";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

// Minimal pydantic-shaped schema for the core.agent_mcp section.
const agentMcpSchema = {
  title: "AgentMcpConfig",
  type: "object",
  properties: {
    enabled: { type: "boolean", title: "Enabled", default: true },
    host: { type: "string", title: "Host", default: "0.0.0.0" },
    port: { type: "integer", title: "Port", default: 8202 },
  },
};

// Empty-properties stubs for the sibling cards so their schema-less fallback
// markup (which also carries a "Port" label) does not collide with the
// Agent MCP card's field labels in getByText.
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

describe("System page — Agent MCP card", () => {
  it("renders the card with its schema fields and the endpoint hint", () => {
    useAppData.mockReturnValue(appData({ enabled: true, host: "0.0.0.0", port: 8202 }));
    render(<SystemPage />);
    expect(screen.getByText("Agent MCP")).toBeInTheDocument();
    expect(screen.getByText("lets external agents drive the assistant")).toBeInTheDocument();
    // Schema-driven fields are rendered.
    expect(screen.getByText("Enabled")).toBeInTheDocument();
    expect(screen.getByText("Host")).toBeInTheDocument();
    expect(screen.getByText("Port")).toBeInTheDocument();
    // A 0.0.0.0 bind host displays as the page's own hostname (jsdom: localhost).
    expect(screen.getByText("Endpoint")).toBeInTheDocument();
    expect(screen.getByText(`http://${window.location.hostname}:8202/mcp`)).toBeInTheDocument();
  });

  it("computes the endpoint from an explicit host/port", () => {
    useAppData.mockReturnValue(appData({ enabled: true, host: "10.0.0.5", port: 9300 }));
    render(<SystemPage />);
    expect(screen.getByText("http://10.0.0.5:9300/mcp")).toBeInTheDocument();
  });

  it("falls back to the default port when values omit it", () => {
    useAppData.mockReturnValue(appData({}));
    render(<SystemPage />);
    expect(screen.getByText(`http://${window.location.hostname}:8202/mcp`)).toBeInTheDocument();
  });
});
