// @vitest-environment jsdom
// Integration test (jsdom) for the offline-NLU inline alias editor. Pins:
//   - a state slot ({on,off} when on/off are action verbs) is NOT shown as a row,
//   - device_id / scene enum ids ARE shown as alias inputs,
//   - typing in an input re-serializes the whole `aliases` field via onChange.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { NluAliasEditor } from "../pages/stages.jsx";
import * as api from "../api.js";

// Mirror stages.vad.test.jsx: mock the api + appData modules. appData is mocked
// only to keep the module graph self-contained (NluAliasEditor never calls it).
vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getOptions: vi.fn(async () => ({ options: [] })),
  getChimes: vi.fn(async () => ({ options: [] })),
  playChime: vi.fn(async () => ({ played: [], offline: [] })),
  getTools: vi.fn(),
}));

afterEach(cleanup);

// Small live catalog: set_light (device_id enum + state enum) and set_scene
// (scene enum). state {on,off} should be filtered out by the action verbs.
function makeSources() {
  return {
    sources: [
      {
        id: "home",
        tools: [
          {
            name: "set_light",
            parameters: {
              type: "object",
              properties: {
                device_id: { type: "string", enum: ["bright_room_light", "night_light"] },
                state: { type: "string", enum: ["on", "off"] },
              },
            },
          },
          {
            name: "set_scene",
            parameters: {
              type: "object",
              properties: { scene: { type: "string", enum: ["night", "morning"] } },
            },
          },
        ],
      },
    ],
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  api.getTools.mockResolvedValue(makeSources());
});

const draft = { aliases: "", actions: "on = включи\noff = выключи" };

describe("NluAliasEditor", () => {
  it("hides the {on,off} state slot and shows device_id / scene ids as rows", async () => {
    const onChange = vi.fn();
    render(<NluAliasEditor draft={draft} onChange={onChange} />);
    // Wait for the async getTools load to settle.
    await waitFor(() => expect(screen.getByText("bright_room_light")).toBeInTheDocument());

    // Entity ids ARE shown.
    expect(screen.getByText("bright_room_light")).toBeInTheDocument();
    expect(screen.getByText("night_light")).toBeInTheDocument();
    expect(screen.getByText("night")).toBeInTheDocument();
    expect(screen.getByText("morning")).toBeInTheDocument();
    // The state slot values (on/off) must NOT appear as rows.
    expect(screen.queryByText("on")).toBeNull();
    expect(screen.queryByText("off")).toBeNull();
  });

  it("typing in an input re-serializes aliases via onChange", async () => {
    const onChange = vi.fn();
    render(<NluAliasEditor draft={draft} onChange={onChange} />);
    await waitFor(() => expect(screen.getByText("bright_room_light")).toBeInTheDocument());

    // The row for bright_room_light: its <code> sits next to the alias input.
    const code = screen.getByText("bright_room_light");
    const row = code.closest("div");
    const input = row.querySelector("input");
    expect(input).toBeTruthy();

    fireEvent.change(input, { target: { value: "свет в зале" } });

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe("aliases");
    const text = onChange.mock.calls[0][1];
    expect(text).toContain("свет в зале = bright_room_light");
  });

  it("shows a hint when no MCP tools are discovered", async () => {
    api.getTools.mockResolvedValue({ sources: [] });
    render(<NluAliasEditor draft={draft} onChange={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByText(/Инструменты из MCP не обнаружены/)).toBeInTheDocument()
    );
  });
});
