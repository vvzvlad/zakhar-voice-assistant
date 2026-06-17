// @vitest-environment jsdom
// Integration test (jsdom) for the offline-NLU inline catalog editor. Pins:
//   - a `state` enum value (e.g. "on") renders as an ACTION input prefilled with
//     its verbs and typing re-serializes the `actions` field via onChange,
//   - a `device_id` id renders as an ENTITY input writing to `aliases`,
//   - a `set_lock.action` value (lock/unlock) is classified action by slot name,
//   - the per-slot kind toggle flips a slot's inputs between aliases/actions.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { NluCatalogEditor } from "../pages/stages.jsx";
import * as api from "../api.js";

// Mirror stages.vad.test.jsx: mock the api + appData modules. appData is mocked
// only to keep the module graph self-contained (NluCatalogEditor never calls it).
vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getOptions: vi.fn(async () => ({ options: [] })),
  getChimes: vi.fn(async () => ({ options: [] })),
  playChime: vi.fn(async () => ({ played: [], offline: [] })),
  getTools: vi.fn(),
}));

afterEach(cleanup);

// Small live catalog: set_light (device_id enum + state enum) and set_lock
// (device_id enum + action enum {lock,unlock}).
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
            name: "set_lock",
            parameters: {
              type: "object",
              properties: {
                device_id: { type: "string", enum: ["main_lock"] },
                action: { type: "string", enum: ["lock", "unlock"] },
              },
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

// Find the <input> sitting in the row whose <code> chip text === value.
function inputForValue(value) {
  const code = screen.getByText(value);
  const row = code.closest("div");
  return row.querySelector("input");
}

describe("NluCatalogEditor", () => {
  it("renders a state value as an ACTION input prefilled with its verbs; typing writes actions", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await waitFor(() => expect(screen.getByText("on")).toBeInTheDocument());

    // "on" is a state-slot value -> action input prefilled with "включи".
    const input = inputForValue("on");
    expect(input).toBeTruthy();
    expect(input.value).toBe("включи");

    fireEvent.change(input, { target: { value: "включи, зажги" } });
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe("actions");
    expect(onChange.mock.calls[0][1]).toContain("on = включи, зажги");
  });

  it("renders a device_id id as an ENTITY input; typing writes aliases", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await waitFor(() => expect(screen.getByText("bright_room_light")).toBeInTheDocument());

    const input = inputForValue("bright_room_light");
    expect(input).toBeTruthy();
    expect(input.value).toBe("");

    fireEvent.change(input, { target: { value: "свет в зале" } });
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe("aliases");
    expect(onChange.mock.calls[0][1]).toContain("свет в зале = bright_room_light");
  });

  it("classifies set_lock.action lock/unlock as ACTION inputs writing to actions", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await waitFor(() => expect(screen.getByText("lock")).toBeInTheDocument());

    // lock/unlock are classified action by SLOT NAME ("action"), not by verbs.
    expect(screen.getByText("unlock")).toBeInTheDocument();

    const input = inputForValue("lock");
    fireEvent.change(input, { target: { value: "запри" } });
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe("actions");
    expect(onChange.mock.calls[0][1]).toContain("lock = запри");
  });

  it("the kind toggle flips a slot's inputs between aliases and actions targets", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await waitFor(() => expect(screen.getByText("main_lock")).toBeInTheDocument());

    // set_lock.device_id (main_lock) is an ENTITY by default -> typing writes aliases.
    fireEvent.change(inputForValue("main_lock"), { target: { value: "замок" } });
    expect(onChange.mock.calls[0][0]).toBe("aliases");
    onChange.mockClear();

    // Four enum slots in encounter order -> four toggle rows, each with a
    // "Действие" button: [0]=set_light.device_id, [1]=set_light.state,
    // [2]=set_lock.device_id, [3]=set_lock.action. Flip set_lock.device_id.
    const actionButtons = screen.getAllByText("Действие");
    expect(actionButtons.length).toBe(4);
    fireEvent.click(actionButtons[2]);

    // Now main_lock is an ACTION input -> typing writes actions.
    fireEvent.change(inputForValue("main_lock"), { target: { value: "замок" } });
    expect(onChange.mock.calls[0][0]).toBe("actions");
    expect(onChange.mock.calls[0][1]).toContain("main_lock = замок");
  });

  it("toggling an entity slot to action strips its stale alias so the value is not in both fields", async () => {
    // One entity slot (set_light.device_id, enum ["lamp"]) with a pre-typed alias
    // in `aliases`. Flipping it to "Действие" must re-emit `aliases` WITHOUT "lamp".
    api.getTools.mockResolvedValue({
      sources: [
        {
          id: "home",
          tools: [
            {
              name: "set_light",
              parameters: { type: "object", properties: { device_id: { type: "string", enum: ["lamp"] } } },
            },
          ],
        },
      ],
    });
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={{ aliases: "свет = lamp", actions: "" }} onChange={onChange} />);
    await waitFor(() => expect(screen.getByText("lamp")).toBeInTheDocument());

    // Single enum slot -> single "Действие" toggle. Click it to reclassify.
    const actionButtons = screen.getAllByText("Действие");
    expect(actionButtons.length).toBe(1);
    fireEvent.click(actionButtons[0]);

    // toggleKind must emit the cleaned `aliases` field with the stale alias gone.
    const aliasCall = onChange.mock.calls.find((c) => c[0] === "aliases");
    expect(aliasCall).toBeTruthy();
    expect(aliasCall[1]).not.toContain("lamp");
  });

  it("shows a hint when no MCP tools are discovered", async () => {
    api.getTools.mockResolvedValue({ sources: [] });
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByText(/Инструменты из MCP не обнаружены/)).toBeInTheDocument()
    );
  });
});
