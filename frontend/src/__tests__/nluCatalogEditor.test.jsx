// @vitest-environment jsdom
// Integration test (jsdom) for the offline-NLU catalog editor. Drives the REAL
// component against a mocked getTools() catalog and asserts on the chip-input DOM +
// the persistence calls it makes via onChange.
//
// Catalog under test:
//   set_light  : device_id {bright_room_light, table_light}, state {on, off}
//   set_switch : state {on, off}                              (dedups with set_light)
//   set_lock   : action {lock, unlock}
//   set_scene  : scene {night, morning}
//   set_dimmer : device_id {night_light}, brightness (integer, required, no enum)
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import NluCatalogEditor from "../components/NluCatalogEditor.jsx";
import * as api from "../api.js";

vi.mock("../api.js", () => ({ getTools: vi.fn() }));

afterEach(cleanup);

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
                device_id: { type: "string", enum: ["bright_room_light", "table_light"] },
                state: { type: "string", enum: ["on", "off"] },
              },
            },
          },
          {
            name: "set_switch",
            parameters: {
              type: "object",
              properties: { state: { type: "string", enum: ["on", "off"] } },
            },
          },
          {
            name: "set_lock",
            parameters: {
              type: "object",
              properties: { action: { type: "string", enum: ["lock", "unlock"] } },
            },
          },
          {
            name: "set_scene",
            parameters: {
              type: "object",
              properties: { scene: { type: "string", enum: ["night", "morning"] } },
            },
          },
          {
            name: "set_dimmer",
            parameters: {
              type: "object",
              required: ["device_id", "brightness"],
              properties: {
                device_id: { type: "string", enum: ["night_light"] },
                brightness: { type: "integer" },
              },
            },
          },
        ],
      },
    ],
  };
}

// Functional in-memory localStorage: the ambient global in this environment is
// Node's experimental (nonfunctional without --localstorage-file) webstorage,
// which shadows jsdom's — stub a real one so the hidden sources are testable.
function makeLocalStorage() {
  let store = {};
  return {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
    clear: () => { store = {}; },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("localStorage", makeLocalStorage());
  api.getTools.mockResolvedValue(makeSources());
});
afterEach(() => vi.unstubAllGlobals());

const draft = { aliases: "", actions: "" };

// The chip-input row: a `.e-row` whose left `<code>` text === value. Returns the
// row element so callers can query its trailing `<input>` or chip tags.
function rowForValue(value) {
  const code = screen.getByText(value);
  return code.closest(".e-row");
}
function inputForValue(value) {
  return rowForValue(value).querySelector(".e-chips input");
}
// Type a chip phrase and commit it with Enter.
function typeChip(value, phrase) {
  const input = inputForValue(value);
  fireEvent.change(input, { target: { value: phrase } });
  fireEvent.keyDown(input, { key: "Enter" });
}

describe("NluCatalogEditor (catalog port)", () => {
  it("places device ids under the Devices & scenes zone", async () => {
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    // device ids resolve to ENTITY rows under the dev zone.
    const code = await screen.findByText("bright_room_light");
    const zone = code.closest(".e-zone");
    expect(zone).toHaveClass("dev");
    expect(screen.getByText("table_light")).toBeInTheDocument();
    expect(screen.getByText("night_light")).toBeInTheDocument();
  });

  it("dedups on/off across set_light + set_switch into a single Commands group", async () => {
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    // "on" and "off" appear exactly once each despite two tools exposing the slot.
    expect(await screen.findAllByText("on")).toHaveLength(1);
    expect(screen.getAllByText("off")).toHaveLength(1);
    // They live in the Commands (cmd) zone.
    const cmdZone = screen.getByText("on").closest(".e-zone");
    expect(cmdZone).toHaveClass("cmd");
    // The merged action group advertises both tools via "used in:" <i> tags.
    const usedIn = Array.from(cmdZone.querySelectorAll(".e-used i")).map((el) => el.textContent);
    expect(usedIn).toEqual(["set_light", "set_switch"]);
  });

  it("classifies set_lock.action lock/unlock as Commands", async () => {
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    const lock = await screen.findByText("lock");
    expect(lock.closest(".e-zone")).toHaveClass("cmd");
    expect(screen.getByText("unlock").closest(".e-zone")).toHaveClass("cmd");
  });

  it("typing a device phrase chip writes aliases as '<phrase> = <id>'", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await screen.findByText("bright_room_light");

    typeChip("bright_room_light", "свет в зале");
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe("aliases");
    expect(onChange.mock.calls[0][1]).toContain("свет в зале = bright_room_light");
  });

  it("typing a verb chip for 'on' writes actions as 'on = <verb>'", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await screen.findByText("on");

    typeChip("on", "включи");
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe("actions");
    expect(onChange.mock.calls[0][1]).toContain("on = включи");
  });

  it("renders a scene value with the scene tone in the Devices zone", async () => {
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    const night = await screen.findByText("night");
    expect(night.closest(".e-zone")).toHaveClass("dev");
    // scene chips use the scene tone — the empty input row marks it as empty.
    expect(rowForValue("night").querySelector(".e-rid")).toHaveClass("is-empty");
  });

  it("shows brightness as a numeric info card, not a chip input", async () => {
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    const brightness = await screen.findByText(/brightness/);
    const numCard = brightness.closest(".e-num");
    expect(numCard).toBeTruthy();
    // It is an info card — no chip input inside it.
    expect(numCard.querySelector(".e-chips")).toBeNull();
    // brightness (integer) is a plain number, not off-capable.
    expect(numCard.querySelector(".tag").textContent).toBe("number");
  });

  it("prefills existing words as removable chips and marks the row filled", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={{ aliases: "свет в зале, люстра = bright_room_light", actions: "" }} onChange={onChange} />);
    await screen.findByText("bright_room_light");

    const row = rowForValue("bright_room_light");
    // Two chips, full marker (not empty).
    const chips = row.querySelectorAll(".e-chip");
    expect(chips).toHaveLength(2);
    expect(row.querySelector(".e-rid")).not.toHaveClass("is-empty");
    expect(row.querySelector(".e-rid .mk")).toHaveClass("full");

    // Remove the first chip -> aliases re-serialized WITHOUT "свет в зале".
    fireEvent.click(chips[0].querySelector(".x"));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe("aliases");
    expect(onChange.mock.calls[0][1]).toBe("люстра = bright_room_light");
  });

  it("removing the last chip empties the value out of the field", async () => {
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={{ aliases: "люстра = bright_room_light", actions: "" }} onChange={onChange} />);
    await screen.findByText("bright_room_light");

    const chip = rowForValue("bright_room_light").querySelector(".e-chip");
    fireEvent.click(chip.querySelector(".x"));
    // Empty words -> the value drops out of aliases entirely.
    expect(onChange.mock.calls[0][0]).toBe("aliases");
    expect(onChange.mock.calls[0][1]).not.toContain("bright_room_light");
  });

  it("flipping a group entity→action strips its values from the OLD field", async () => {
    // set_scene.scene starts as an ENTITY with a pre-typed alias. Flipping it to
    // Action via the type override must re-emit `aliases` WITHOUT the scene values.
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={{ aliases: "ночь = night", actions: "" }} onChange={onChange} />);
    await screen.findByText("night");

    // Open the scene group's TypeOverride dropdown and pick "Action".
    const sceneGroup = screen.getByText("night").closest(".e-slot");
    fireEvent.click(sceneGroup.querySelector(".e-typ-btn"));
    const actionOpt = Array.from(sceneGroup.querySelectorAll(".e-typ-opt")).find((o) =>
      o.textContent.includes("Action")
    );
    fireEvent.click(actionOpt);

    const aliasCall = onChange.mock.calls.find((c) => c[0] === "aliases");
    expect(aliasCall).toBeTruthy();
    expect(aliasCall[1]).not.toContain("night"); // stale alias stripped
  });

  it("hides the Sources bar when only one source owns slots", async () => {
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    await screen.findByText("bright_room_light");
    // Single source "home" -> no Sources bar.
    expect(screen.queryByText("Sources")).toBeNull();
  });

  it("shows the Sources bar and a per-source value count with two slot-owning sources", async () => {
    api.getTools.mockResolvedValue({
      sources: [
        {
          id: "home",
          tools: [{ name: "set_light", parameters: { type: "object", properties: { device_id: { type: "string", enum: ["lamp"] } } } }],
        },
        {
          id: "garden",
          tools: [{ name: "set_pump", parameters: { type: "object", properties: { device_id: { type: "string", enum: ["fountain"] } } } }],
        },
      ],
    });
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    await screen.findByText("lamp");
    expect(screen.getByText("Sources")).toBeInTheDocument();
    // Both sources are listed by id; both default to active so both ids render.
    expect(screen.getByText("home")).toBeInTheDocument();
    expect(screen.getByText("garden")).toBeInTheDocument();
    expect(screen.getByText("fountain")).toBeInTheDocument();
  });

  it("clicking a source chip hides its values and remembers it in this browser, not in the config", async () => {
    api.getTools.mockResolvedValue({
      sources: [
        {
          id: "home",
          tools: [{ name: "set_light", parameters: { type: "object", properties: { device_id: { type: "string", enum: ["lamp"] } } } }],
        },
        {
          id: "garden",
          tools: [{ name: "set_pump", parameters: { type: "object", properties: { device_id: { type: "string", enum: ["fountain"] } } } }],
        },
      ],
    });
    const onChange = vi.fn();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await screen.findByText("fountain");

    fireEvent.click(screen.getByText("garden").closest(".e-srv"));
    expect(screen.queryByText("fountain")).toBeNull();
    expect(onChange).not.toHaveBeenCalled();
    expect(JSON.parse(localStorage.getItem("zakhar.commands.hidden_sources"))).toEqual(["garden"]);

    // A reload keeps it hidden.
    cleanup();
    render(<NluCatalogEditor draft={draft} onChange={onChange} />);
    await screen.findByText("lamp");
    expect(screen.queryByText("fountain")).toBeNull();
  });

  it("shows every source when localStorage is unavailable", async () => {
    api.getTools.mockResolvedValue({
      sources: [
        {
          id: "home",
          tools: [{ name: "set_light", parameters: { type: "object", properties: { device_id: { type: "string", enum: ["lamp"] } } } }],
        },
        {
          id: "garden",
          tools: [{ name: "set_pump", parameters: { type: "object", properties: { device_id: { type: "string", enum: ["fountain"] } } } }],
        },
      ],
    });
    const blocked = () => { throw new Error("blocked"); };
    vi.stubGlobal("localStorage", { getItem: blocked, setItem: blocked });
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    await screen.findByText("lamp");
    expect(screen.getByText("fountain")).toBeInTheDocument();
    // Hiding still works for the session.
    fireEvent.click(screen.getByText("garden").closest(".e-srv"));
    expect(screen.queryByText("fountain")).toBeNull();
  });

  it("shows an offline card with Retry when getTools fails, then refetches", async () => {
    api.getTools.mockRejectedValueOnce(new Error("boom"));
    render(<NluCatalogEditor draft={draft} onChange={vi.fn()} />);
    const retry = await screen.findByText("Retry");
    expect(screen.getByText("No devices found")).toBeInTheDocument();

    // Retry refetches: the second call resolves with the catalog.
    fireEvent.click(retry);
    expect(await screen.findByText("bright_room_light")).toBeInTheDocument();
    expect(api.getTools).toHaveBeenCalledTimes(2);
  });
});
