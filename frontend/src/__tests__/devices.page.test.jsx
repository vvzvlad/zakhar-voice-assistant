// @vitest-environment jsdom
// Devices page: every write of the device list starts from the server's current
// list (a cutoff set from a slider, a device that connected while the page was
// open), the Edit dialog opens from the re-read config, shows the id read-only, saves a
// display label and a wake-word model, and shows why its controls cannot load, and a slider write or a
// newly connected device refreshes the app's config snapshot.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { Devices } from "../pages/integrations.jsx";
import { useAppData } from "../appData.jsx";
import * as api from "../api.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getConfig: vi.fn(),
  getDeviceControls: vi.fn(),
  setDeviceControl: vi.fn(),
  getWakeProbability: vi.fn(async () => ({ online: true, value: null })),
  setWakeProbabilityStream: vi.fn(async () => ({})),
  getWakeModels: vi.fn(async () => ({
    options: ["zakhar/zakhar_v8", "zakhar/zakhar_v31"],
    release: "zakhar/zakhar_v31",
  })),
}));

afterEach(cleanup);

// What the page rendered from: kitchen at the old cutoff, no hall yet.
const STALE = [{ name: "kitchen", enabled: true, wake_cutoff: 80, vad_cutoff: 30 }];
// What the server holds by the time of the write.
const FRESH = [
  { name: "kitchen", enabled: true, wake_cutoff: 55, vad_cutoff: 30 },
  { name: "hall", enabled: true, wake_cutoff: 80, vad_cutoff: 30 },
];

let data;
beforeEach(() => {
  vi.clearAllMocks();
  api.getConfig.mockResolvedValue({ core: { devices: FRESH } });
  data = {
    catalog: { core: { values: { devices: STALE, audio: { port: 8200 } } } },
    system: { devices: [{ name: "kitchen", online: true, versions: [] }] },
    patch: vi.fn(async () => {}),
    refreshConfig: vi.fn(async () => {}),
  };
  useAppData.mockReturnValue(data);
});

describe("Devices page", () => {
  it("toggles enabled on the server's current list", async () => {
    render(<Devices />);
    fireEvent.click(screen.getByText("Disable"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], enabled: false },
      FRESH[1],
    ] } });
  });

  it("adds to the server's current list", async () => {
    render(<Devices />);
    fireEvent.click(screen.getByText("Add speaker"));
    fireEvent.change(screen.getByPlaceholderText("e.g. fc012cd1e5ec"), { target: { value: "porch" } });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      ...FRESH, { name: "porch", enabled: true },
    ] } });
  });

  it("shows why the controls cannot load (a disabled device is a 404)", async () => {
    api.getDeviceControls.mockRejectedValue(new Error("unknown device 'kitchen'"));
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    expect(await screen.findByText("unknown device 'kitchen'")).toBeInTheDocument();
    expect(screen.queryByText("Loading…")).toBeNull();
  });

  it("re-reads the config when the heartbeat lists a device the page does not know", async () => {
    data.system = { devices: [
      { name: "kitchen", online: true, versions: [] },
      { name: "hall", online: true, versions: [] },
    ] };
    render(<Devices />);
    await waitFor(() => expect(data.refreshConfig).toHaveBeenCalledTimes(1));
  });

  it("edit saves a label onto the server's current list and the table shows it", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    // A model set elsewhere (an agent) since the page loaded survives a label edit.
    const fresh = [{ ...FRESH[0], wake_model: "zakhar/zakhar_v32" }, FRESH[1]];
    api.getConfig.mockResolvedValue({ core: { devices: fresh } });
    const { rerender } = render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    expect(await screen.findByDisplayValue("kitchen")).toHaveAttribute("readonly");
    fireEvent.change(screen.getByPlaceholderText("kitchen"), { target: { value: " Кухня " } });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...fresh[0], label: "Кухня" },
      FRESH[1],
    ] } });

    data.catalog.core.values.devices = [{ ...STALE[0], label: "Кухня" }];
    rerender(<Devices />);
    expect(screen.getByText("Кухня")).toBeInTheDocument();
  });

  it("edit picks a wake-word model from the release's and saves its name", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    data.catalog.core.values.devices = [{ ...STALE[0], wake_model: "zakhar/zakhar_v31" }];
    // A label set elsewhere since the page loaded survives a model edit.
    const fresh = [{ ...FRESH[0], label: "Кухня" }, FRESH[1]];
    api.getConfig.mockResolvedValue({ core: { devices: fresh } });
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    fireEvent.click(await screen.findByText("zakhar/zakhar_v31"));
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Release default (zakhar/zakhar_v31)", "zakhar/zakhar_v8", "zakhar/zakhar_v31",
    ]);
    fireEvent.click(screen.getByRole("option", { name: "zakhar/zakhar_v8" }));
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...fresh[0], wake_model: "zakhar/zakhar_v8" },
      FRESH[1],
    ] } });
  });

  it("edit opens with the label and model saved since the page loaded", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    data.refreshConfig.mockImplementation(async () => {
      await Promise.resolve(); // the network read: the catalog changes a tick later
      data.catalog.core.values.devices = [{ ...STALE[0], label: "Кухня", wake_model: "zakhar/zakhar_v8" }];
    });
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    expect(await screen.findByDisplayValue("Кухня")).toBeInTheDocument();
    expect(await screen.findByText("zakhar/zakhar_v8")).toBeInTheDocument();
  });

  it("edit shows the id read-only and a slider write refreshes the config", async () => {
    const snap = {
      device: "kitchen", online: true, versions: [],
      controls: [{ id: "wake_probability_cutoff", name: "Wake word cutoff", value: 80, min: 0, max: 100, step: 1, unit: "%" }],
    };
    api.getDeviceControls.mockResolvedValue(snap);
    api.setDeviceControl.mockResolvedValue(snap);
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));

    expect(await screen.findByDisplayValue("kitchen")).toHaveAttribute("readonly");
    data.refreshConfig.mockClear(); // the re-read on opening the dialog

    await screen.findByText("Wake word cutoff");
    const track = document.querySelector(".z-trk");
    track.getBoundingClientRect = () => ({ left: 0, width: 100, top: 0, height: 10, right: 100, bottom: 10 });
    fireEvent(track, new MouseEvent("pointerdown", { bubbles: true, clientX: 55 }));
    await waitFor(() => expect(api.setDeviceControl).toHaveBeenCalledWith("kitchen", "wake_probability_cutoff", 55));
    await waitFor(() => expect(data.refreshConfig).toHaveBeenCalledTimes(1));
  });
});
