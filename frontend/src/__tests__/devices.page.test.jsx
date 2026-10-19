// @vitest-environment jsdom
// Devices page: every write of the device list starts from the server's current
// list (a cutoff set from a slider, a device that connected while the page was
// open), the Edit dialog opens from the re-read config, shows the id read-only, saves a
// display label, a wake-word model (a listed or an uploaded one), the device log and the auto volume switches, and shows why its controls cannot load, and a slider write or a
// newly connected device refreshes the app's config snapshot. The dialog also says a
// typed text on the speaker to check the volume and shows the auto volume's readings.
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
  sayOnDevice: vi.fn(),
  getWakeProbability: vi.fn(async () => ({ online: true, value: null })),
  setWakeProbabilityStream: vi.fn(async () => ({})),
  getWakeModels: vi.fn(async () => ({
    options: ["zakhar/zakhar_v8", "zakhar/zakhar_v31"],
    release: "zakhar/zakhar_v31",
  })),
  uploadWakeModel: vi.fn(),
}));

afterEach(cleanup);

// The switch of the dialog field labelled `label`.
const switchOf = async (label) => (await screen.findByText(label)).closest(".z-f").querySelector('[role="switch"]');

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

  it("edit uploads a .tflite, selects it and saves its name", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    api.uploadWakeModel.mockResolvedValueOnce({ name: "uploaded/zakhar_v34" });
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    const input = await screen.findByLabelText("Wake-word model file");
    api.getWakeModels.mockResolvedValueOnce({
      options: ["uploaded/zakhar_v34", "zakhar/zakhar_v8", "zakhar/zakhar_v31"], release: "zakhar/zakhar_v31",
    });
    const file = new File([new Uint8Array([1, 2])], "zakhar_v34.tflite");
    fireEvent.change(input, { target: { files: [file] } });
    expect(await screen.findByText("uploaded/zakhar_v34")).toBeInTheDocument();
    expect(api.uploadWakeModel).toHaveBeenCalledWith(file);
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], wake_model: "uploaded/zakhar_v34" },
      FRESH[1],
    ] } });
  });

  it("edit switches the device log on and saves it onto the server's current list", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    data.catalog.core.values.device_log_days = 7;
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    const sw = await switchOf("Device log");
    expect(sw).toHaveAttribute("aria-checked", "false");
    expect(screen.getByText(/keeps the last 7 days in data\/device_logs\/kitchen\//)).toBeInTheDocument();
    expect(screen.getByText("Save")).toBeDisabled();
    fireEvent.click(sw);
    expect(sw).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], log: true },
      FRESH[1],
    ] } });
  });

  it("edit switches auto volume off and saves it onto the server's current list", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    data.catalog.core.values.devices = [{ ...STALE[0], auto_volume: true }];
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    const sw = await switchOf("Auto volume");
    expect(sw).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("Save")).toBeDisabled();
    fireEvent.click(sw);
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], auto_volume: false },
      FRESH[1],
    ] } });
  });

  it("edit shows the auto volume readings, — before the first value, and a fractional gain", async () => {
    const snap = (noise, boost) => ({
      device: "kitchen", online: true, versions: [],
      controls: [{ id: "auto_volume_gain", name: "Auto volume gain", value: 2, min: 0, max: 10, step: 0.1, unit: "%/dB" }],
      readings: [
        { id: "ambient_noise", name: "Ambient noise", value: noise, unit: "dBFS" },
        { id: "volume_boost", name: "Volume boost", value: boost, unit: "%" },
      ],
    });
    api.getDeviceControls.mockResolvedValue(snap(null, null));
    const { unmount } = render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    expect((await screen.findByText("Ambient noise")).nextSibling).toHaveTextContent(/^—$/);
    expect(screen.getByText("Volume boost").nextSibling).toHaveTextContent(/^—$/);
    expect(screen.getByText("2.0 %/dB")).toBeInTheDocument();
    unmount();

    api.getDeviceControls.mockResolvedValue(snap(-47.26, 26));
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    expect(await screen.findByText("-47.3 dBFS")).toBeInTheDocument();
    expect(screen.getByText("26%")).toBeInTheDocument();
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

  it("edit says the typed text on the speaker to check the volume", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    let finish;
    api.sayOnDevice.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    const input = await screen.findByDisplayValue("Проверка громкости");
    fireEvent.change(input, { target: { value: "" } });
    expect(screen.getByText("Say")).toBeDisabled();
    fireEvent.change(input, { target: { value: "раз два" } });
    fireEvent.click(screen.getByText("Say"));
    expect(api.sayOnDevice).toHaveBeenCalledWith("kitchen", "раз два");
    expect(await screen.findByText("Saying…")).toBeDisabled();
    finish({ ok: true });
    expect(await screen.findByText("Say")).toBeEnabled();
  });

  it("edit shows why the speaker could not say the text, and an offline speaker cannot", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    api.sayOnDevice.mockRejectedValue(new Error("tts down"));
    const { unmount } = render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    await screen.findByDisplayValue("Проверка громкости");
    fireEvent.click(screen.getByText("Say"));
    expect(await screen.findByText("tts down")).toBeInTheDocument();
    unmount();

    data.system = { devices: [{ name: "kitchen", online: false, versions: [] }] };
    render(<Devices />);
    fireEvent.click(screen.getByText("Edit"));
    await screen.findByDisplayValue("Проверка громкости");
    expect(screen.getByText("Say")).toBeDisabled();
    expect(screen.getByText("Say")).toHaveAttribute("title", "Speaker offline");
  });
});
