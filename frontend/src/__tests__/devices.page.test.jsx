// @vitest-environment jsdom
// Devices page: every write of the device list starts from the server's current
// list (a cutoff set from a slider, a device that connected while the page was
// open), and a slider write or a newly connected device refreshes the app's config snapshot.
// A click on a row opens the Edit dialog from the re-read config; it shows the id as text and
// the firmware without its build hash in its header, saves a display label, the enabled switch,
// a wake-word model (a listed or an uploaded one), the device log, the auto volume and the reply
// compression switches — edits on different tabs survive the switch, mark their tabs and go out
// in one save; only the open tab shows — marks its live controls, deletes the speaker after a
// confirmation, closes on Escape but not on a backdrop click, and shows on every tab why its
// controls cannot load. It
// also says a typed text on the speaker to check the volume — the phrase in flight and the text
// outlive a tab switch — shows the auto volume's settings and readings only while auto volume is
// on and writes the stop word's mode picked on a scale as its number.
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
  getDeviceLogs: vi.fn(async () => ({ devices: {} })),
  getDeviceLog: vi.fn(),
}));

afterEach(cleanup);

// Open the Edit dialog of the page's only speaker.
const openEdit = () => fireEvent.click(screen.getByTestId("device-row"));
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
  it("edit switches the speaker off on General and saves it onto the server's current list", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    render(<Devices />);
    openEdit();
    const sw = await switchOf("Enabled");
    expect(sw).toHaveAttribute("aria-checked", "true");
    fireEvent.click(sw);
    expect(screen.getAllByTestId("unsaved-dot").map((d) => d.parentElement.textContent)).toEqual(["General"]);
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], enabled: false },
      FRESH[1],
    ] } });
  });

  it("Delete speaker asks first, then deletes from the server's current list and closes the dialog", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Delete speaker"));
    expect(confirm).toHaveBeenLastCalledWith('Delete speaker "kitchen"?');
    expect(data.patch).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Delete speaker"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledWith({ core: { devices: [FRESH[1]] } }));
    await waitFor(() => expect(screen.queryByText("Edit speaker")).toBeNull());
    confirm.mockRestore();
  });

  it("the dialog closes on Escape, not on a backdrop click; Escape in an open dropdown closes only it", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Wake word"));
    fireEvent.click(await screen.findByText("Release default (zakhar/zakhar_v31)"));
    fireEvent.keyDown(screen.getAllByRole("option")[0], { key: "Escape" });
    expect(screen.queryByRole("option")).toBeNull();
    fireEvent.click(document.querySelector(".z-modal"));
    expect(screen.getByText("Edit speaker")).toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByText("Edit speaker")).toBeNull();
  });

  it("shows the firmware without its build hash, the whole version in its tooltip, in the table and the dialog", async () => {
    const versions = [{ id: "config_version", value: "voicepe 0.1.0 (dc1868d-dirty)" }];
    data.system = { devices: [{ name: "kitchen", online: true, versions }] };
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions, controls: [] });
    render(<Devices />);
    expect(screen.getByText("voicepe 0.1.0")).toHaveAttribute("title", "voicepe 0.1.0 (dc1868d-dirty)");
    openEdit();
    await waitFor(() => expect(screen.getAllByText("voicepe 0.1.0")).toHaveLength(2));
    expect(screen.getAllByText("voicepe 0.1.0")[1]).toHaveAttribute("title", "voicepe 0.1.0 (dc1868d-dirty)");
  });

  it("adds to the server's current list", async () => {
    render(<Devices />);
    fireEvent.click(screen.getByText("Add speaker"));
    fireEvent.change(screen.getByPlaceholderText("e.g. fc012cd1e5ec"), { target: { value: "porch" } });
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      ...FRESH, { name: "porch", enabled: true },
    ] } });
  });

  it("shows why the controls cannot load (a disabled device is a 404)", async () => {
    api.getDeviceControls.mockRejectedValue(new Error("unknown device 'kitchen'"));
    render(<Devices />);
    openEdit();
    expect(await screen.findByText("unknown device 'kitchen'")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Wake word"));
    expect(screen.getByText("unknown device 'kitchen'")).toBeVisible();
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
    openEdit();
    fireEvent.change(await screen.findByPlaceholderText("kitchen"), { target: { value: " Кухня " } });
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
    openEdit();
    fireEvent.click(await screen.findByText("Wake word"));
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
    openEdit();
    fireEvent.click(await screen.findByText("Wake word"));
    const input = await screen.findByLabelText("Wake word model file");
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
    openEdit();
    fireEvent.click(await screen.findByText("Log"));
    const sw = await switchOf("Device log");
    expect(sw).toHaveAttribute("aria-checked", "false");
    expect(screen.getByText(/keeps the last 7 days in data\/device_logs\/kitchen\//)).toBeInTheDocument();
    expect(screen.getByText("Save")).toBeDisabled();
    fireEvent.click(sw);
    expect(sw).toHaveAttribute("aria-checked", "true");
    expect(screen.getAllByTestId("unsaved-dot").map((d) => d.parentElement.textContent)).toEqual(["Log"]);
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], log: true },
      FRESH[1],
    ] } });
  });

  it("edit switches auto volume and reply compression off and saves them onto the server's current list", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    data.catalog.core.values.devices = [{ ...STALE[0], auto_volume: true, compress_replies: true }];
    render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Sound"));
    const sw = await switchOf("Auto volume");
    const compress = await switchOf("Compress replies");
    expect(sw).toHaveAttribute("aria-checked", "true");
    expect(compress).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("Save")).toBeDisabled();
    fireEvent.click(sw);
    fireEvent.click(compress);
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], auto_volume: false, compress_replies: false },
      FRESH[1],
    ] } });
  });

  it("a tab switched to while scrolled opens at its first row, the header keeps its scroll", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    render(<Devices />);
    openEdit();
    const bar = (await screen.findByText("Sound")).closest(".z-tabs");
    const body = bar.parentElement;
    // Scrolled 300 px, the header ends 212 px above the stuck bar: the bar caught on at 88.
    Object.defineProperty(body, "scrollTop", { value: 300, writable: true });
    bar.getBoundingClientRect = () => ({ top: 0 });
    bar.previousElementSibling.getBoundingClientRect = () => ({ bottom: -212 });
    fireEvent.click(screen.getByText("Wake word"));
    expect(body.scrollTop).toBe(88);
  });

  it("the Log tab reads the device's log only while it is open", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    api.getDeviceLogs.mockResolvedValue({ devices: { kitchen: ["2026-10-19"] } });
    api.getDeviceLog.mockResolvedValue({ text: "10:00:00.000 I (1) wifi: up\n", offset: 28 });
    render(<Devices />);
    openEdit();
    await screen.findByText("Wake word");
    expect(api.getDeviceLogs).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Log"));
    expect((await screen.findByTestId("device-log-text")).textContent).toBe("10:00:00.000 I (1) wifi: up");
    expect(api.getDeviceLog).toHaveBeenCalledWith("kitchen", "2026-10-19", 0);
    fireEvent.click(screen.getByText("General"));
    expect(screen.queryByTestId("device-log-text")).toBeNull();
  });

  it("edits on two tabs survive the switch, mark both tabs and go out in one save", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    render(<Devices />);
    openEdit();
    fireEvent.change(await screen.findByPlaceholderText("kitchen"), { target: { value: "Кухня" } });
    fireEvent.click(screen.getByText("Sound"));
    // Only the open tab shows.
    expect(screen.getByText("Say on speaker")).toBeVisible();
    expect(screen.getByPlaceholderText("kitchen")).not.toBeVisible();
    fireEvent.click(await switchOf("Auto volume"));
    fireEvent.click(screen.getByText("General"));
    expect(screen.getByText("Say on speaker")).not.toBeVisible();
    expect(screen.getByDisplayValue("Кухня")).toBeInTheDocument();
    expect(screen.getAllByTestId("unsaved-dot").map((d) => d.parentElement.textContent)).toEqual(["General", "Sound"]);
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch).toHaveBeenCalledWith({ core: { devices: [
      { ...FRESH[0], label: "Кухня", auto_volume: true },
      FRESH[1],
    ] } });
  });

  it("edit shows the auto volume settings and readings only while it is on, — before the first value, and a fractional gain", async () => {
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
    openEdit();
    fireEvent.click(await screen.findByText("Sound"));
    // Auto volume off: once the snapshot is in, neither its settings nor its readings show.
    await waitFor(() => expect(screen.queryAllByText("Loading…")).toHaveLength(0));
    expect(screen.queryByText("Auto volume gain")).toBeNull();
    expect(screen.queryByText("Ambient noise")).toBeNull();
    fireEvent.click(await switchOf("Auto volume"));
    expect(screen.getByText("Ambient noise").nextSibling).toHaveTextContent(/^—$/);
    expect(screen.getByText("Volume boost").nextSibling).toHaveTextContent(/^—$/);
    expect(screen.getByText("2.0 %/dB")).toBeInTheDocument();
    unmount();

    api.getDeviceControls.mockResolvedValue(snap(-47.26, 26));
    data.catalog.core.values.devices = [{ ...STALE[0], auto_volume: true }];
    render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Sound"));
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
    openEdit();
    expect(await screen.findByDisplayValue("Кухня")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Wake word"));
    expect(await screen.findByText("zakhar/zakhar_v8")).toBeInTheDocument();
  });

  it("edit shows the id as text in the header, marks a live control and a slider write refreshes the config", async () => {
    const snap = {
      device: "kitchen", online: true, versions: [],
      controls: [{ id: "wake_probability_cutoff", name: "Wake word cutoff", value: 80, min: 0, max: 100, step: 1, unit: "%" }],
    };
    api.getDeviceControls.mockResolvedValue(snap);
    api.setDeviceControl.mockResolvedValue(snap);
    render(<Devices />);
    openEdit();

    expect(await screen.findByTitle(/^Sent by the firmware when it connects/)).toHaveTextContent(/^kitchen$/);
    expect(screen.queryByDisplayValue("kitchen")).toBeNull();
    data.refreshConfig.mockClear(); // the re-read on opening the dialog

    fireEvent.click(screen.getByText("Wake word"));
    const label = await screen.findByText("Wake word cutoff");
    // The live control alone carries the mark; the fields Save sends do not.
    expect(screen.getByText("live")).toHaveAttribute("title", "Applies at once, without Save");
    expect(label.contains(screen.getByText("live"))).toBe(true);
    const track = label.closest(".z-f").querySelector(".z-trk");
    track.getBoundingClientRect = () => ({ left: 0, width: 100, top: 0, height: 10, right: 100, bottom: 10 });
    fireEvent(track, new MouseEvent("pointerdown", { bubbles: true, clientX: 55 }));
    await waitFor(() => expect(api.setDeviceControl).toHaveBeenCalledWith("kitchen", "wake_probability_cutoff", 55));
    await waitFor(() => expect(data.refreshConfig).toHaveBeenCalledTimes(1));
  });

  it("edit picks the stop word's mode on a scale and writes its number", async () => {
    const snap = {
      device: "kitchen", online: true, versions: [],
      controls: [{ id: "echo_cancel_mode", name: "Stop word echo cancellation mode (0 SR low, 1 SR high, 2 FD low, 3 FD high)",
        value: 0, min: 0, max: 3, step: 1, unit: "" }],
    };
    api.getDeviceControls.mockResolvedValue(snap);
    api.setDeviceControl.mockResolvedValue(snap);
    render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Stop word"));
    expect(await screen.findByText("Mode")).toBeInTheDocument();
    fireEvent.click(screen.getByText("FD high"));
    await waitFor(() => expect(api.setDeviceControl).toHaveBeenCalledWith("kitchen", "echo_cancel_mode", 3));
  });

  it("edit says the typed text on the speaker to check the volume", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    let finish;
    api.sayOnDevice.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Sound"));
    const input = await screen.findByDisplayValue("Проверка громкости.");
    fireEvent.change(input, { target: { value: "" } });
    expect(screen.getByText("Say")).toBeDisabled();
    fireEvent.change(input, { target: { value: "раз два" } });
    fireEvent.click(screen.getByText("Say"));
    expect(api.sayOnDevice).toHaveBeenCalledWith("kitchen", "раз два");
    expect(await screen.findByText("Saying…")).toBeDisabled();
    // A tab switch keeps the phrase in flight and the typed text.
    fireEvent.click(screen.getByText("General"));
    fireEvent.click(screen.getByText("Sound"));
    expect(screen.getByText("Saying…")).toBeDisabled();
    expect(screen.getByDisplayValue("раз два")).toBeInTheDocument();
    finish({ ok: true });
    expect(await screen.findByText("Say")).toBeEnabled();
  });

  it("edit shows why the speaker could not say the text, and an offline speaker cannot", async () => {
    api.getDeviceControls.mockResolvedValue({ device: "kitchen", online: true, versions: [], controls: [] });
    api.sayOnDevice.mockRejectedValue(new Error("tts down"));
    const { unmount } = render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Sound"));
    await screen.findByDisplayValue("Проверка громкости.");
    fireEvent.click(screen.getByText("Say"));
    expect(await screen.findByText("tts down")).toBeInTheDocument();
    unmount();

    data.system = { devices: [{ name: "kitchen", online: false, versions: [] }] };
    render(<Devices />);
    openEdit();
    fireEvent.click(await screen.findByText("Sound"));
    await screen.findByDisplayValue("Проверка громкости.");
    expect(screen.getByText("Say")).toBeDisabled();
    expect(screen.getByText("Say")).toHaveAttribute("title", "Speaker offline");
  });
});
