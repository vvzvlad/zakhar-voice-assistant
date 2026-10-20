// @vitest-environment jsdom
// A speaker's log view: opens on the device's newest day, shows its text without the ANSI colour
// escapes, appends what the next poll brings without losing a selection, polls only the newest
// day, drops a read that lands after the choice changed, and stops on leaving.
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, act, fireEvent } from "@testing-library/react";
import DeviceLog from "../components/DeviceLog.jsx";
import * as api from "../api.js";

vi.mock("../api.js", () => ({ getDeviceLogs: vi.fn(), getDeviceLog: vi.fn() }));

afterEach(() => { cleanup(); vi.useRealTimers(); vi.resetAllMocks(); window.getSelection().removeAllRanges(); });

const advance = (ms) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
// The shown log, one entry per line block.
const shown = () => [...screen.getByTestId("device-log-text").querySelectorAll(".z-logline")].map((d) => d.textContent);

describe("Device log view", () => {
  it("shows the device's newest day and appends what the poll brings", async () => {
    vi.useFakeTimers();
    api.getDeviceLogs.mockResolvedValue({
      devices: { hall: ["2026-10-01"], kitchen: ["2026-10-19", "2026-10-18"] },
    });
    api.getDeviceLog
      .mockResolvedValueOnce({ text: "10:00:00.000 \x1b[0;32mI (1) wifi: up\x1b[0m\n", offset: 40 })
      .mockResolvedValueOnce({ text: "10:00:02.000 I (2) link: ok\n", offset: 68 });
    render(<DeviceLog device="kitchen" />);
    await advance(0);
    expect(screen.getByRole("button", { name: "2026-10-19" })).toBeInTheDocument();
    expect(api.getDeviceLog).toHaveBeenLastCalledWith("kitchen", "2026-10-19", 0);
    expect(shown()).toEqual(["10:00:00.000 I (1) wifi: up"]);

    await advance(2000);
    expect(api.getDeviceLog).toHaveBeenLastCalledWith("kitchen", "2026-10-19", 40);
    expect(shown()).toEqual(["10:00:00.000 I (1) wifi: up", "10:00:02.000 I (2) link: ok"]);
  });

  it("keeps a selection in the shown text when the poll appends", async () => {
    vi.useFakeTimers();
    api.getDeviceLogs.mockResolvedValue({ devices: { kitchen: ["2026-10-19"] } });
    api.getDeviceLog
      .mockResolvedValueOnce({ text: "10:00:00.000 I (1) wifi: up\n", offset: 28 })
      .mockResolvedValueOnce({ text: "10:00:02.000 I (2) link: ok\n", offset: 56 });
    render(<DeviceLog device="kitchen" />);
    await advance(0);
    const node = screen.getByTestId("device-log-text").querySelector(".z-logline").firstChild;
    const range = document.createRange();
    range.setStart(node, 19);
    range.setEnd(node, 23);
    window.getSelection().addRange(range);
    expect(window.getSelection().toString()).toBe("wifi");

    await advance(2000);
    expect(shown()).toContain("10:00:02.000 I (2) link: ok");
    expect(window.getSelection().toString()).toBe("wifi");
  });

  it("drops a read that lands after the day changed, polls only the newest day, stops on leaving", async () => {
    vi.useFakeTimers();
    api.getDeviceLogs.mockResolvedValue({ devices: { kitchen: ["2026-10-19", "2026-10-18"] } });
    let late;
    api.getDeviceLog
      .mockReturnValueOnce(new Promise((resolve) => { late = resolve; }))
      .mockResolvedValue({ text: "09:00:00.000 I (1) x: older day\n", offset: 32 });
    const { unmount } = render(<DeviceLog device="kitchen" />);
    await advance(0);
    fireEvent.click(screen.getByRole("button", { name: "2026-10-19" }));
    fireEvent.click(screen.getByRole("option", { name: "2026-10-18" }));
    await advance(0);
    expect(api.getDeviceLog).toHaveBeenLastCalledWith("kitchen", "2026-10-18", 0);

    late({ text: "10:00:00.000 I (1) x: today\n", offset: 28 });
    await advance(4000);
    expect(shown()).toEqual(["09:00:00.000 I (1) x: older day"]);
    expect(api.getDeviceLog).toHaveBeenCalledTimes(2);  // the older day is read once, not polled

    fireEvent.click(screen.getByRole("button", { name: "2026-10-18" }));
    fireEvent.click(screen.getByRole("option", { name: "2026-10-19" }));
    await advance(2000);
    const calls = api.getDeviceLog.mock.calls.length;
    unmount();
    await advance(4000);
    expect(api.getDeviceLog).toHaveBeenCalledTimes(calls);
  });

  it("says there is no log yet for a device the server keeps none for", async () => {
    api.getDeviceLogs.mockResolvedValue({ devices: { hall: ["2026-10-01"] } });
    render(<DeviceLog device="kitchen" />);
    expect(await screen.findByText("No log yet")).toBeInTheDocument();
    expect(api.getDeviceLog).not.toHaveBeenCalled();
  });
});
