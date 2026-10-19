// @vitest-environment jsdom
// Device Log page: opens on the first device's newest day, shows its text without the ANSI
// colour escapes, appends what the next poll brings without losing a selection, polls only the
// newest day, drops a read that lands after the choice changed, and stops on leaving.
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, act, fireEvent } from "@testing-library/react";
import DeviceLog from "../pages/devlog.jsx";
import * as api from "../api.js";

vi.mock("../appData.jsx", () => ({ useDeviceList: () => [{ name: "kitchen", label: "Kitchen" }] }));
vi.mock("../api.js", () => ({ getDeviceLogs: vi.fn(), getDeviceLog: vi.fn() }));

afterEach(() => { cleanup(); vi.useRealTimers(); vi.resetAllMocks(); window.getSelection().removeAllRanges(); });

const advance = (ms) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

describe("Device Log page", () => {
  it("shows the newest day of the first device and appends what the poll brings", async () => {
    vi.useFakeTimers();
    api.getDeviceLogs.mockResolvedValue({
      devices: { kitchen: ["2026-10-19", "2026-10-18"], hall: ["2026-10-01"] },
    });
    api.getDeviceLog
      .mockResolvedValueOnce({ text: "10:00:00.000 \x1b[0;32mI (1) wifi: up\x1b[0m\n", offset: 40 })
      .mockResolvedValueOnce({ text: "10:00:02.000 I (2) link: ok\n", offset: 68 });
    render(<DeviceLog />);
    await advance(0);
    expect(screen.getByRole("button", { name: "Kitchen" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "2026-10-19" })).toBeInTheDocument();
    expect(api.getDeviceLog).toHaveBeenLastCalledWith("kitchen", "2026-10-19", 0);
    expect(screen.getByTestId("device-log-text").textContent).toBe("10:00:00.000 I (1) wifi: up\n");

    await advance(2000);
    expect(api.getDeviceLog).toHaveBeenLastCalledWith("kitchen", "2026-10-19", 40);
    expect(screen.getByTestId("device-log-text").textContent)
      .toBe("10:00:00.000 I (1) wifi: up\n10:00:02.000 I (2) link: ok\n");
  });

  it("keeps a selection in the shown text when the poll appends", async () => {
    vi.useFakeTimers();
    api.getDeviceLogs.mockResolvedValue({ devices: { kitchen: ["2026-10-19"] } });
    api.getDeviceLog
      .mockResolvedValueOnce({ text: "10:00:00.000 I (1) wifi: up\n", offset: 28 })
      .mockResolvedValueOnce({ text: "10:00:02.000 I (2) link: ok\n", offset: 56 });
    render(<DeviceLog />);
    await advance(0);
    const node = screen.getByTestId("device-log-text").firstChild;
    const range = document.createRange();
    range.setStart(node, 19);
    range.setEnd(node, 23);
    window.getSelection().addRange(range);
    expect(window.getSelection().toString()).toBe("wifi");

    await advance(2000);
    expect(screen.getByTestId("device-log-text").textContent).toContain("link: ok");
    expect(window.getSelection().toString()).toBe("wifi");
  });

  it("drops a read that lands after the day changed, polls only the newest day, stops on leaving", async () => {
    vi.useFakeTimers();
    api.getDeviceLogs.mockResolvedValue({ devices: { kitchen: ["2026-10-19", "2026-10-18"] } });
    let late;
    api.getDeviceLog
      .mockReturnValueOnce(new Promise((resolve) => { late = resolve; }))
      .mockResolvedValue({ text: "09:00:00.000 I (1) x: older day\n", offset: 32 });
    const { unmount } = render(<DeviceLog />);
    await advance(0);
    fireEvent.click(screen.getByRole("button", { name: "2026-10-19" }));
    fireEvent.click(screen.getByRole("option", { name: "2026-10-18" }));
    await advance(0);
    expect(api.getDeviceLog).toHaveBeenLastCalledWith("kitchen", "2026-10-18", 0);

    late({ text: "10:00:00.000 I (1) x: today\n", offset: 28 });
    await advance(4000);
    expect(screen.getByTestId("device-log-text").textContent).toBe("09:00:00.000 I (1) x: older day\n");
    expect(api.getDeviceLog).toHaveBeenCalledTimes(2);  // the older day is read once, not polled

    fireEvent.click(screen.getByRole("button", { name: "2026-10-18" }));
    fireEvent.click(screen.getByRole("option", { name: "2026-10-19" }));
    await advance(2000);
    const calls = api.getDeviceLog.mock.calls.length;
    unmount();
    await advance(4000);
    expect(api.getDeviceLog).toHaveBeenCalledTimes(calls);
  });
});
