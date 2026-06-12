// Unit tests for the display formatters (src/format.js).
// TZ is pinned to UTC in src/test/setup.js so fmtStarted is deterministic.
import { describe, it, expect } from "vitest";
import { fmtUptime, fmtStarted, fmtBytes, deviceVersion } from "../format.js";

describe("fmtUptime", () => {
  it("renders an em dash for null / NaN input", () => {
    expect(fmtUptime(null)).toBe("—");
    expect(fmtUptime(NaN)).toBe("—");
  });

  it("clamps negative seconds to 0m", () => {
    expect(fmtUptime(-50)).toBe("0m");
  });

  it("shows only minutes below one hour", () => {
    expect(fmtUptime(59)).toBe("0m");
    expect(fmtUptime(125)).toBe("2m");
  });

  it("shows hours + zero-padded minutes at the 3600s boundary", () => {
    // 3600s -> exactly 1h 00m (minutes zero-padded).
    expect(fmtUptime(3600)).toBe("1h 00m");
    expect(fmtUptime(3600 + 5 * 60)).toBe("1h 05m");
  });

  it("shows days + zero-padded hours and minutes at the 86400s boundary", () => {
    // 86400s -> exactly 1d 00h 00m.
    expect(fmtUptime(86400)).toBe("1d 00h 00m");
    expect(fmtUptime(86400 + 3600 + 60)).toBe("1d 01h 01m");
  });
});

describe("fmtStarted", () => {
  it("formats a valid ISO timestamp as padded 'YYYY-MM-DD HH:MM' (UTC)", () => {
    expect(fmtStarted("2024-01-02T03:04:05Z")).toBe("2024-01-02 03:04");
  });

  it("returns the raw input verbatim for an unparseable string", () => {
    expect(fmtStarted("not a date")).toBe("not a date");
  });

  it("returns an em dash for a falsy input", () => {
    expect(fmtStarted("")).toBe("—");
    expect(fmtStarted(null)).toBe("—");
    expect(fmtStarted(undefined)).toBe("—");
  });
});

describe("fmtBytes", () => {
  it("renders an em dash for null / NaN input", () => {
    expect(fmtBytes(null)).toBe("—");
    expect(fmtBytes(undefined)).toBe("—");
    expect(fmtBytes(NaN)).toBe("—");
  });

  it("renders raw bytes below 1 KiB", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(512)).toBe("512 B");
  });

  it("renders one decimal in KB for values under 10", () => {
    expect(fmtBytes(1024)).toBe("1.0 KB");
    expect(fmtBytes(1536)).toBe("1.5 KB");
  });

  it("steps up to MB at the 1 MiB boundary", () => {
    expect(fmtBytes(1048576)).toBe("1.0 MB");
  });

  it("rounds to an integer when the value is >= 10 in its unit", () => {
    expect(fmtBytes(12 * 1024 * 1024)).toBe("12 MB");
  });
});

describe("deviceVersion", () => {
  const entry = {
    name: "kitchen", online: true, enabled: true,
    versions: [
      { id: "config_version", name: "Config Version", value: "v16" },
      { id: "model_version", name: "Model Version", value: "night-v3" },
    ],
  };

  it("returns the matching version value by id", () => {
    expect(deviceVersion(entry, "config_version")).toBe("v16");
    expect(deviceVersion(entry, "model_version")).toBe("night-v3");
  });

  it("returns null when the id is not reported", () => {
    expect(deviceVersion(entry, "nope")).toBeNull();
  });

  it("returns null for a missing entry or empty versions (offline / disabled)", () => {
    expect(deviceVersion(null, "config_version")).toBeNull();
    expect(deviceVersion(undefined, "config_version")).toBeNull();
    expect(deviceVersion({ versions: [] }, "config_version")).toBeNull();
    expect(deviceVersion({}, "config_version")).toBeNull();
  });

  it("returns null when the version value has not arrived yet", () => {
    // The backend reports value: null until the first state push.
    expect(deviceVersion({ versions: [{ id: "config_version", value: null }] }, "config_version")).toBeNull();
  });
});
