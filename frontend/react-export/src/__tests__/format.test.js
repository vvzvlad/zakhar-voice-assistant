// Unit tests for the display formatters (src/format.js).
// TZ is pinned to UTC in src/test/setup.js so fmtStarted is deterministic.
import { describe, it, expect } from "vitest";
import { fmtUptime, fmtStarted } from "../format.js";

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
