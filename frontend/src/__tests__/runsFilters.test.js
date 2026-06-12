// Unit tests for matchesFilters (src/runsFilters.js), extracted from pages/log.jsx
// (R-FE-1). Decides whether a live-pushed run matches the current UI filters.
import { describe, it, expect } from "vitest";
import { matchesFilters } from "../runsFilters.js";

const NO_FILTERS = { result: "all", search: "", device: "" };

describe("matchesFilters", () => {
  it("passes every row when all filters are empty", () => {
    const row = { device: "kitchen", result: "ok", stt_text: "x", llm_text: "y" };
    expect(matchesFilters(row, NO_FILTERS)).toBe(true);
  });

  it("device filter is exact-match and is trimmed before comparing", () => {
    const row = { device: "kitchen", result: "ok", stt_text: "", llm_text: "" };
    expect(matchesFilters(row, { ...NO_FILTERS, device: "  kitchen  " })).toBe(true);
    expect(matchesFilters(row, { ...NO_FILTERS, device: "bedroom" })).toBe(false);
    // Substring of the device name must NOT match (exact only).
    expect(matchesFilters(row, { ...NO_FILTERS, device: "kit" })).toBe(false);
  });

  it("result 'errors' accepts only 'error' rows", () => {
    expect(matchesFilters({ device: "d", result: "error" }, { ...NO_FILTERS, result: "errors" })).toBe(true);
    expect(matchesFilters({ device: "d", result: "ok" }, { ...NO_FILTERS, result: "errors" })).toBe(false);
    expect(matchesFilters({ device: "d", result: "tool" }, { ...NO_FILTERS, result: "errors" })).toBe(false);
  });

  it("result 'ok' accepts both 'ok' AND 'tool' rows, but not errors/empty", () => {
    expect(matchesFilters({ device: "d", result: "ok" }, { ...NO_FILTERS, result: "ok" })).toBe(true);
    expect(matchesFilters({ device: "d", result: "tool" }, { ...NO_FILTERS, result: "ok" })).toBe(true);
    expect(matchesFilters({ device: "d", result: "error" }, { ...NO_FILTERS, result: "ok" })).toBe(false);
    expect(matchesFilters({ device: "d", result: "empty" }, { ...NO_FILTERS, result: "ok" })).toBe(false);
  });

  it("search is a case-insensitive substring over stt_text + llm_text", () => {
    const row = { device: "d", result: "ok", stt_text: "Turn ON the Light", llm_text: "Done" };
    // Matches recognized text, case-insensitively.
    expect(matchesFilters(row, { ...NO_FILTERS, search: "light" })).toBe(true);
    // Matches response text too.
    expect(matchesFilters(row, { ...NO_FILTERS, search: "DONE" })).toBe(true);
    // No match -> filtered out.
    expect(matchesFilters(row, { ...NO_FILTERS, search: "weather" })).toBe(false);
  });

  it("tolerates missing stt_text/llm_text when searching", () => {
    const row = { device: "d", result: "ok" };
    expect(matchesFilters(row, { ...NO_FILTERS, search: "anything" })).toBe(false);
    // Empty/whitespace search is treated as no search.
    expect(matchesFilters(row, { ...NO_FILTERS, search: "   " })).toBe(true);
  });
});
