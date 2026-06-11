// @vitest-environment jsdom
// Component tests for the remote-search dynamic select (search: "remote" in the
// field schema, e.g. the fish.audio voice catalog): typing in the dropdown's
// search input re-queries the server (debounced 300 ms) via optionsFor(field,
// query) instead of filtering the preloaded list client-side; emptying the
// query restores the baseline list. Fields WITHOUT the flag keep the single
// mount-time load + local filtering.
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, fireEvent, cleanup, act } from "@testing-library/react";
import SchemaForm from "../components/SchemaForm.jsx";

beforeEach(() => { vi.useFakeTimers(); });
afterEach(() => { vi.useRealTimers(); cleanup(); });

const REMOTE_SCHEMA = {
  properties: {
    reference_id: { type: "string", options: "dynamic", freeform: true, search: "remote" },
  },
};

// optionsFor stub mimicking the backend: a query returns the server-filtered
// hit, no query returns the baseline catalog.
const makeOptionsFor = () =>
  vi.fn(async (field, q) => (q ? [`hit-${q}`] : ["base-1", "base-2"]));

// Flush the mount-time baseline load (a microtask chain, no timers involved).
const flush = () => act(async () => { await Promise.resolve(); });

async function renderRemote(optionsFor) {
  const r = render(
    <SchemaForm schema={REMOTE_SCHEMA} values={{ reference_id: "base-1" }}
      onChange={() => {}} optionsFor={optionsFor} />
  );
  await flush();
  return r;
}

describe("remote-search dynamic select (search: 'remote')", () => {
  it("loads the baseline list once on mount with no query", async () => {
    const optionsFor = makeOptionsFor();
    await renderRemote(optionsFor);
    expect(optionsFor).toHaveBeenCalledTimes(1);
    expect(optionsFor).toHaveBeenCalledWith("reference_id", undefined);
  });

  it("renders the search input even though the baseline list is short", async () => {
    const optionsFor = makeOptionsFor();
    const { container } = await renderRemote(optionsFor);
    fireEvent.click(container.querySelector(".z-select"));
    expect(screen.getByPlaceholderText("Search…")).toBeInTheDocument();
  });

  it("debounces typing and replaces the options with the server result", async () => {
    const optionsFor = makeOptionsFor();
    const { container } = await renderRemote(optionsFor);
    fireEvent.click(container.querySelector(".z-select"));
    const input = screen.getByPlaceholderText("Search…");
    // Two quick keystrokes -> only ONE search request after the 300 ms debounce.
    fireEvent.change(input, { target: { value: "ann" } });
    fireEvent.change(input, { target: { value: "anna" } });
    expect(optionsFor).toHaveBeenCalledTimes(1); // nothing before the debounce fires
    await act(async () => { vi.advanceTimersByTime(300); });
    expect(optionsFor).toHaveBeenCalledTimes(2);
    expect(optionsFor).toHaveBeenLastCalledWith("reference_id", "anna");
    // The list is the server result (plus the still-selectable current value);
    // the old baseline entry is gone — no client-side filtering involved.
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels.some((l) => l.includes("hit-anna"))).toBe(true);
    expect(labels.some((l) => l.includes("base-2"))).toBe(false);
    expect(labels.some((l) => l.includes("base-1"))).toBe(true); // current value kept
  });

  it("emptying the query restores the baseline from cache without a refetch", async () => {
    const optionsFor = makeOptionsFor();
    const { container } = await renderRemote(optionsFor);
    fireEvent.click(container.querySelector(".z-select"));
    const input = screen.getByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "anna" } });
    await act(async () => { vi.advanceTimersByTime(300); });
    expect(optionsFor).toHaveBeenCalledTimes(2); // mount baseline + search
    fireEvent.change(input, { target: { value: "" } });
    // The baseline is back IMMEDIATELY (synchronous cache restore, no debounce,
    // no network) — the backend would just return the same no-query list.
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels.some((l) => l.includes("base-2"))).toBe(true);
    expect(labels.some((l) => l.includes("hit-anna"))).toBe(false);
    await act(async () => { vi.advanceTimersByTime(1000); });
    expect(optionsFor).toHaveBeenCalledTimes(2); // still: no extra load
  });

  it("clearing the query before the debounce fires cancels the pending search", async () => {
    const optionsFor = makeOptionsFor();
    const { container } = await renderRemote(optionsFor);
    fireEvent.click(container.querySelector(".z-select"));
    const input = screen.getByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "an" } });
    fireEvent.change(input, { target: { value: "" } }); // cleared within 300 ms
    await act(async () => { vi.advanceTimersByTime(1000); });
    // No search request was ever made — only the mount-time baseline load.
    expect(optionsFor).toHaveBeenCalledTimes(1);
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels.some((l) => l.includes("base-2"))).toBe(true);
  });

  it("closing and reopening the dropdown shows the baseline again without a refetch", async () => {
    const optionsFor = makeOptionsFor();
    const { container } = await renderRemote(optionsFor);
    const trigger = container.querySelector(".z-select");
    fireEvent.click(trigger);
    const input = screen.getByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "anna" } });
    await act(async () => { vi.advanceTimersByTime(300); });
    expect(optionsFor).toHaveBeenCalledTimes(2); // mount baseline + search
    // Close: the Select resets its query AND notifies the parent, which
    // restores the cached baseline — so reopening shows an empty input over
    // the baseline list, not the stale "anna" results.
    fireEvent.click(trigger);
    fireEvent.click(trigger);
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels.some((l) => l.includes("base-2"))).toBe(true);
    expect(labels.some((l) => l.includes("hit-anna"))).toBe(false);
    await act(async () => { vi.advanceTimersByTime(1000); });
    expect(optionsFor).toHaveBeenCalledTimes(2); // restore came from the cache
  });

  it("ignores an out-of-order response: a slow earlier search never clobbers a later one", async () => {
    // First search resolves AFTER the second one; only the latest result may land.
    const resolvers = [];
    const optionsFor = vi.fn((field, q) => {
      if (!q) return Promise.resolve(["base-1"]);
      return new Promise((res) => resolvers.push({ q, res }));
    });
    const { container } = await renderRemote(optionsFor);
    fireEvent.click(container.querySelector(".z-select"));
    const input = screen.getByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "slow" } });
    await act(async () => { vi.advanceTimersByTime(300); });
    fireEvent.change(input, { target: { value: "fast" } });
    await act(async () => { vi.advanceTimersByTime(300); });
    expect(resolvers.map((r) => r.q)).toEqual(["slow", "fast"]);
    // The fast (latest) request resolves first...
    await act(async () => { resolvers[1].res(["hit-fast"]); });
    // ...then the stale slow one arrives and must be dropped.
    await act(async () => { resolvers[0].res(["hit-slow"]); });
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels.some((l) => l.includes("hit-fast"))).toBe(true);
    expect(labels.some((l) => l.includes("hit-slow"))).toBe(false);
  });

  it("a late search response never clobbers a baseline restored from cache", async () => {
    // Restoring the baseline bumps the request seq, so a search that was still
    // in flight when the query was cleared must be dropped when it resolves.
    const resolvers = [];
    const optionsFor = vi.fn((field, q) => {
      if (!q) return Promise.resolve(["base-1", "base-2"]);
      return new Promise((res) => resolvers.push({ q, res }));
    });
    const { container } = await renderRemote(optionsFor);
    fireEvent.click(container.querySelector(".z-select"));
    const input = screen.getByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "slow" } });
    await act(async () => { vi.advanceTimersByTime(300); }); // search now in flight
    fireEvent.change(input, { target: { value: "" } }); // baseline restored from cache
    await act(async () => { resolvers[0].res(["hit-slow"]); }); // stale response lands late
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels.some((l) => l.includes("base-2"))).toBe(true);
    expect(labels.some((l) => l.includes("hit-slow"))).toBe(false);
  });
});

describe("dynamic select WITHOUT the remote flag (regression)", () => {
  it("loads once on mount and filters locally — typing triggers no reload", async () => {
    const many = Array.from({ length: 12 }, (_, i) => `model-${i}`);
    const optionsFor = vi.fn(async () => many);
    const schema = { properties: { model: { type: "string", options: "dynamic" } } };
    const { container } = render(
      <SchemaForm schema={schema} values={{ model: "model-0" }} onChange={() => {}}
        optionsFor={optionsFor} />
    );
    await flush();
    fireEvent.click(container.querySelector(".z-select"));
    const input = screen.getByPlaceholderText("Search…"); // >10 options -> searchable
    fireEvent.change(input, { target: { value: "model-11" } });
    await act(async () => { vi.advanceTimersByTime(1000); });
    // Still the single mount-time load; the filtering happened client-side.
    expect(optionsFor).toHaveBeenCalledTimes(1);
    const opts = screen.getAllByRole("option");
    expect(opts).toHaveLength(1);
    expect(opts[0].textContent).toContain("model-11");
  });
});
