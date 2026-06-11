// @vitest-environment jsdom
// Component tests for the searchable Select (components/primitives.jsx): the
// in-dropdown search box (long dynamic lists, e.g. LLM model catalogs), keyboard
// behavior, and the allowCustom "Use \"<query>\"" freeform escape hatch.
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { Select } from "../components/primitives.jsx";

afterEach(cleanup);

const OPTS = ["alpha", "beta", "gamma", { value: "x/custom-id", label: "Custom Label" }];

function openSelect(props) {
  const r = render(<Select value="alpha" options={OPTS} {...props} />);
  fireEvent.click(r.container.querySelector(".z-select"));
  return r;
}

const searchInput = (container) => container.querySelector('[role="listbox"] input');

describe("Select without searchable (regression)", () => {
  it("renders no search input and all options when open", () => {
    const { container } = openSelect({});
    expect(searchInput(container)).toBeNull();
    expect(screen.getAllByRole("option")).toHaveLength(OPTS.length);
  });
});

describe("Select searchable", () => {
  it("renders a search input inside the open dropdown", () => {
    const { container } = openSelect({ searchable: true });
    expect(searchInput(container)).toBeTruthy();
  });

  it("filters options case-insensitively by label", () => {
    const { container } = openSelect({ searchable: true });
    fireEvent.change(searchInput(container), { target: { value: "GA" } });
    const opts = screen.getAllByRole("option");
    expect(opts).toHaveLength(1);
    expect(opts[0].textContent).toContain("gamma");
  });

  it("matches on the option VALUE too, not only the label", () => {
    const { container } = openSelect({ searchable: true });
    fireEvent.change(searchInput(container), { target: { value: "custom-id" } });
    const opts = screen.getAllByRole("option");
    expect(opts).toHaveLength(1);
    expect(opts[0].textContent).toContain("Custom Label");
  });

  it("Enter in the search input picks the FIRST filtered option and closes", () => {
    const onChange = vi.fn();
    const { container } = openSelect({ searchable: true, onChange });
    const input = searchInput(container);
    fireEvent.change(input, { target: { value: "be" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith("beta");
    expect(container.querySelector('[role="listbox"]')).toBeNull();
  });

  it("Escape in the search input closes the dropdown", () => {
    const { container } = openSelect({ searchable: true });
    fireEvent.keyDown(searchInput(container), { key: "Escape" });
    expect(container.querySelector('[role="listbox"]')).toBeNull();
  });

  it("resets the query on reopen (full list shows again)", () => {
    const { container } = openSelect({ searchable: true });
    fireEvent.change(searchInput(container), { target: { value: "gamma" } });
    expect(screen.getAllByRole("option")).toHaveLength(1);
    const trigger = container.querySelector(".z-select");
    fireEvent.click(trigger);  // close
    fireEvent.click(trigger);  // reopen
    expect(searchInput(container).value).toBe("");
    expect(screen.getAllByRole("option")).toHaveLength(OPTS.length);
  });
});

describe("Select allowCustom", () => {
  it("offers a 'Use \"<query>\"' row for an unmatched query and emits the raw query", () => {
    const onChange = vi.fn();
    const { container } = openSelect({ searchable: true, allowCustom: true, onChange });
    fireEvent.change(searchInput(container), { target: { value: "my/own-model" } });
    const row = screen.getByText('Use "my/own-model"');
    fireEvent.click(row);
    expect(onChange).toHaveBeenCalledWith("my/own-model");
    expect(container.querySelector('[role="listbox"]')).toBeNull();
  });

  it("does NOT offer the custom row when the query exactly equals an option value", () => {
    const { container } = openSelect({ searchable: true, allowCustom: true });
    fireEvent.change(searchInput(container), { target: { value: "alpha" } });
    expect(screen.queryByText('Use "alpha"')).toBeNull();
  });

  it("does NOT offer the custom row without allowCustom", () => {
    const { container } = openSelect({ searchable: true });
    fireEvent.change(searchInput(container), { target: { value: "my/own-model" } });
    expect(screen.queryByText('Use "my/own-model"')).toBeNull();
  });

  it("Enter with an empty filtered list emits the raw query and closes", () => {
    const onChange = vi.fn();
    const { container } = openSelect({ allowCustom: true, onChange });
    const input = searchInput(container);
    fireEvent.change(input, { target: { value: "my/own-model" } });
    // No option matches, so Enter falls through to the custom value.
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith("my/own-model");
    expect(container.querySelector('[role="listbox"]')).toBeNull();
  });

  it("allowCustom ALONE (no searchable) renders the input even for a short list", () => {
    // Regression: with a short/empty fetched list (e.g. provider with no api_key)
    // the custom value must still be typeable — the input cannot depend on
    // `searchable` alone.
    const onChange = vi.fn();
    const r = render(<Select value="a" options={["a", "b"]} allowCustom onChange={onChange} />);
    fireEvent.click(r.container.querySelector(".z-select"));
    const input = searchInput(r.container);
    expect(input).toBeTruthy();
    fireEvent.change(input, { target: { value: "vendor/custom" } });
    fireEvent.click(screen.getByText('Use "vendor/custom"'));
    expect(onChange).toHaveBeenCalledWith("vendor/custom");
    expect(r.container.querySelector('[role="listbox"]')).toBeNull();
  });
});
