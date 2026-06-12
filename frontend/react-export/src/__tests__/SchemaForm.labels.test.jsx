// @vitest-environment jsdom
// Component tests for the persisted human-label companion field (<field>_label):
// a dynamic select stores its chosen option's value AND its human label in a
// hidden companion property, so the collapsed control shows the name on the very
// FIRST render (from the config payload) instead of flickering id->name once the
// catalog loads.
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, act } from "@testing-library/react";
import SchemaForm from "../components/SchemaForm.jsx";

afterEach(() => { cleanup(); });

// Flush the mount-time baseline load (a microtask chain, no timers involved).
const flush = () => act(async () => { await Promise.resolve(); });

// A dynamic select with its hidden <field>_label companion property.
const LABEL_SCHEMA = {
  properties: {
    reference_id: { type: "string", options: "dynamic", freeform: true },
    reference_id_label: { type: "string", hidden: true },
  },
};

describe("hidden <field>_label companion", () => {
  it("does not render the hidden _label property as its own input", async () => {
    const optionsFor = vi.fn(async () => ["id1"]);
    const { container } = render(
      <SchemaForm schema={LABEL_SCHEMA}
        values={{ reference_id: "id1", reference_id_label: "Nice Name" }}
        onChange={() => {}} optionsFor={optionsFor} />
    );
    await flush();
    // Exactly one control renders (the dynamic select); the _label field is hidden.
    expect(container.querySelectorAll(".z-select")).toHaveLength(1);
    // No extra plain text input for the label.
    expect(container.querySelectorAll(".z-inp input")).toHaveLength(0);
    // The field is mounted only once -> optionsFor called only for reference_id.
    expect(optionsFor).toHaveBeenCalledTimes(1);
    expect(optionsFor).toHaveBeenCalledWith("reference_id", undefined);
  });

  it("seeds the collapsed control with the persisted label, no id->name flicker", async () => {
    // The baseline catalog does NOT contain id1, so before this feature the
    // control would have shown the bare id "id1" until a catalog resolve.
    const optionsFor = vi.fn(async () => ["other-1", "other-2"]);
    const { container } = render(
      <SchemaForm schema={LABEL_SCHEMA}
        values={{ reference_id: "id1", reference_id_label: "Nice Name [ru]" }}
        onChange={() => {}} optionsFor={optionsFor} />
    );
    const trigger = container.querySelector(".z-select");
    // Immediately on first render: the persisted label, never the bare id.
    expect(trigger.textContent).toContain("Nice Name [ru]");
    expect(trigger.textContent).not.toContain("id1");
    // After the baseline load flushes: still the persisted label (the current
    // value is kept selectable with its label, prepended to the fetched list).
    await flush();
    expect(trigger.textContent).toContain("Nice Name [ru]");
    expect(trigger.textContent).not.toContain("id1");
  });

  it("shows the stored label with no swap when the value IS in the catalog", async () => {
    // The baseline catalog DOES contain id1 (with a matching label), so the
    // keep-current-value merge returns the fetched list unchanged. The displayed
    // label must equal the stored one both BEFORE and AFTER the baseline flush.
    const optionsFor = vi.fn(async () => [
      { value: "id1", label: "Nice Name [ru]" },
      { value: "id2", label: "Other [en]" },
    ]);
    const { container } = render(
      <SchemaForm schema={LABEL_SCHEMA}
        values={{ reference_id: "id1", reference_id_label: "Nice Name [ru]" }}
        onChange={() => {}} optionsFor={optionsFor} />
    );
    const trigger = container.querySelector(".z-select");
    // Before the baseline load: seeded from the persisted label.
    expect(trigger.textContent).toContain("Nice Name [ru]");
    expect(trigger.textContent).not.toContain("id1");
    // After the baseline load flushes: the fetched list already contains id1 with
    // the same label, so the visible label is unchanged (no id->name swap).
    await flush();
    expect(trigger.textContent).toContain("Nice Name [ru]");
    expect(trigger.textContent).not.toContain("Other [en]");
  });

  it("selecting an option writes BOTH the value and its label", async () => {
    const onChange = vi.fn();
    const optionsFor = vi.fn(async () => [{ value: "id2", label: "Other Name" }]);
    const { container } = render(
      <SchemaForm schema={LABEL_SCHEMA}
        values={{ reference_id: "id1", reference_id_label: "Nice Name" }}
        onChange={onChange} optionsFor={optionsFor} />
    );
    await flush();
    fireEvent.click(container.querySelector(".z-select"));
    const opt = screen.getAllByRole("option").find((o) => o.textContent.includes("Other Name"));
    fireEvent.click(opt);
    expect(onChange).toHaveBeenCalledWith("reference_id", "id2");
    expect(onChange).toHaveBeenCalledWith("reference_id_label", "Other Name");
  });
});
