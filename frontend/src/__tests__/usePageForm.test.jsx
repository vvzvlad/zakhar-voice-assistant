// @vitest-environment jsdom
// Tests for the page form (src/usePageForm.js) on a FormPage: parts merge into
// one patch, the save bar shows only while there is something to save, inputs
// are disabled while saving, a successful save leaves no edit behind, and a
// rejected one keeps the drafts and shows the server's errors.
import React, { useState } from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, act } from "@testing-library/react";
import { mergePatch, usePageForm } from "../usePageForm.js";
import { FormPage } from "../components/primitives.jsx";

afterEach(cleanup);

describe("mergePatch", () => {
  it("merges objects and replaces lists and scalars", () => {
    expect(mergePatch(
      { core: { vad: { silence_ms: 900 }, chains: [1, 2] }, x: 1 },
      { core: { vad: { mic_channel: 1 }, chains: [3] }, x: 2 },
    )).toEqual({ core: { vad: { silence_ms: 900, mic_channel: 1 }, chains: [3] }, x: 2 });
  });
});

// One-part page whose upstream values the test changes, as a catalog refresh would.
let setUpstream;
function Page({ patch }) {
  const [values, setValues] = useState({ name: "a" });
  setUpstream = setValues;
  const form = usePageForm([{ key: "p", values, toPatch: (d) => ({ p: d }) }], patch);
  return <FormPage form={form}>
    <input aria-label="name" value={form.draftOf("p").name} onChange={(e) => form.onChangeOf("p")("name", e.target.value)} />
  </FormPage>;
}

const input = () => screen.getByLabelText("name");
const type = (v) => fireEvent.change(input(), { target: { value: v } });
const saveButton = () => screen.getByRole("button", { name: /Save changes|Saving…/ });

describe("usePageForm on a FormPage", () => {
  it("shows the save bar only while there are unsaved changes", () => {
    render(<Page patch={vi.fn()} />);
    expect(screen.queryByTestId("save-bar")).toBeNull();
    type("b");
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    type("a");
    expect(screen.queryByTestId("save-bar")).toBeNull();
  });

  it("disables inputs while saving, ignores an edit made then and leaves no edit behind", async () => {
    let finish;
    const patch = vi.fn(() => new Promise((resolve) => { finish = resolve; }));
    render(<Page patch={patch} />);
    type("b");
    fireEvent.click(saveButton());
    expect(patch).toHaveBeenCalledWith({ p: { name: "b" } });
    expect(input()).toBeDisabled();
    expect(saveButton()).toHaveTextContent("Saving…");
    type("c");
    expect(input()).toHaveValue("b");

    await act(async () => { setUpstream({ name: "b" }); finish(); });
    expect(input()).toHaveValue("b");
    expect(input()).toBeEnabled();
    expect(screen.queryByTestId("save-bar")).toBeNull();
    // The old values coming back must not resurrect an edit from before the save.
    act(() => setUpstream({ name: "a" }));
    expect(input()).toHaveValue("a");
    expect(screen.queryByTestId("save-bar")).toBeNull();
  });

  it("keeps the drafts and shows the server's errors when the save is rejected", async () => {
    const patch = vi.fn(async () => {
      throw Object.assign(new Error("422"), { detail: [{ loc: ["body", "p", "name"], msg: "too short" }] });
    });
    render(<Page patch={patch} />);
    type("b");
    await act(async () => { fireEvent.click(saveButton()); });
    expect(screen.getByText(/Not saved\./).parentElement).toHaveTextContent("p.name: too short");
    expect(input()).toHaveValue("b");
    expect(input()).toBeEnabled();
    expect(saveButton()).toBeEnabled();
  });
});
