// @vitest-environment jsdom
// Component tests for SchemaForm's widget selection (SchemaField). We assert the
// IDENTITY of the chosen widget (which primitive renders) and the onChange payload,
// NOT exact markup. This catches two concrete defects:
//   - an API-key string rendered as a plaintext input (secret leak), and
//   - a float value flowing into an integer field (pydantic 422).
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import SchemaForm from "../components/SchemaForm.jsx";

afterEach(cleanup);

// Render a one-property schema and return the rendered container.
function renderField(name, node, value, onChange = () => {}) {
  const schema = { properties: { [name]: node } };
  const { container } = render(
    <SchemaForm schema={schema} values={{ [name]: value }} onChange={onChange} />
  );
  return container;
}

describe("SchemaField widget selection", () => {
  it("renders a masked password input (KeyInput) for a secret-named string", () => {
    const c = renderField("api_key", { type: "string" }, "sk-123");
    const input = c.querySelector("input");
    expect(input).toBeTruthy();
    // KeyInput identity: a password-type input (masked) rather than a plaintext one.
    expect(input.getAttribute("type")).toBe("password");
    // Toggling SHOW/HIDE is unique to KeyInput.
    expect(screen.getByText("SHOW")).toBeInTheDocument();
  });

  it("renders a plaintext input (NOT masked) for a non-secret string", () => {
    const c = renderField("endpoint", { type: "string" }, "http://x");
    const input = c.querySelector("input");
    // No password masking, and no SHOW/HIDE button -> not a KeyInput.
    expect(input.getAttribute("type")).not.toBe("password");
    expect(screen.queryByText("SHOW")).toBeNull();
  });

  it("renders a Seg for an enum of length <= 3", () => {
    const c = renderField("mode", { type: "string", enum: ["a", "b", "c"] }, "a");
    // Seg identity: .z-seg with one <button> per option; not a .z-select.
    expect(c.querySelector(".z-seg")).toBeTruthy();
    expect(c.querySelector(".z-select")).toBeNull();
    expect(c.querySelectorAll(".z-seg button")).toHaveLength(3);
  });

  it("renders a Select for an enum of length > 3", () => {
    const c = renderField("mode", { type: "string", enum: ["a", "b", "c", "d"] }, "a");
    // Select identity: .z-select role=button with listbox popup affordance.
    expect(c.querySelector(".z-select")).toBeTruthy();
    expect(c.querySelector(".z-seg")).toBeNull();
  });

  it("renders a Toggle for a boolean even when it carries an enum", () => {
    // boolean wins over the enum branch (type check precedes the enum length split).
    const c = renderField("enabled", { type: "boolean", enum: [true, false] }, true);
    const toggle = c.querySelector(".z-toggle");
    expect(toggle).toBeTruthy();
    expect(toggle.getAttribute("role")).toBe("switch");
    expect(c.querySelector(".z-seg")).toBeNull();
  });

  it("renders a Slider for integer with min/max and widget:'slider'", () => {
    const c = renderField("vol", { type: "integer", minimum: 0, maximum: 10, widget: "slider" }, 5);
    expect(c.querySelector(".z-slider")).toBeTruthy();
    expect(c.querySelector(".z-stepper")).toBeNull();
  });

  it("renders a Stepper for an integer without the slider widget", () => {
    const c = renderField("port", { type: "integer", minimum: 0, maximum: 9999 }, 8080);
    expect(c.querySelector(".z-stepper")).toBeTruthy();
    expect(c.querySelector(".z-slider")).toBeNull();
  });

  it("renders a DynamicSelect (.z-select) for a string field with options:'dynamic' and an optionsFor", () => {
    const schema = { properties: { sound_path: { type: "string", options: "dynamic" } } };
    const { container } = render(
      <SchemaForm
        schema={schema}
        values={{ sound_path: "assets/chimes/a.wav" }}
        onChange={() => {}}
        optionsFor={async () => ["assets/chimes/a.wav", "assets/chimes/b.wav"]}
      />
    );
    // The dynamic branch renders a Select (initial single-option fallback from the value),
    // not a plaintext input.
    expect(container.querySelector(".z-select")).toBeTruthy();
    expect(container.querySelector(".z-inp input")).toBeNull();
  });

  it("renders a textarea for a string field with widget:'textarea'", () => {
    const onChange = vi.fn();
    const c = renderField("prompt", { type: "string", widget: "textarea" }, "hello", onChange);
    const ta = c.querySelector("textarea");
    expect(ta).toBeTruthy();
    // It is NOT the generic single-line .z-inp input.
    expect(c.querySelector(".z-inp input")).toBeNull();
    fireEvent.change(ta, { target: { value: "world" } });
    expect(onChange).toHaveBeenCalledWith("prompt", "world");
  });
});

describe("SchemaField numeric onChange payload", () => {
  it("emits a rounded INTEGER for an integer field (float typed -> Math.round)", () => {
    const onChange = vi.fn();
    const c = renderField("port", { type: "integer", minimum: 0, maximum: 9999 }, 8080, onChange);
    const input = c.querySelector(".z-stepper input");
    // Type a fractional value and commit on blur; an integer field must never emit a float.
    fireEvent.change(input, { target: { value: "1.5" } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenCalledWith("port", 2);
    // Assert the emitted numeric is an integer (the 422-prevention contract).
    const emitted = onChange.mock.calls[onChange.mock.calls.length - 1][1];
    expect(Number.isInteger(emitted)).toBe(true);
  });

  it("keeps decimals for a number (float) field", () => {
    const onChange = vi.fn();
    // multipleOf:0.1 -> fractional stepper; temperature-style field.
    const c = renderField(
      "temperature",
      { type: "number", minimum: 0, maximum: 2, multipleOf: 0.1 },
      0.5,
      onChange
    );
    const input = c.querySelector(".z-stepper input");
    fireEvent.change(input, { target: { value: "1.5" } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenCalledWith("temperature", 1.5);
  });
});

describe("SchemaField ScaleSeg (segment scale) selection", () => {
  it("renders a labeled ScaleSeg (word labels + poles + readout) for an integer with choices", () => {
    const onChange = vi.fn();
    const c = renderField(
      "aggressiveness",
      {
        type: "integer", minimum: 0, maximum: 3,
        choices: [
          { value: 0, label: "Lenient" },
          { value: 1, label: "Balanced" },
          { value: 2, label: "Strict" },
          { value: 3, label: "Strictest" },
        ],
        poles: ["waits longest", "cuts off soonest"],
        readout: true,
      },
      2, onChange
    );
    // Segment buttons carry WORD labels (not numbers) and there is no Stepper.
    const btns = c.querySelectorAll(".z-seg.full button");
    expect(btns).toHaveLength(4);
    expect([...btns].map((b) => b.textContent)).toEqual(["Lenient", "Balanced", "Strict", "Strictest"]);
    expect(c.querySelector(".z-stepper")).toBeNull();
    // Pole captions and the "label · value" readout render.
    expect(screen.getByText("waits longest")).toBeInTheDocument();
    expect(screen.getByText("cuts off soonest")).toBeInTheDocument();
    expect(screen.getByText("Strict · 2")).toBeInTheDocument();
    // Clicking a segment emits the NUMERIC value, not the label string.
    fireEvent.click(btns[0]);
    expect(onChange).toHaveBeenCalledWith("aggressiveness", 0);
  });

  it("renders a numeric ScaleSeg with pole captions for an enum decorated with poles", () => {
    const onChange = vi.fn();
    const c = renderField(
      "mic_channel",
      { type: "integer", enum: [0, 1], poles: ["more processed / louder", "raw / cleaner"] },
      0, onChange
    );
    const btns = c.querySelectorAll(".z-seg.full button");
    expect(btns).toHaveLength(2);
    // Numeric segment labels (the enum values), pole captions present.
    expect([...btns].map((b) => b.textContent)).toEqual(["0", "1"]);
    expect(screen.getByText("more processed / louder")).toBeInTheDocument();
    expect(screen.getByText("raw / cleaner")).toBeInTheDocument();
    // Emits the numeric enum value on click.
    fireEvent.click(btns[1]);
    expect(onChange).toHaveBeenCalledWith("mic_channel", 1);
  });

  it("ScaleSeg with an undefined value renders the buttons but no readout and does not crash", () => {
    const c = renderField(
      "aggressiveness",
      {
        type: "integer",
        choices: [{ value: 0, label: "Lenient" }, { value: 1, label: "Balanced" }],
        readout: true,
      },
      undefined
    );
    // Buttons still render; with no current value there is no "·" readout line.
    expect(c.querySelectorAll(".z-seg.full button")).toHaveLength(2);
    expect(c.textContent).not.toContain("·");
  });
});
