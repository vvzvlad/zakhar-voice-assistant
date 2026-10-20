// @vitest-environment jsdom
// Component tests for SchemaForm's widget selection (SchemaField). We assert the
// IDENTITY of the chosen widget (which primitive renders) and the onChange payload,
// NOT exact markup. This catches two concrete defects:
//   - an API-key string rendered as a plaintext input (secret leak), and
//   - a float value flowing into an integer field (pydantic 422).
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
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
  it("renders a masked password input (KeyInput) for a node carrying secret:true", () => {
    // Masking is EXPLICIT: it is driven by the resolved node's `secret` flag (set by
    // the backend via json_schema_extra), NOT by the field name.
    const c = renderField("api_key", { type: "string", secret: true }, "sk-123");
    const input = c.querySelector("input");
    expect(input).toBeTruthy();
    // KeyInput identity: a password-type input (masked) rather than a plaintext one.
    expect(input.getAttribute("type")).toBe("password");
    // Toggling Show/Hide is unique to KeyInput.
    expect(screen.getByText("Show")).toBeInTheDocument();
  });

  it("renders a plaintext input (NOT masked) for a string WITHOUT the secret flag", () => {
    const c = renderField("endpoint", { type: "string" }, "http://x");
    const input = c.querySelector("input");
    // No password masking, and no Show/Hide button -> not a KeyInput.
    expect(input.getAttribute("type")).not.toBe("password");
    expect(screen.queryByText("Show")).toBeNull();
  });

  it("does NOT mask a 'key'-ish field name without the secret flag (e.g. keywords)", () => {
    // Only `secret: true` masks a field: a name containing "key" (the wakeword `keywords`
    // field) is a plain text input.
    const c = renderField("keywords", { type: "string" }, "computer, alexa");
    const input = c.querySelector("input");
    expect(input.getAttribute("type")).not.toBe("password");
    expect(screen.queryByText("Show")).toBeNull();
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

  it("decorates enum Select options with enumLabels but emits the bare value", () => {
    const onChange = vi.fn();
    const c = renderField(
      "model_size",
      {
        type: "string", widget: "select",
        enum: ["tiny", "turbo3.1", "big_poetry", "turbo"],
        enumLabels: {
          tiny: "tiny (~10 MB)", "turbo3.1": "turbo3.1 (~360 MB)",
          big_poetry: "big_poetry (~700 MB)", turbo: "turbo (~330 MB)",
        },
      },
      "turbo3.1", onChange
    );
    // The collapsed trigger shows the decorated label, not the bare id.
    expect(c.querySelector(".z-select").textContent).toContain("turbo3.1 (~360 MB)");
    // Opening and picking another option emits the BARE model id (not the label).
    fireEvent.click(c.querySelector(".z-select"));
    fireEvent.click(screen.getByText("big_poetry (~700 MB)"));
    expect(onChange).toHaveBeenCalledWith("model_size", "big_poetry");
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

  it("auto-enables in-dropdown search for a dynamic field whose list exceeds 10 options", async () => {
    const many = Array.from({ length: 12 }, (_, i) => `model-${i}`);
    const schema = { properties: { model: { type: "string", options: "dynamic" } } };
    const { container } = render(
      <SchemaForm schema={schema} values={{ model: "model-0" }} onChange={() => {}}
        optionsFor={async () => many} />
    );
    fireEvent.click(container.querySelector(".z-select"));
    // The search box appears once the long list has loaded.
    const input = await screen.findByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "model-11" } });
    const opts = screen.getAllByRole("option");
    expect(opts).toHaveLength(1);
    expect(opts[0].textContent).toContain("model-11");
  });

  it("keeps a short NON-freeform dynamic list (<= 10 options) non-searchable", async () => {
    const schema = { properties: { voice: { type: "string", options: "dynamic" } } };
    const { container } = render(
      <SchemaForm schema={schema} values={{ voice: "a" }} onChange={() => {}}
        optionsFor={async () => ["a", "b", "c"]} />
    );
    fireEvent.click(container.querySelector(".z-select"));
    await waitFor(() => expect(screen.getAllByRole("option")).toHaveLength(3));
    expect(screen.queryByPlaceholderText("Search…")).toBeNull();
  });

  it("a FREEFORM dynamic field keeps the search input even for a short list", async () => {
    // A freeform model field whose provider returns a short/empty
    // list (e.g. missing api_key) must still allow typing an arbitrary id.
    const onChange = vi.fn();
    const schema = { properties: { model: { type: "string", options: "dynamic", freeform: true } } };
    const { container } = render(
      <SchemaForm schema={schema} values={{ model: "a" }} onChange={onChange}
        optionsFor={async () => ["a", "b"]} />
    );
    fireEvent.click(container.querySelector(".z-select"));
    const input = await screen.findByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "vendor/typed-by-hand" } });
    fireEvent.click(screen.getByText('Use "vendor/typed-by-hand"'));
    expect(onChange).toHaveBeenCalledWith("model", "vendor/typed-by-hand");
  });

  it("propagates the freeform annotation: the custom 'Use ...' row emits the raw query", async () => {
    const onChange = vi.fn();
    const many = Array.from({ length: 12 }, (_, i) => `model-${i}`);
    const schema = { properties: { model: { type: "string", options: "dynamic", freeform: true } } };
    const { container } = render(
      <SchemaForm schema={schema} values={{ model: "model-0" }} onChange={onChange}
        optionsFor={async () => many} />
    );
    fireEvent.click(container.querySelector(".z-select"));
    const input = await screen.findByPlaceholderText("Search…");
    fireEvent.change(input, { target: { value: "vendor/not-in-list" } });
    fireEvent.click(screen.getByText('Use "vendor/not-in-list"'));
    expect(onChange).toHaveBeenCalledWith("model", "vendor/not-in-list");
  });

  it("renders a string-array field as a joined text input and emits a real array on edit", () => {
    // A list[str] field (e.g. wakeword `keywords`) must NOT fall through to the plain
    // text input (which would store a string and 422 on pydantic list[str]). It renders
    // the array joined by ", " and, on edit, stores a trimmed array with empties dropped.
    const onChange = vi.fn();
    const c = renderField(
      "keywords",
      { type: "array", items: { type: "string" } },
      ["окей захар", "захар"],
      onChange
    );
    const input = c.querySelector(".z-inp input");
    expect(input).toBeTruthy();
    // The current array is shown joined by ", " (not coerced via String([...]) with bare commas).
    expect(input.value).toBe("окей захар, захар");
    // Typing "a, b, c" stores ["a","b","c"] — a real array, trimmed, no empties.
    fireEvent.change(input, { target: { value: "a, b, c" } });
    expect(onChange).toHaveBeenCalledWith("keywords", ["a", "b", "c"]);
    const emitted = onChange.mock.calls[onChange.mock.calls.length - 1][1];
    expect(Array.isArray(emitted)).toBe(true);
    // Trailing/empty fragments and stray whitespace are dropped (commas + newlines).
    fireEvent.change(input, { target: { value: " x , ,\n y ,," } });
    expect(onChange).toHaveBeenLastCalledWith("keywords", ["x", "y"]);
  });

  it("preserves an in-progress separator while typing (controlled round-trip): 'захар, ' is NOT reset to 'захар'", () => {
    // Reproduces the controlled-input bug: the input is re-rendered from the value
    // its own onChange just emitted (the real SchemaForm round-trip). With the old
    // inline input (value={arr.join(", ")} + parse on every keystroke) the trailing
    // ", " is stripped immediately, so a comma/space could never be typed. The
    // StringListInput holds the raw text locally, so the separator survives while
    // the STORED value stays a real array.
    function Controlled() {
      // Mirror renderField's one-property schema, but keep the value in local state
      // and feed the emitted array back in — exactly how SchemaForm is driven.
      const [val, setVal] = React.useState(["захар"]);
      const schema = { properties: { keywords: { type: "array", items: { type: "string" } } } };
      return (
        <SchemaForm
          schema={schema}
          values={{ keywords: val }}
          onChange={(_f, v) => setVal(v)}
        />
      );
    }
    const { container } = render(<Controlled />);
    const input = container.querySelector(".z-inp input");
    expect(input.value).toBe("захар");
    // Typing a trailing comma+space: the displayed text keeps the separator (so the
    // user can keep typing), while the emitted/stored value is the parsed ["захар"].
    fireEvent.change(input, { target: { value: "захар, " } });
    expect(input.value).toBe("захар, ");
    // Continuing to a second entry parses to a two-element array.
    fireEvent.change(input, { target: { value: "захар, оскар" } });
    expect(input.value).toBe("захар, оскар");
  });

  it("emits the parsed array across a controlled round-trip (separator typed, then second entry)", () => {
    // Companion assertion to the test above, focused on the emitted PARSED value:
    // 'захар, ' -> ['захар'] (trailing empty fragment dropped), then
    // 'захар, оскар' -> ['захар','оскар'].
    const emitted = [];
    function Controlled() {
      const [val, setVal] = React.useState(["захар"]);
      const schema = { properties: { keywords: { type: "array", items: { type: "string" } } } };
      return (
        <SchemaForm
          schema={schema}
          values={{ keywords: val }}
          onChange={(_f, v) => { emitted.push(v); setVal(v); }}
        />
      );
    }
    const { container } = render(<Controlled />);
    const input = container.querySelector(".z-inp input");
    fireEvent.change(input, { target: { value: "захар, " } });
    // The trailing comma+space contributes no entry: still a single-element array.
    expect(emitted[emitted.length - 1]).toEqual(["захар"]);
    fireEvent.change(input, { target: { value: "захар, оскар" } });
    expect(emitted[emitted.length - 1]).toEqual(["захар", "оскар"]);
  });

  it("re-syncs displayed text on a BIDIRECTIONAL external value reset (multi-word phrase vs split entries)", () => {
    // Keyword entries may contain a space (a multi-word wake phrase, e.g. "окей захар"). The
    // re-sync effect must compare structurally: join(" ") makes ["окей захар"] and
    // ["окей","захар"] look equal, so an external value reset (save/reload re-seeding the form
    // via useStageForm without a remount) would be skipped. `rerender` keeps the StringListInput
    // mounted.
    const schema = { properties: { keywords: { type: "array", items: { type: "string" } } } };
    const onChange = vi.fn();
    const { container, rerender } = render(
      <SchemaForm schema={schema} values={{ keywords: ["окей захар"] }} onChange={onChange} />
    );
    const input = container.querySelector(".z-inp input");
    // One-phrase value renders as the single phrase (its inner space is NOT a separator).
    expect(input.value).toBe("окей захар");

    // The user edits the text to two comma-separated entries; the parsed array is emitted.
    fireEvent.change(input, { target: { value: "окей, захар" } });
    expect(input.value).toBe("окей, захар");
    expect(onChange).toHaveBeenLastCalledWith("keywords", ["окей", "захар"]);

    // EXTERNAL reset back to the ONE-phrase value (same component instance, no remount).
    // With the old join(" ") comparison both sides stringify to "окей захар", so the
    // effect skips the re-sync and the input keeps the stale "окей, захар". With the
    // JSON.stringify comparison the parsed ["окей","захар"] != ["окей захар"], so it
    // re-syncs and the input shows the canonical "окей захар".
    rerender(
      <SchemaForm schema={schema} values={{ keywords: ["окей захар"] }} onChange={onChange} />
    );
    expect(input.value).toBe("окей захар");

    // The reverse direction: external reset from a one-phrase value to a two-entry value
    // updates the displayed text (joined by ", ").
    rerender(
      <SchemaForm schema={schema} values={{ keywords: ["окей", "захар"] }} onChange={onChange} />
    );
    expect(input.value).toBe("окей, захар");
  });

  it("treats an array field with unspecified item type as a string array", () => {
    // No `items` schema -> default to string-array handling (joined text input), not
    // the generic plaintext fallback.
    const onChange = vi.fn();
    const c = renderField("tags", { type: "array" }, ["one", "two"], onChange);
    const input = c.querySelector(".z-inp input");
    expect(input.value).toBe("one, two");
    fireEvent.change(input, { target: { value: "p, q" } });
    expect(onChange).toHaveBeenCalledWith("tags", ["p", "q"]);
  });

  it("shows the comma-separated hint for a string-array field", () => {
    renderField("keywords", { type: "array", items: { type: "string" }, description: "Wake words." }, ["захар"]);
    // The widget format note is appended to the schema description.
    expect(screen.getByText(/Comma-separated/)).toBeInTheDocument();
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

describe("SchemaField percent fields", () => {
  // A probability stored as 0..1 shows and is edited as 0-100 %, and is written back as 0..1.
  function Controlled({ node, onChange, initial = 0.5 }) {
    const [val, setVal] = React.useState(initial);
    return <SchemaForm schema={{ properties: { p: node } }} values={{ p: val }}
      onChange={(f, v) => { onChange(f, v); setVal(v); }} />;
  }

  it("a stepper shows 0.5 as 50 % and writes 70 and 50 back as 0.7 and 0.5", () => {
    const onChange = vi.fn();
    const { container } = render(<Controlled node={{ type: "number", minimum: 0, maximum: 1, percent: true }} onChange={onChange} />);
    const input = container.querySelector(".z-stepper input");
    expect(input.value).toBe("50");
    expect(container.querySelector(".z-stepper").nextSibling.textContent).toBe("%");
    fireEvent.change(input, { target: { value: "70" } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenLastCalledWith("p", 0.7);
    expect(input.value).toBe("70");
    fireEvent.change(input, { target: { value: "50" } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenLastCalledWith("p", 0.5);
  });

  it("a stored fractional percent shows whole and is not rewritten by a focus and blur", () => {
    const onChange = vi.fn();
    const { container } = render(<Controlled node={{ type: "number", minimum: 0, maximum: 1, percent: true }} onChange={onChange} initial={0.855} />);
    const input = container.querySelector(".z-stepper input");
    expect(input.value).toBe("86");
    fireEvent.focus(input);
    fireEvent.blur(input);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("a slider shows 0.5 as 50% and writes a drag to 50 % back as 0.5", () => {
    const onChange = vi.fn();
    const { container } = render(<Controlled node={{ type: "number", minimum: 0, maximum: 1, widget: "slider", percent: true }} onChange={onChange} />);
    expect(container.querySelector(".z-sval").textContent).toBe("50%");
    const track = container.querySelector(".z-trk");
    track.getBoundingClientRect = () => ({ left: 0, width: 100, top: 0, height: 10, right: 100, bottom: 10 });
    fireEvent(track, new MouseEvent("pointerdown", { bubbles: true, clientX: 70 }));
    expect(onChange).toHaveBeenLastCalledWith("p", 0.7);
    expect(container.querySelector(".z-sval").textContent).toBe("70%");
    fireEvent(track, new MouseEvent("pointerdown", { bubbles: true, clientX: 50 }));
    expect(onChange).toHaveBeenLastCalledWith("p", 0.5);
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

  it("names a numeric enum by its string-keyed enumLabels and stores the number", () => {
    // The schema as the API sends it: JSON turns the labels' int keys into strings.
    const onChange = vi.fn();
    const c = renderField("mic_channel", { type: "integer", enum: [0, 1], enumLabels: { "0": "Processed", "1": "Raw" } }, 1, onChange);
    const btns = c.querySelectorAll(".z-seg.full button");
    expect([...btns].map((b) => b.textContent)).toEqual(["Processed", "Raw"]);
    expect(btns[1]).toHaveClass("on");
    fireEvent.click(btns[0]);
    expect(onChange).toHaveBeenCalledWith("mic_channel", 0);
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

describe("SchemaForm advanced fields", () => {
  it("puts `advanced: true` fields into a closed Advanced disclosure after the others", () => {
    const schema = { properties: { timeout_s: { type: "integer", advanced: true }, model: { type: "string" } } };
    const { container } = render(<SchemaForm schema={schema} values={{ timeout_s: 5, model: "m" }} onChange={() => {}} />);
    const details = container.querySelector("details");
    expect(details.open).toBe(false);
    expect(details.querySelector("summary").textContent).toBe("Advanced");
    expect(details.textContent).toContain("Timeout S");
    expect(details.textContent).not.toContain("Model");
    expect(container.lastElementChild).toBe(details);
  });

  it("renders no disclosure when no field is advanced", () => {
    const c = renderField("model", { type: "string" }, "m");
    expect(c.querySelector("details")).toBeNull();
  });

  it("puts the fields of a `group` after the ungrouped ones, under the group's name", () => {
    const schema = { properties: {
      reply_a: { type: "string", title: "Reply A", advanced: true, group: "Fallback replies" },
      timeout: { type: "integer", title: "Timeout", advanced: true },
      reply_b: { type: "string", title: "Reply B", advanced: true, group: "Fallback replies" },
    } };
    const { container } = render(<SchemaForm schema={schema} values={{}} onChange={() => {}} />);
    const order = [...container.querySelectorAll("details .z-fl b, details .z-sl")].map((e) => e.textContent);
    expect(order).toEqual(["Timeout", "Fallback replies", "Reply A", "Reply B"]);
  });
});

describe("SchemaForm enum labels", () => {
  it("names a short enum's options by enumLabels and stores the bare value", () => {
    const onChange = vi.fn();
    renderField("on_error", { type: "string", enum: ["open", "closed"], enumLabels: { open: "Accept the wake", closed: "Reject the wake" } }, "open", onChange);
    expect(screen.queryByText("open")).toBeNull();
    fireEvent.click(screen.getByText("Reject the wake"));
    expect(onChange).toHaveBeenCalledWith("on_error", "closed");
  });
});

describe("SchemaForm text fonts", () => {
  it("shows a `phrase` field in the text font and an id field in monospace", () => {
    expect(renderField("reply", { type: "string", phrase: true }, "Готово.").querySelector(".z-inp")).not.toHaveClass("mono");
    cleanup();
    expect(renderField("model", { type: "string" }, "m").querySelector(".z-inp")).toHaveClass("mono");
    cleanup();
    expect(renderField("keywords", { type: "array", items: { type: "string" }, phrase: true }, ["захар"]).querySelector(".z-inp")).not.toHaveClass("mono");
  });
});

describe("SchemaForm hiddenWhen", () => {
  it("leaves out a `hiddenWhen` field while its boolean sibling is on", () => {
    const schema = { properties: {
      tiny_mode: { type: "boolean", title: "Low-memory mode" },
      use_dictionary: { type: "boolean", title: "Stress dictionary", hiddenWhen: "tiny_mode" },
    } };
    const { rerender } = render(<SchemaForm schema={schema} values={{ tiny_mode: false, use_dictionary: true }} onChange={() => {}} />);
    expect(screen.getByText("Stress dictionary")).toBeInTheDocument();
    rerender(<SchemaForm schema={schema} values={{ tiny_mode: true, use_dictionary: true }} onChange={() => {}} />);
    expect(screen.queryByText("Stress dictionary")).toBeNull();
    expect(screen.getByText("Low-memory mode")).toBeInTheDocument();
  });
});
