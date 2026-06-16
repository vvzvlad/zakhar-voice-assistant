// Unit tests for the pydantic-JSON-Schema resolver (src/schema.js).
// This is security-relevant: a wrong resolution can cause a secret-typed field
// to be rendered as a plaintext text input instead of a masked key input. Masking
// itself is now driven by the explicit `secret` flag on the resolved node, not by a
// name heuristic; the widget-selection tests for that live in SchemaForm.widgets.test.jsx.
import { describe, it, expect } from "vitest";
import { deref, resolve, enumOf, humanize } from "../schema.js";

describe("deref", () => {
  it("follows a #/$defs/Name $ref to its target", () => {
    const root = { $defs: { Foo: { type: "object", title: "Foo" } } };
    expect(deref({ $ref: "#/$defs/Foo" }, root)).toEqual({ type: "object", title: "Foo" });
  });

  it("preserves a sibling `default` next to the $ref (merged over the target)", () => {
    const root = { $defs: { Color: { enum: ["red", "green"] } } };
    // The default lives on the property node, not the $def; it must survive deref.
    expect(deref({ $ref: "#/$defs/Color", default: "red" }, root)).toEqual({
      enum: ["red", "green"],
      default: "red",
    });
  });

  it("returns the node unchanged for an unknown $ref (no $def target)", () => {
    // ACTUAL current behaviour: with no target the loop breaks and the node is
    // returned still carrying its $ref (it is NOT collapsed to {}).
    const node = { $ref: "#/$defs/DoesNotExist" };
    expect(deref(node, { $defs: {} })).toEqual({ $ref: "#/$defs/DoesNotExist" });
  });

  it("does not follow a non-$defs ref shape", () => {
    const node = { $ref: "#/components/schemas/X" };
    expect(deref(node, { $defs: {} })).toEqual({ $ref: "#/components/schemas/X" });
  });

  it("returns {} for a nullish node", () => {
    expect(deref(null, {})).toEqual({});
  });
});

describe("resolve", () => {
  it("resolves a $ref into the target object schema", () => {
    const root = { $defs: { Foo: { type: "object", properties: { a: { type: "string" } } } } };
    expect(resolve({ $ref: "#/$defs/Foo" }, root)).toEqual({
      type: "object",
      properties: { a: { type: "string" } },
    });
  });

  it("collapses Optional anyOf:[X,{type:null}] to the non-null branch and keeps `default`", () => {
    const node = {
      anyOf: [{ type: "integer", minimum: 1, maximum: 5 }, { type: "null" }],
      default: 3,
    };
    const r = resolve(node, { $defs: {} });
    expect(r.type).toBe("integer");
    expect(r.minimum).toBe(1);
    expect(r.maximum).toBe(5);
    expect(r.default).toBe(3);
    // The anyOf wrapper itself must be gone after collapsing.
    expect(r.anyOf).toBeUndefined();
  });

  it("derefs a $ref nested inside an anyOf branch", () => {
    const root = {
      $defs: { Mode: { type: "string", enum: ["a", "b"] } },
    };
    const node = { anyOf: [{ $ref: "#/$defs/Mode" }, { type: "null" }] };
    const r = resolve(node, root);
    expect(r.type).toBe("string");
    expect(r.enum).toEqual(["a", "b"]);
  });

  it("terminates on a cyclic $ref at the 10-iteration guard (no infinite loop)", () => {
    const root = { $defs: {} };
    root.$defs.A = { $ref: "#/$defs/B" };
    root.$defs.B = { $ref: "#/$defs/A" };
    // Must return (the guard caps iterations); the exact node still carries a $ref.
    const r = resolve({ $ref: "#/$defs/A" }, root);
    expect(r).toBeTruthy();
    expect(r.$ref).toMatch(/^#\/\$defs\/[AB]$/);
  });
});

describe("enumOf", () => {
  it("returns a top-level enum array", () => {
    expect(enumOf({ enum: ["x", "y"] }, { $defs: {} })).toEqual(["x", "y"]);
  });

  it("finds an enum carried by an anyOf branch", () => {
    const node = { anyOf: [{ enum: ["on", "off"] }, { type: "null" }] };
    expect(enumOf(node, { $defs: {} })).toEqual(["on", "off"]);
  });

  it("returns null when no enum is present anywhere", () => {
    expect(enumOf({ type: "string" }, { $defs: {} })).toBeNull();
  });
});

describe("humanize", () => {
  it("title-cases a snake_case name (min_speech_ms -> 'Min Speech Ms')", () => {
    expect(humanize("min_speech_ms")).toBe("Min Speech Ms");
  });

  it("does not crash on leading / double underscores", () => {
    // Empty segments become empty words; the function must not throw.
    expect(() => humanize("_leading")).not.toThrow();
    expect(humanize("_leading")).toBe(" Leading");
    expect(humanize("double__under")).toBe("Double  Under");
  });
});
