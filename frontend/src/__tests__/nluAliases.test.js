// Unit tests (node env) for the pure simple-nlu alias helpers.
import { describe, it, expect } from "vitest";
import {
  parseActionNames,
  entitySlotsFromTools,
  parseAliasText,
  serializeAliasText,
} from "../nluAliases.js";

describe("parseActionNames", () => {
  it("returns lowercased, trimmed names before '=' and skips blank/# / no-= lines", () => {
    const text = [
      "on = включи, включай",
      "  OFF =выключи  ",
      "",
      "# comment = ignored",
      "no equals sign here",
    ].join("\n");
    expect(parseActionNames(text)).toEqual(["on", "off"]);
  });

  it("handles empty/undefined input", () => {
    expect(parseActionNames("")).toEqual([]);
    expect(parseActionNames(undefined)).toEqual([]);
  });
});

describe("entitySlotsFromTools", () => {
  const sources = [
    {
      tools: [
        {
          name: "set_light",
          parameters: {
            type: "object",
            properties: {
              device_id: { type: "string", enum: ["bright_room_light", "night_light"] },
              state: { type: "string", enum: ["on", "off"] },
            },
          },
        },
        {
          name: "set_dimmer",
          parameters: {
            type: "object",
            properties: {
              device_id: { type: "string", enum: ["night_light"] },
              brightness: { type: "integer" }, // no enum -> omitted
            },
          },
        },
        {
          name: "set_scene",
          parameters: {
            type: "object",
            properties: { scene: { type: "string", enum: ["night", "morning"] } },
          },
        },
      ],
    },
  ];

  it("excludes state slots ({on,off}) and includes device_id/scene enums", () => {
    const entities = entitySlotsFromTools(sources, ["on", "off"]);
    // state slot {on,off} is excluded; brightness (no enum) is omitted.
    expect(entities).toEqual([
      { tool: "set_light", slot: "device_id", type: "string", values: ["bright_room_light", "night_light"] },
      { tool: "set_dimmer", slot: "device_id", type: "string", values: ["night_light"] },
      { tool: "set_scene", slot: "scene", type: "string", values: ["night", "morning"] },
    ]);
  });

  it("state-slot exclusion is case-insensitive against action names", () => {
    const entities = entitySlotsFromTools(sources, ["ON", "OFF"]);
    expect(entities.some((e) => e.slot === "state")).toBe(false);
  });

  it("keeps the {on,off} slot when those are NOT action names", () => {
    const entities = entitySlotsFromTools(sources, []);
    expect(entities.some((e) => e.tool === "set_light" && e.slot === "state")).toBe(true);
  });
});

describe("parseAliasText", () => {
  it("maps bare 'phrases = value' lines and preserves comments + explicit lines", () => {
    const text = [
      "# header comment",
      "свет в зале, люстра = bright_room_light",
      "",
      "ночь, спать = night",
      "яркость = set_light.brightness:80", // explicit tool.slot:value -> extra
    ].join("\n");
    const { map, extraLines } = parseAliasText(text);
    expect(map).toEqual({
      bright_room_light: "свет в зале, люстра",
      night: "ночь, спать",
    });
    // Comment, blank line, and the explicit form are preserved verbatim.
    expect(extraLines).toEqual([
      "# header comment",
      "",
      "яркость = set_light.brightness:80",
    ]);
  });

  it("treats a stray colon (no dot) as a bare value, but tool.slot:value as explicit", () => {
    // No dot before the colon -> the whole `a:b` is a bare value (server-consistent).
    const bare = parseAliasText("фраза = a:b");
    expect(bare.map["a:b"]).toBe("фраза");
    expect(bare.extraLines).not.toContain("фраза = a:b");
    // Dot before a non-empty value after `:` -> explicit form, preserved verbatim.
    const explicit = parseAliasText("фраза = set_light.entity:bright");
    expect(explicit.extraLines).toContain("фраза = set_light.entity:bright");
    expect(explicit.map["set_light.entity:bright"]).toBeUndefined();
  });

  it("last duplicate value key wins", () => {
    expect(parseAliasText("a = x\nb = x").map.x).toBe("b");
  });
});

describe("serializeAliasText", () => {
  const entities = [
    { tool: "set_light", slot: "device_id", values: ["bright_room_light", "night_light"] },
    { tool: "set_scene", slot: "scene", values: ["night"] },
  ];

  it("emits managed lines in order, skips blank-phrase values, then appends extras", () => {
    const valueToPhrases = {
      bright_room_light: "свет в зале",
      night_light: "   ", // whitespace-only -> skipped
      night: "ночь",
    };
    const extraLines = ["# keep me", "яркость = set_light.brightness:80"];
    const out = serializeAliasText(entities, valueToPhrases, extraLines);
    expect(out).toBe(
      [
        "свет в зале = bright_room_light",
        "ночь = night",
        "# keep me",
        "яркость = set_light.brightness:80",
      ].join("\n")
    );
  });

  it("round-trips through parse + serialize (managed lines + preserved extras)", () => {
    const original = [
      "# header",
      "свет в зале = bright_room_light",
      "ночь = night",
      "яркость = set_light.brightness:80",
    ].join("\n");
    const { map, extraLines } = parseAliasText(original);
    const out = serializeAliasText(entities, map, extraLines);
    // Managed lines first (discovered order), then preserved extras.
    expect(out).toBe(
      [
        "свет в зале = bright_room_light",
        "ночь = night",
        "# header",
        "яркость = set_light.brightness:80",
      ].join("\n")
    );
  });

  it("preserves a managed bare alias whose value is NOT in the discovered catalog", () => {
    // MCP offline at load -> no entities -> the value must still survive.
    const out = serializeAliasText([], { unknown_id: "свет" }, []);
    expect(out).toContain("свет = unknown_id");
  });

  it("parse->serialize keeps an unknown bare value and its preserved extras", () => {
    const { map, extraLines } = parseAliasText("свет = unknown_id\n# note");
    // entities WITHOUT unknown_id: it isn't in the discovered catalog.
    const out = serializeAliasText(entities, map, extraLines);
    expect(out).toContain("свет = unknown_id");
    expect(out).toContain("# note");
  });
});
