// Unit tests (node env) for the pure simple-nlu alias helpers.
import { describe, it, expect } from "vitest";
import {
  parseActionNames,
  parseActions,
  entitySlotsFromTools,
  allEnumSlots,
  enumSlotsWithSource,
  numberSlotsFromTools,
  classifySlotKind,
  parseAliasText,
  serializeAliasText,
  serializeActions,
  splitWords,
  joinWords,
  groupSlots,
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

describe("parseActions", () => {
  it("maps lowercased name -> verbs verbatim and preserves blank/# / no-= lines", () => {
    const text = [
      "on = включи, включай",
      "  OFF =выключи  ",
      "",
      "# comment line",
      "no equals sign here",
    ].join("\n");
    const { map, extraLines } = parseActions(text);
    // Name is lowercased+trimmed; RHS is trimmed verbatim (case/spacing kept).
    expect(map).toEqual({ on: "включи, включай", off: "выключи" });
    expect(extraLines).toEqual(["", "# comment line", "no equals sign here"]);
  });

  it("last duplicate name key wins", () => {
    expect(parseActions("on = a\non = b").map.on).toBe("b");
  });

  it("handles empty/undefined input", () => {
    expect(parseActions("").map).toEqual({});
    expect(parseActions(undefined).map).toEqual({});
  });
});

describe("serializeActions", () => {
  const actionSlots = [
    { tool: "set_light", slot: "state", values: ["on", "off"] },
    { tool: "set_lock", slot: "action", values: ["lock", "unlock"] },
  ];

  it("emits 'value = verbs' in order, skips blank, appends custom + extras, dedups", () => {
    const nameToVerbs = {
      on: "включи",
      off: "   ", // whitespace-only -> skipped
      lock: "запри",
      unlock: "отопри",
      custom: "сделай", // not in any action slot -> appended after
    };
    const extraLines = ["# note"];
    const out = serializeActions(actionSlots, nameToVerbs, extraLines);
    expect(out).toBe(
      [
        "on = включи",
        "lock = запри",
        "unlock = отопри",
        "custom = сделай",
        "# note",
      ].join("\n")
    );
  });

  it("dedups a value shared across action slots (one line per unique value)", () => {
    const slots = [
      { tool: "a", slot: "state", values: ["on", "off"] },
      { tool: "b", slot: "state", values: ["on", "off"] }, // same values
    ];
    const out = serializeActions(slots, { on: "включи", off: "выключи" }, []);
    expect(out).toBe(["on = включи", "off = выключи"].join("\n"));
  });

  it("round-trips through parseActions + serializeActions", () => {
    const { map, extraLines } = parseActions("on = включи\noff = выключи\n# note");
    const out = serializeActions(actionSlots, map, extraLines);
    expect(out).toBe(["on = включи", "off = выключи", "# note"].join("\n"));
  });
});

describe("allEnumSlots", () => {
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
        { name: "no_params" }, // missing parameters -> tolerated
      ],
    },
  ];

  it("returns EVERY enum slot incl. the state slot, in encounter order", () => {
    expect(allEnumSlots(sources)).toEqual([
      { tool: "set_light", slot: "device_id", type: "string", values: ["bright_room_light", "night_light"] },
      { tool: "set_light", slot: "state", type: "string", values: ["on", "off"] },
      { tool: "set_dimmer", slot: "device_id", type: "string", values: ["night_light"] },
    ]);
  });

  it("tolerates missing/undefined sources", () => {
    expect(allEnumSlots(undefined)).toEqual([]);
    expect(allEnumSlots([])).toEqual([]);
  });
});

describe("classifySlotKind", () => {
  it("classifies a slot named 'state'/'action' as action regardless of values", () => {
    expect(classifySlotKind("state", ["red", "green"], [])).toBe("action");
    expect(classifySlotKind("action", ["lock", "unlock"], [])).toBe("action");
  });

  it("classifies values ⊆ actionNames as action (case-insensitive)", () => {
    expect(classifySlotKind("mystery", ["on", "off"], ["ON", "OFF"])).toBe("action");
  });

  it("classifies device_id / scene as entity", () => {
    expect(classifySlotKind("device_id", ["bright_room_light"], ["on", "off"])).toBe("entity");
    expect(classifySlotKind("scene", ["night", "morning"], ["on", "off"])).toBe("entity");
  });

  it("an explicit override wins over the heuristics", () => {
    // slot named 'state' would be action, but override forces entity.
    expect(classifySlotKind("state", ["on", "off"], ["on", "off"], "entity")).toBe("entity");
    // device_id would be entity, but override forces action.
    expect(classifySlotKind("device_id", ["x"], [], "action")).toBe("action");
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

  it("knownCatalogValues suppresses an in-catalog action value but keeps an out-of-catalog alias", () => {
    // `on` is a real catalog value living in an ACTION slot (its verbs are in
    // `actions`), so its map entry must NOT be re-emitted into aliases. The
    // out-of-catalog `unknown_id` manual alias must still survive.
    const map = { on: "включи", unknown_id: "свет" };
    const known = new Set(["on", "off", "bright_room_light", "night_light", "night"]);
    // entities is the ENTITY slots only (no `on`), so `on` reaches the safety-net loop.
    const out = serializeAliasText(entities, map, [], known);
    expect(out).not.toContain("on");
    expect(out).toContain("свет = unknown_id");
  });
});

describe("splitWords / joinWords", () => {
  it("splits on comma, trims, drops empties and dedups", () => {
    expect(splitWords("свет в зале,  люстра , свет в зале,, ")).toEqual(["свет в зале", "люстра"]);
  });

  it("strips '=' so an input word can't corrupt the `phrases = value` line", () => {
    // The `=` is removed and the surrounding words are joined by a single space,
    // NOT split into two entries (which would mis-bind the value on round-trip).
    expect(splitWords("свет = люстра")).toEqual(["свет люстра"]);
  });

  it("collapses a newline within a word (no comma -> single entry)", () => {
    expect(splitWords("a\nb")).toEqual(["a b"]);
  });

  it("still splits on comma and trims after sanitizing", () => {
    expect(splitWords("ванная,  ванна ")).toEqual(["ванная", "ванна"]);
  });

  it("dedups identical words", () => {
    expect(splitWords("свет, свет")).toEqual(["свет"]);
  });

  it("handles empty/undefined input", () => {
    expect(splitWords("")).toEqual([]);
    expect(splitWords(undefined)).toEqual([]);
    expect(splitWords("   ")).toEqual([]);
  });

  it("joinWords joins with comma-space", () => {
    expect(joinWords(["включи", "зажги"])).toBe("включи, зажги");
    expect(joinWords([])).toBe("");
  });

  it("round-trips split -> join (trimmed, deduped)", () => {
    expect(joinWords(splitWords("a, b ,a , c"))).toBe("a, b, c");
  });
});

describe("enumSlotsWithSource", () => {
  const sources = [
    {
      id: "home",
      tools: [
        {
          name: "set_light",
          parameters: {
            type: "object",
            properties: {
              device_id: { type: "string", enum: ["bright_room_light"] },
              state: { type: "string", enum: ["on", "off"] },
              brightness: { type: "integer" }, // no enum -> omitted
            },
          },
        },
      ],
    },
    {
      id: "garden",
      tools: [
        { name: "set_pump", parameters: { type: "object", properties: { action: { type: "string", enum: ["start", "stop"] } } } },
      ],
    },
  ];

  it("carries the owning source id on every enum slot, in encounter order", () => {
    expect(enumSlotsWithSource(sources)).toEqual([
      { server: "home", tool: "set_light", slot: "device_id", type: "string", values: ["bright_room_light"] },
      { server: "home", tool: "set_light", slot: "state", type: "string", values: ["on", "off"] },
      { server: "garden", tool: "set_pump", slot: "action", type: "string", values: ["start", "stop"] },
    ]);
  });

  it("tolerates missing sources", () => {
    expect(enumSlotsWithSource(undefined)).toEqual([]);
  });
});

describe("numberSlotsFromTools", () => {
  const sources = [
    {
      id: "home",
      tools: [
        {
          name: "set_dimmer",
          parameters: {
            type: "object",
            required: ["device_id", "brightness"],
            properties: {
              device_id: { type: "string", enum: ["night_light"] }, // has enum -> not numeric
              brightness: { type: "integer" }, // required, no enum, integer -> numeric
            },
          },
        },
        {
          name: "set_climate",
          parameters: {
            type: "object",
            required: ["temperature"],
            properties: {
              temperature: { type: "string" }, // required string -> off-capable number
              note: { type: "string" }, // not required -> skipped
            },
          },
        },
      ],
    },
  ];

  it("emits required, enum-less, number/integer/string slots with their source", () => {
    expect(numberSlotsFromTools(sources)).toEqual([
      { server: "home", tool: "set_dimmer", slot: "brightness", type: "integer" },
      { server: "home", tool: "set_climate", slot: "temperature", type: "string" },
    ]);
  });

  it("skips optional (non-required) properties even when enum-less and numeric", () => {
    const out = numberSlotsFromTools([
      {
        id: "x",
        tools: [{ name: "t", parameters: { type: "object", properties: { n: { type: "integer" } } } }],
      },
    ]);
    expect(out).toEqual([]); // n is not in `required`
  });

  it("tolerates missing sources", () => {
    expect(numberSlotsFromTools(undefined)).toEqual([]);
  });
});

describe("groupSlots", () => {
  // Classify by slot name (state/action -> action; otherwise entity).
  const kindOf = (s) => classifySlotKind(s.slot, s.values, []);

  it("merges identical ACTION value-sets across tools (union of tools = used-in)", () => {
    const slots = [
      { server: "home", tool: "set_light", slot: "state", type: "string", values: ["on", "off"] },
      { server: "home", tool: "set_switch", slot: "state", type: "string", values: ["off", "on"] }, // same set, different order
    ];
    const groups = groupSlots(slots, kindOf);
    expect(groups).toHaveLength(1);
    expect(groups[0].kind).toBe("action");
    expect(groups[0].key).toBe("off on"); // sorted(values).join(" ")
    expect(groups[0].tools).toEqual(["set_light", "set_switch"]); // union
    expect(groups[0].values).toEqual(["on", "off"]); // first-seen order preserved
  });

  it("keeps ENTITY slots separate per tool.slot even with identical values", () => {
    const slots = [
      { server: "home", tool: "set_light", slot: "device_id", type: "string", values: ["lamp"] },
      { server: "home", tool: "set_switch", slot: "device_id", type: "string", values: ["lamp"] },
    ];
    const groups = groupSlots(slots, kindOf);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.key)).toEqual(["set_light.device_id", "set_switch.device_id"]);
  });

  it("accumulates servers across merged action groups", () => {
    const slots = [
      { server: "home", tool: "set_light", slot: "state", type: "string", values: ["on", "off"] },
      { server: "garden", tool: "set_pump", slot: "state", type: "string", values: ["on", "off"] },
    ];
    const groups = groupSlots(slots, kindOf);
    expect(groups).toHaveLength(1);
    expect(groups[0].servers).toEqual(["home", "garden"]);
  });
});
