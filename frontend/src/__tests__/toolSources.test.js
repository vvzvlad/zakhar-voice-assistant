// The menu's Tools counter: the sources switched on, external and built-in.
import { describe, it, expect } from "vitest";
import { enabledToolSources } from "../toolSources.js";

describe("enabledToolSources", () => {
  it("counts the switched-on external servers and built-in sources, a missing flag or section as on", () => {
    expect(enabledToolSources({
      mcp_servers: [{ name: "home" }, { name: "search", enabled: true }, { name: "lab", enabled: false }],
      calendar: { enabled: false },
      reminders: { enabled: true },
      memory: { enabled: false },
    })).toBe(2 + 3);
    expect(enabledToolSources(undefined)).toBe(5);
  });
});
