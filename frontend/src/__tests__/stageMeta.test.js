// Unit tests for the static stage metadata (src/stageMeta.js). Guards the R2
// regression where vad became a catalog category: every stage must map to a
// catalog category id so provider cards/labels resolve from /api/catalog.
import { describe, it, expect } from "vitest";
import { STAGES, STAGE_ORDER, STAGE_COLOR } from "../stageMeta.js";

describe("STAGES", () => {
  it("lists the four pipeline stages in STAGE_ORDER", () => {
    expect(STAGES.map((s) => s.key)).toEqual(STAGE_ORDER);
  });

  it("maps every stage to a catalog category (vad included since R2)", () => {
    for (const s of STAGES) expect(s.cat).toBe(s.key);
  });

  it("has a color for every stage", () => {
    for (const s of STAGES) expect(STAGE_COLOR[s.key]).toBeTruthy();
  });
});
