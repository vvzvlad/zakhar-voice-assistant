// Unit tests for the fetch wrapper (src/api.js). `request` itself is module-local,
// so it is exercised through the exported `getCatalog` GET wrapper; `getRuns` is
// tested directly for its query-string builder. global.fetch is stubbed per test.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ApiError, getCatalog, getRuns } from "../api.js";

// Build a minimal Response-like stub for the request() body-reading path.
function mockResponse({ ok = true, status = 200, text = "" }) {
  return { ok, status, text: async () => text };
}

describe("request (via getCatalog)", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("parses a 2xx JSON body", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({ ok: true, status: 200, text: JSON.stringify({ hello: "world" }) })
    );
    await expect(getCatalog()).resolves.toEqual({ hello: "world" });
  });

  it("returns null for a 204 (empty body)", async () => {
    global.fetch.mockResolvedValue(mockResponse({ ok: true, status: 204, text: "" }));
    await expect(getCatalog()).resolves.toBeNull();
  });

  it("throws ApiError with detail from a non-2xx {detail} body", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({ ok: false, status: 422, text: JSON.stringify({ detail: [{ msg: "bad" }] }) })
    );
    const err = await getCatalog().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.detail).toEqual([{ msg: "bad" }]);
    expect(err.body).toEqual({ detail: [{ msg: "bad" }] });
  });

  it("uses {error} as the message for a non-2xx body that carries it", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({ ok: false, status: 500, text: JSON.stringify({ error: "kaboom" }) })
    );
    const err = await getCatalog().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(500);
    expect(err.message).toBe("kaboom");
    expect(err.body).toEqual({ error: "kaboom" });
  });

  it("throws ApiError for a non-2xx non-JSON body (falls back to HTTP <status>)", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({ ok: false, status: 502, text: "<html>Bad Gateway</html>" })
    );
    const err = await getCatalog().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(502);
    expect(err.message).toBe("HTTP 502");
    // Non-JSON body is preserved verbatim on .body.
    expect(err.body).toBe("<html>Bad Gateway</html>");
  });

  it("wraps a fetch network rejection as ApiError with status 0", async () => {
    global.fetch.mockRejectedValue(new Error("ECONNREFUSED"));
    const err = await getCatalog().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.message).toContain("ECONNREFUSED");
  });
});

describe("getRuns query builder", () => {
  let urls;
  beforeEach(() => {
    urls = [];
    global.fetch = vi.fn((url) => {
      urls.push(url);
      return Promise.resolve(mockResponse({ ok: true, status: 200, text: JSON.stringify({ runs: [] }) }));
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("hits the bare /api/runs path when no params are given", async () => {
    await getRuns();
    expect(urls[0]).toBe("/api/runs");
  });

  it("drops undefined / null / empty-string params but KEEPS 0", async () => {
    await getRuns({ device: undefined, result: null, search: "", limit: 0 });
    // Only limit=0 survives (0 is a valid value, not empty).
    expect(urls[0]).toBe("/api/runs?limit=0");
  });

  it("builds the EXACT URL with device/result/search/limit in fixed order", async () => {
    await getRuns({ device: "kitchen", result: "errors", search: "hello world", limit: 50 });
    // URLSearchParams form-encodes spaces as '+'; insertion order is device,result,search,limit.
    expect(urls[0]).toBe("/api/runs?device=kitchen&result=errors&search=hello+world&limit=50");
  });

  it("encodes special characters (& and spaces) in the search term", async () => {
    await getRuns({ search: "a & b" });
    // '&' -> %26, space -> '+'.
    expect(urls[0]).toBe("/api/runs?search=a+%26+b");
  });
});
