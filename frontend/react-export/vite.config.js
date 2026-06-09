import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` (5173) proxies relative /api calls to the aiohttp backend on
// :8201, so the panel uses the same relative paths it will use in production
// (where the backend serves the built dist). No CORS dance in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8201", ws: true },
    },
  },
  // Vitest config. Default to the fast `node` environment; the two component
  // tests opt into jsdom per-file via a `// @vitest-environment jsdom` docblock.
  // The setup file loads jest-dom matchers and pins TZ=UTC (R-Tool-2) so the
  // date/time formatters render deterministically across machines.
  test: {
    environment: "node",
    globals: true,
    setupFiles: "./src/test/setup.js",
  },
});
