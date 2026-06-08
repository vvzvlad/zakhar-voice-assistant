import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` (5173) proxies relative /api calls to the aiohttp backend on
// :8201, so the panel uses the same relative paths it will use in production
// (where the backend serves the built dist). No CORS dance in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8201",
    },
  },
});
