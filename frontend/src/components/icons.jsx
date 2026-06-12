import React from "react";

export const ICONS = {
  dashboard: "M3 3h7v7H3zM14 3h7v5h-7zM14 11h7v10h-7zM3 14h7v7H3z",
  vad: "M12 3v18M8 7v10M16 7v10M4 10v4M20 10v4",
  stt: "M4 7h16M4 12h16M4 17h9",
  llm: "M12 3a4 4 0 014 4 4 4 0 010 8 3.5 3.5 0 01-8 0 4 4 0 010-8 4 4 0 014-4zM12 7v10",
  stress: "M7 17l5-11 5 11M9 13h6M10 3.5l3-1.5",
  tts: "M4 9v6h4l5 4V5L8 9zM17 8a5 5 0 010 8M19.5 5.5a9 9 0 010 13",
  mcp: "M12 2v5M12 17v5M5 12H2M22 12h-3M7 7l-2-2M19 5l-2 2M7 17l-2 2M19 19l-2-2M12 8a4 4 0 100 8 4 4 0 000-8z",
  prompt: "M5 3h11l3 3v15H5zM9 9h7M9 13h7M9 17h4",
  context: "M3 12a9 9 0 109-9 9 9 0 00-7 3.4M3 3v3.4h3.4M12 7v5l3 2",
  devices: "M5 4h14v12H5zM9 20h6M12 16v4M9 8h6M9 11h3",
  log: "M4 5h16M4 10h16M4 15h10M4 20h13",
  network: "M12 2a10 10 0 100 20 10 10 0 000-20zM2 12h20M12 2c3.5 3 3.5 17 0 20M12 2c-3.5 3-3.5 17 0 20",
  system: "M4 7h10M18 7h2M4 17h6M14 17h6M14 5v4M10 15v4",
  restart: "M4 12a8 8 0 112.3 5.6M4 20v-5h5",
  search: "M11 11m-7 0a7 7 0 1014 0 7 7 0 10-14 0M21 21l-4-4",
  chev: "M2 4l3.5 3.5L9 4", play: "M2 1l9 5-9 5z", close: "M4 4l8 8M12 4l-8 8",
  test: "M9 12l2 2 4-4M12 3a9 9 0 100 18 9 9 0 000-18z",
  add: "M12 5v14M5 12h14", trash: "M4 6h16M9 6V4h6v2M6 6l1 14h10l1-14",
  ext: "M7 13l6-6M8 7h5v5", clock: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 7v5l3 2",
};

export function Ic({ n, w = 16 }) {
  return <svg width={w} height={w} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={ICONS[n]} /></svg>;
}
