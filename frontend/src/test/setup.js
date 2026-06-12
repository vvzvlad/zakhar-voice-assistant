// Vitest global setup.
// R-Tool-2: pin the timezone to UTC BEFORE anything reads it, so the date/time
// formatters (format.js fmtStarted, runsModel.js fmtTime) produce deterministic
// output regardless of the host machine's timezone.
process.env.TZ = "UTC";

// jest-dom custom matchers (toBeInTheDocument, toHaveAttribute, ...) for the
// jsdom component tests. Harmless to load in the node-env files too.
import "@testing-library/jest-dom/vitest";
