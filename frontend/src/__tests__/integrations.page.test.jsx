// @vitest-environment jsdom
// Render-level tests for the Prompt, Tools and Devices pages: the profile's
// preferred voice is shown by name and flagged when it is not the voice speaking
// now, the prompt syntax reference is one collapsed disclosure, each page saves
// with one Save (the Tools page's server edits included), deleting an MCP server
// asks first, the built-in sources start folded and say "built-in" once, a
// switched-off server keeps its controls unfaded and a switched-off speaker fades.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import { MCP, Prompt, Devices } from "../pages/integrations.jsx";
import { useAppData } from "../appData.jsx";
import { getOptions, getPromptProfile, updatePromptProfile } from "../api.js";
import { faded } from "./faded.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getPromptProfiles: vi.fn(async () => ({ profiles: [{ id: 1, name: "default" }], active_id: 1 })),
  getPromptProfile: vi.fn(),
  updatePromptProfile: vi.fn(async () => ({})),
  getOptions: vi.fn(async () => ({ options: [{ value: "bbbb", label: "Boris" }] })),
  getTools: vi.fn(async () => ({ sources: [] })),
  getConfig: vi.fn(async () => ({ core: { devices: [{ name: "kitchen" }] } })),
}));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

const fish = {
  id: "fishaudio",
  label: "Fish Audio",
  schema: { properties: { reference_id: {}, reference_id_label: { hidden: true }, model: {} } },
  values: { reference_id: "aaaa", reference_id_label: "Anna", model: "s2-pro" },
};

// The core schema's external MCP server entry, as the backend emits it.
const serverSchema = () => ({
  $defs: { McpServerConfig: { type: "object", properties: { enabled: { type: "boolean", title: "Enabled" } } } },
  properties: { mcp_servers: { type: "array", items: { $ref: "#/$defs/McpServerConfig" } } },
});

function appData(core = {}) {
  const patch = vi.fn(async () => {});
  return {
    patch,
    reload: vi.fn(),
    catalog: {
      core: { schema: serverSchema(), values: core },
      categories: [{ id: "tts", variants: [{ provider: "fishaudio", enabled: true }], providers: [fish] }],
    },
  };
}

const withMarker = (marker) => getPromptProfile.mockResolvedValue({ id: 1, name: "default", text: `hi ${marker}` });

describe("Prompt — preferred voice", () => {
  it("names another fishaudio voice through the field's options and flags it as not the current one", async () => {
    useAppData.mockReturnValue(appData());
    withMarker("<<<<<VOICE provider=fishaudio reference_id=bbbb model=s2-pro>>>>>");
    render(<Prompt />);
    await waitFor(() => expect(screen.getByTestId("marker-voice").textContent).toBe("Apply voice on activate: Fish Audio · Boris"));
    expect(getOptions).toHaveBeenCalledWith("tts", "fishaudio", "reference_id");
    expect(screen.getByText("differs from current TTS voice")).toBeInTheDocument();
  });

  it("names the current voice from the TTS settings without a lookup", async () => {
    useAppData.mockReturnValue(appData());
    withMarker("<<<<<VOICE provider=fishaudio reference_id=aaaa model=s2-pro>>>>>");
    render(<Prompt />);
    expect((await screen.findByTestId("marker-voice")).textContent).toBe("Apply voice on activate: Fish Audio · Anna");
    expect(getOptions).not.toHaveBeenCalled();
    expect(screen.queryByText("differs from current TTS voice")).not.toBeInTheDocument();
  });
});

describe("Prompt — syntax reference", () => {
  it("is one collapsed Syntax disclosure holding the three references", async () => {
    useAppData.mockReturnValue(appData());
    getPromptProfile.mockResolvedValue({ id: 1, name: "default", text: "hi" });
    render(<Prompt />);
    const summary = await screen.findByText("Syntax");
    expect(summary.tagName).toBe("SUMMARY");
    const details = summary.closest("details");
    expect(details.open).toBe(false);
    for (const t of ["The date/time variable", "Preferred voice", "Notation"]) expect(within(details).getByText(t)).toBeInTheDocument();
    expect(screen.queryByText("Dialog context")).toBeNull();
  });
});

describe("one Save per page", () => {
  it("Tools: edits in several source cards go out in one patch", async () => {
    const data = appData();
    const field = (name) => ({ type: "object", properties: { [name]: { type: "string", title: name } } });
    Object.assign(data.catalog.core.schema.properties, { openweathermap: field("city"), calendar: field("url") });
    useAppData.mockReturnValue(data);
    render(<MCP />);
    fireEvent.change(await screen.findByDisplayValue("Moscow"), { target: { value: "Kazan" } });
    fireEvent.change(screen.getByText("url").closest(".z-f").querySelector("input"), { target: { value: "https://dav" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch.mock.calls[0][0]).toEqual({ core: {
      openweathermap: { enabled: true, api_key: "", city: "Kazan" },
      calendar: { enabled: true, url: "https://dav", username: "", password: "", calendar: "" },
    } });
  });

  it("Tools: adding and switching off external servers waits for the page's Save, with the built-ins' edits", async () => {
    const data = appData({ mcp_servers: [{ name: "home", url: "http://h/mcp" }] });
    data.catalog.core.schema.properties.openweathermap = { type: "object", properties: { city: { type: "string", title: "city" } } };
    useAppData.mockReturnValue(data);
    render(<MCP />);
    const home = (await screen.findByText("http://h/mcp")).closest(".z-card");
    fireEvent.click(within(home).getByRole("switch"));
    fireEvent.click(screen.getByRole("button", { name: "Add server" }));
    fireEvent.change(screen.getByPlaceholderText("e.g. home"), { target: { value: "lab" } });
    fireEvent.change(screen.getByPlaceholderText("http://10.0.0.5:8123/mcp_server/sse"), { target: { value: "http://l/mcp" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    fireEvent.change(screen.getByDisplayValue("Moscow"), { target: { value: "Kazan" } });
    expect(within(screen.getByText("lab").closest(".z-card")).getByText("not saved")).toBeInTheDocument();
    expect(data.patch).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(data.patch).toHaveBeenCalledTimes(1));
    expect(data.patch.mock.calls[0][0]).toEqual({ core: {
      mcp_servers: [
        { name: "home", url: "http://h/mcp", enabled: false },
        { name: "lab", url: "http://l/mcp", token: "", transport: "auto", prompt: "", enabled: true, slow: false },
      ],
      openweathermap: { enabled: true, api_key: "", city: "Kazan" },
    } });
  });

  it("Tools: Edit → Done leaves the page clean, and a changed URL goes out on Save", async () => {
    const data = appData({ mcp_servers: [{ name: "home", url: "http://h/mcp", slow: true }] });
    useAppData.mockReturnValue(data);
    render(<MCP />);
    fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.queryByTestId("save-bar")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByDisplayValue("http://h/mcp"), { target: { value: "http://h2/mcp" } });
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(data.patch).toHaveBeenCalledWith({ core: { mcp_servers: [{ name: "home", url: "http://h2/mcp", slow: true }] } }));
  });

  it("Prompt: the save bar saves the text, and Reload asks before dropping it", async () => {
    useAppData.mockReturnValue(appData());
    getPromptProfile.mockResolvedValue({ id: 1, name: "default", text: "hi" });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<Prompt />);
    fireEvent.change(await screen.findByDisplayValue("hi"), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Reload" }));
    expect(confirm).toHaveBeenCalledWith("Discard unsaved changes?");
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(updatePromptProfile).toHaveBeenCalledWith(1, { text: "hello" }));
    await waitFor(() => expect(screen.queryByTestId("save-bar")).toBeNull());
    confirm.mockRestore();
  });
});

describe("delete asks first", () => {
  afterEach(() => vi.restoreAllMocks());

  it("deletes an MCP server only after confirmation", async () => {
    const data = appData({ mcp_servers: [{ name: "home", url: "http://h/mcp" }, { name: "lab", url: "http://l/mcp" }] });
    useAppData.mockReturnValue(data);
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    render(<MCP />);
    fireEvent.click((await screen.findAllByText("Delete"))[0]);
    expect(screen.getByText("Tools")).toBeInTheDocument();
    expect(screen.queryByText("external")).not.toBeInTheDocument();
    expect(confirm).toHaveBeenLastCalledWith('Delete MCP server "home"?');
    expect(screen.getByText("home")).toBeInTheDocument();
    fireEvent.click(screen.getAllByText("Delete")[0]);
    expect(screen.queryByText("home")).toBeNull();
    expect(data.patch).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(data.patch).toHaveBeenCalledWith({ core: { mcp_servers: [{ name: "lab", url: "http://l/mcp" }] } }));
  });
});

describe("Tools — built-in sources", () => {
  it("start folded and say built-in once, without a config path", async () => {
    useAppData.mockReturnValue(appData());
    render(<MCP />);
    const head = await screen.findByText("Calendar");
    const card = head.closest("details");
    expect(card.open).toBe(false);
    expect(head.closest("summary")).toBeTruthy();
    expect(within(card).getAllByText(/built-in/i)).toHaveLength(1);
    expect(screen.queryByText(/core\./)).toBeNull();
  });
});

describe("a switched-off item fades its content, not its buttons", () => {
  it("speaker (its row has no buttons: a click opens its dialog)", () => {
    useAppData.mockReturnValue(appData({ devices: [{ name: "kitchen", enabled: false }] }));
    render(<Devices />);
    expect(faded(screen.getByText("kitchen"))).toBe(true);
  });

  it("MCP server", async () => {
    useAppData.mockReturnValue(appData({ mcp_servers: [{ name: "home", url: "http://h/mcp", enabled: false }] }));
    render(<MCP />);
    const card = (await screen.findByText("http://h/mcp")).closest(".z-card");
    expect(faded(within(card).getByText("home"))).toBe(true);
    for (const b of ["Edit", "Delete"]) expect(faded(within(card).getByRole("button", { name: b }))).toBe(false);
    expect(faded(within(card).getByRole("switch"))).toBe(false);
  });
});
