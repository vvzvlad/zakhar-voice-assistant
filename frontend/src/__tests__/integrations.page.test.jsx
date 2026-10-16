// @vitest-environment jsdom
// Render-level tests for the Prompt, Tools and Devices pages: the profile's
// preferred voice is shown by name and flagged when it is not the voice speaking
// now, each page saves with one Save, and deleting an MCP server or a speaker
// asks first.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MCP, Prompt, Devices } from "../pages/integrations.jsx";
import { useAppData } from "../appData.jsx";
import { getOptions, getPromptProfile, updatePromptProfile } from "../api.js";

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

function appData(core = {}) {
  const patch = vi.fn(async () => {});
  return {
    patch,
    reload: vi.fn(),
    catalog: {
      core: { schema: { $defs: {}, properties: {} }, values: core },
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
    await waitFor(() => expect(screen.getByTestId("marker-voice").textContent).toBe("Apply voice: Fish Audio · Boris"));
    expect(getOptions).toHaveBeenCalledWith("tts", "fishaudio", "reference_id");
    expect(screen.getByText("differs from current TTS voice")).toBeInTheDocument();
  });

  it("names the current voice from the TTS settings without a lookup", async () => {
    useAppData.mockReturnValue(appData());
    withMarker("<<<<<VOICE provider=fishaudio reference_id=aaaa model=s2-pro>>>>>");
    render(<Prompt />);
    expect((await screen.findByTestId("marker-voice")).textContent).toBe("Apply voice: Fish Audio · Anna");
    expect(getOptions).not.toHaveBeenCalled();
    expect(screen.queryByText("differs from current TTS voice")).not.toBeInTheDocument();
  });
});

describe("one Save per page", () => {
  it("Tools: edits in several source cards go out in one patch", async () => {
    const data = appData();
    const field = (name) => ({ type: "object", properties: { [name]: { type: "string", title: name } } });
    data.catalog.core.schema.properties = { openweathermap: field("city"), calendar: field("url") };
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
    expect(data.patch).not.toHaveBeenCalled();
    fireEvent.click(screen.getAllByText("Delete")[0]);
    await waitFor(() => expect(data.patch).toHaveBeenCalledWith({ core: { mcp_servers: [{ name: "lab", url: "http://l/mcp" }] } }));
  });

  it("deletes a speaker only after confirmation", async () => {
    const data = appData({ devices: [{ name: "kitchen" }] });
    useAppData.mockReturnValue(data);
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    render(<Devices />);
    fireEvent.click(screen.getByText("Delete"));
    expect(confirm).toHaveBeenLastCalledWith('Delete speaker "kitchen"?');
    expect(data.patch).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Delete"));
    await waitFor(() => expect(data.patch).toHaveBeenCalledWith({ core: { devices: [] } }));
  });
});
