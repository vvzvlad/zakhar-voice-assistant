// @vitest-environment jsdom
// Render-level tests for the Character Prompt, System Prompt, Tools and Devices pages:
// the Character Prompt's profiles column lists every profile with its voice (named
// through the options once per provider field, a long bare id cut short) and size, the
// active one marked, and a click on a row opens it; the selected profile's voice is
// flagged when it is not the voice speaking now; Make active applies the voice only with
// the "with voice" box checked, which a profile without a marker lacks; the active
// profile has neither Make active nor Delete in its ⋯ menu, which closes on Escape and
// when focus leaves it; the prompt syntax reference is one collapsed disclosure, each page saves
// with one Save (the Tools page's server edits included), typing in the System Prompt
// editor keeps the page's scroll position, deleting an MCP server
// asks first, the built-in sources start folded and say "built-in" once, the Tools
// page's Memory card edits the saved facts under the same Save, its Reminders card lists
// the scheduled rows and edits and deletes them under the same Save, a
// switched-off server keeps its controls unfaded and a switched-off speaker fades.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within, act } from "@testing-library/react";
import { MCP } from "../pages/tools.jsx";
import { Prompt, SystemPrompt } from "../pages/prompt.jsx";
import { Devices } from "../pages/devices.jsx";
import { useAppData } from "../appData.jsx";
import { getOptions, getPromptProfiles, getPromptProfile, updatePromptProfile, activatePromptProfile, getTools, getMemory, putMemory, getReminders, patchReminder, deleteReminder } from "../api.js";
import { faded } from "./faded.js";

vi.mock("../appData.jsx", () => ({ useAppData: vi.fn() }));
vi.mock("../api.js", () => ({
  getPromptProfiles: vi.fn(async () => ({ profiles: [{ id: 1, name: "default", chars: 2, voice: null }], active_id: 1 })),
  getPromptProfile: vi.fn(),
  updatePromptProfile: vi.fn(async () => ({})),
  activatePromptProfile: vi.fn(async () => ({ ok: true })),
  getOptions: vi.fn(async () => ({ options: [{ value: "bbbb", label: "Boris" }] })),
  getTools: vi.fn(async () => ({ sources: [] })),
  getMemory: vi.fn(async () => ({ facts: [] })),
  putMemory: vi.fn(async (facts) => ({ facts })),
  getReminders: vi.fn(async () => ({ items: [] })),
  patchReminder: vi.fn(async () => ({})),
  deleteReminder: vi.fn(async () => ({ ok: true })),
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

const fishVoice = (id) => ({ provider: "fishaudio", fields: { reference_id: id, model: "s2-pro" } });
const marker = (id) => `<<<<<VOICE provider=fishaudio reference_id=${id} model=s2-pro>>>>>`;

describe("Prompt — profiles", () => {
  // "default" is active and its voice is another fishaudio voice than the current one,
  // "work" names a voice the options do not know, "plain" has no voice marker.
  const WORK_ID = "cccccccc11112222";
  const TEXTS = { 1: `hi ${marker("bbbb")}`, 2: `work ${marker(WORK_ID)}`, 3: "plain" };
  const withProfiles = (defaultVoice = "bbbb") => {
    getPromptProfiles.mockResolvedValue({ active_id: 1, profiles: [
      { id: 1, name: "default", chars: 5, voice: fishVoice(defaultVoice) },
      { id: 2, name: "work", chars: 7, voice: fishVoice(WORK_ID) },
      { id: 3, name: "plain", chars: 5, voice: null },
    ] });
    getPromptProfile.mockImplementation(async (id) => ({ id, name: "p" + id, text: id === 1 ? `hi ${marker(defaultVoice)}` : TEXTS[id] }));
  };
  const row = (name) => within(screen.getByText("Profiles").closest(".z-card")).getByRole("button", { name: new RegExp(`^${name}`) });
  afterEach(() => getPromptProfiles.mockRestore());

  it("lists every profile with its voice and size, the active one marked; voices are looked up once per provider field", async () => {
    useAppData.mockReturnValue(appData());
    withProfiles();
    render(<Prompt />);
    await waitFor(() => expect(within(row("default")).getByText("Fish Audio · Boris · 5 chars")).toBeInTheDocument());
    expect(within(row("default")).getByText("Active")).toBeInTheDocument();
    expect(within(row("work")).getByText("Fish Audio · cccccccc… · 7 chars")).toBeInTheDocument();
    expect(row("work")).toHaveAttribute("title", WORK_ID);
    expect(within(row("work")).queryByText("Active")).toBeNull();
    expect(within(row("plain")).getByText("5 chars")).toBeInTheDocument();
    expect(getOptions).toHaveBeenCalledTimes(1);
    expect(getOptions).toHaveBeenCalledWith("tts", "fishaudio", "reference_id");
    expect(screen.getByText("differs from current TTS voice")).toBeInTheDocument();
  });

  it("names the current voice from the TTS settings without a lookup", async () => {
    useAppData.mockReturnValue(appData());
    withProfiles("aaaa");
    getPromptProfiles.mockResolvedValue({ active_id: 1, profiles: [{ id: 1, name: "default", chars: 5, voice: fishVoice("aaaa") }] });
    render(<Prompt />);
    await waitFor(() => expect(within(row("default")).getByText("Fish Audio · Anna · 5 chars")).toBeInTheDocument());
    expect(getOptions).not.toHaveBeenCalled();
    expect(screen.queryByText("differs from current TTS voice")).not.toBeInTheDocument();
  });

  it("a click on a row opens that profile; one without a voice marker offers Make active without the box", async () => {
    useAppData.mockReturnValue(appData());
    withProfiles();
    render(<Prompt />);
    await screen.findByDisplayValue(TEXTS[1]);
    fireEvent.click(row("plain"));
    expect(await screen.findByDisplayValue("plain")).toBeInTheDocument();
    expect(getPromptProfile).toHaveBeenLastCalledWith(3);
    expect(row("plain")).toHaveAttribute("aria-current", "true");
    expect(row("default")).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("button", { name: "Make active" })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("Make active applies the voice with the box checked and not with it unchecked", async () => {
    useAppData.mockReturnValue(appData());
    withProfiles();
    render(<Prompt />);
    await screen.findByDisplayValue(TEXTS[1]);
    fireEvent.click(row("work"));
    await screen.findByDisplayValue(TEXTS[2]);
    const box = screen.getByRole("checkbox", { name: "with voice" });
    expect(box.checked).toBe(true);
    const activate = () => screen.getByRole("button", { name: "Make active" });
    fireEvent.click(activate());
    await waitFor(() => expect(activatePromptProfile).toHaveBeenCalledWith(2, { applyVoice: true }));
    await waitFor(() => expect(activate()).not.toBeDisabled());
    fireEvent.click(box);
    fireEvent.click(activate());
    await waitFor(() => expect(activatePromptProfile).toHaveBeenLastCalledWith(2, { applyVoice: false }));
  });

  it("the active profile has no Make active and no Delete in its ⋯ menu; another profile's menu has Delete", async () => {
    useAppData.mockReturnValue(appData());
    withProfiles();
    render(<Prompt />);
    await screen.findByDisplayValue(TEXTS[1]);
    expect(screen.queryByRole("button", { name: "Make active" })).toBeNull();
    expect(screen.queryByRole("checkbox")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    expect(screen.getByRole("menuitem", { name: "Rename…" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Delete" })).toBeNull();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("menu")).toBeNull();
    const more = screen.getByRole("button", { name: "More actions" });
    expect(document.activeElement).toBe(more);
    fireEvent.click(more);
    act(() => row("plain").focus());
    expect(screen.queryByRole("menu")).toBeNull();
    fireEvent.click(row("work"));
    await screen.findByDisplayValue(TEXTS[2]);
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    expect(screen.getByRole("menuitem", { name: "Delete" })).toBeInTheDocument();
  });
});

describe("Prompt — syntax reference", () => {
  it("is one collapsed Syntax disclosure holding the four references", async () => {
    useAppData.mockReturnValue(appData());
    getPromptProfile.mockResolvedValue({ id: 1, name: "default", text: "hi" });
    render(<Prompt />);
    const summary = await screen.findByText("Syntax");
    expect(summary.tagName).toBe("SUMMARY");
    const details = summary.closest("details");
    expect(details.open).toBe(false);
    for (const t of ["The date/time variable", "The wake-word variable", "The memory variable", "Preferred voice"]) expect(within(details).getByText(t)).toBeInTheDocument();
    expect(screen.queryByText("Notation")).toBeNull();
    expect(screen.queryByText("Dialog context")).toBeNull();
  });

  it("on the System Prompt page holds the three variables, not the profile's voice marker", async () => {
    useAppData.mockReturnValue(appData());
    render(<SystemPrompt />);
    const details = (await screen.findByText("Syntax")).closest("details");
    expect(details.open).toBe(false);
    for (const t of ["The date/time variable", "The wake-word variable", "The memory variable"]) expect(within(details).getByText(t)).toBeInTheDocument();
    expect(screen.queryByText("Preferred voice")).toBeNull();
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

  it("System Prompt: the save bar sends the edited text to the config", async () => {
    const data = appData({ prompt: { system: "rules" } });
    // The real patch() refreshes the catalog with the stored value.
    data.patch.mockImplementation(async (p) => { data.catalog.core.values.prompt.system = p.core.prompt.system; });
    useAppData.mockReturnValue(data);
    render(<SystemPrompt />);
    fireEvent.change(await screen.findByDisplayValue("rules"), { target: { value: "new rules" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(data.patch).toHaveBeenCalledWith({ core: { prompt: { system: "new rules" } } }));
    expect(updatePromptProfile).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByTestId("save-bar")).toBeNull());
    expect(screen.getByDisplayValue("new rules")).toBeInTheDocument();
  });

  it("System Prompt: typing keeps the page's scroll while the editor regrows", async () => {
    useAppData.mockReturnValue(appData({ prompt: { system: "rules" } }));
    const { container } = render(<div className="z-scroll"><SystemPrompt /></div>);
    const ta = await screen.findByDisplayValue("rules");
    const scroller = container.querySelector(".z-scroll");
    scroller.scrollTop = 689;
    // jsdom has no layout. The browser lays the page out on this read: with the editor at
    // "auto" the page is shorter and the scroller clamps its position.
    Object.defineProperty(ta, "offsetHeight", { configurable: true, get() {
      if (ta.style.height === "auto") scroller.scrollTop = Math.min(scroller.scrollTop, 113);
      return 0;
    } });
    fireEvent.change(ta, { target: { value: "rules!" } });
    expect(scroller.scrollTop).toBe(689);
  });

  it("System Prompt: Reload drops an unsaved edit back to the stored text", async () => {
    const data = appData({ prompt: { system: "rules" } });
    useAppData.mockReturnValue(data);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<SystemPrompt />);
    fireEvent.change(await screen.findByDisplayValue("rules"), { target: { value: "new rules" } });
    fireEvent.click(screen.getByRole("button", { name: "Reload" }));
    expect(confirm).toHaveBeenCalledWith("Discard unsaved changes?");
    expect(await screen.findByDisplayValue("rules")).toBeInTheDocument();
    expect(data.reload).toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByTestId("save-bar")).toBeNull());
    expect(data.patch).not.toHaveBeenCalled();
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

describe("Tools — memory", () => {
  // The core schema's memory section: its Enabled switch.
  const withMemorySchema = (data) => {
    data.catalog.core.schema.properties.memory = { type: "object", properties: { enabled: { type: "boolean", title: "Enabled" } } };
    return data;
  };

  it("its card holds the Enabled switch, the saved facts one per line and the source's tools", async () => {
    useAppData.mockReturnValue(withMemorySchema(appData()));
    getTools.mockResolvedValueOnce({ sources: [{ id: "memory", kind: "builtin", online: true, tools: [{ name: "remember" }, { name: "forget" }] }] });
    getMemory.mockResolvedValueOnce({ facts: ["Кота зовут Барсик", "Пользователь — мужчина"] });
    render(<MCP />);
    const box = await screen.findByLabelText("Memory facts");
    expect(box.value).toBe("Кота зовут Барсик\nПользователь — мужчина");
    const card = box.closest("details");
    expect(within(card).getByText("Memory")).toBeInTheDocument();
    expect(within(card).getByRole("switch")).toBeInTheDocument();
    expect(within(card).getByText("Online")).toBeInTheDocument();
    expect(within(card).getByText("remember")).toBeInTheDocument();
  });

  it("Save sends the lines without a config patch and shows the list as stored", async () => {
    const data = appData();
    useAppData.mockReturnValue(data);
    getMemory.mockResolvedValueOnce({ facts: ["a"] }).mockResolvedValueOnce({ facts: ["a"] });
    putMemory.mockResolvedValueOnce({ facts: ["a", "c"] });
    render(<MCP />);
    fireEvent.change(await screen.findByLabelText("Memory facts"), { target: { value: "a\n\n- c" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.queryByTestId("save-bar")).toBeNull());
    expect(putMemory).toHaveBeenCalledWith(["a", "", "- c"]);
    expect(data.patch).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Memory facts").value).toBe("a\nc");
  });

  it("Save sends the switch to the config too, and a failed memory save stays in the save bar", async () => {
    const data = withMemorySchema(appData());
    useAppData.mockReturnValue(data);
    putMemory.mockRejectedValueOnce(new Error("disk full"));
    render(<MCP />);
    const box = await screen.findByLabelText("Memory facts");
    fireEvent.change(box, { target: { value: "a" } });
    fireEvent.click(within(box.closest("details")).getByRole("switch"));
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(within(screen.getByTestId("save-bar")).getByText(/disk full/)).toBeInTheDocument());
    expect(data.patch).toHaveBeenCalledWith({ core: { memory: { enabled: false } } });
    expect(putMemory).toHaveBeenCalledWith(["a"]);
    expect(screen.getByLabelText("Memory facts").value).toBe("a");
  });

  it("a config-only Save leaves the saved facts alone", async () => {
    const data = withMemorySchema(appData());
    useAppData.mockReturnValue(data);
    getMemory.mockResolvedValueOnce({ facts: ["a"] });
    render(<MCP />);
    const box = await screen.findByLabelText("Memory facts");
    fireEvent.click(within(box.closest("details")).getByRole("switch"));
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(data.patch).toHaveBeenCalled());
    expect(putMemory).not.toHaveBeenCalled();
  });

  it("Save applies the owner's edit to the facts as they are now, keeping what the assistant saved meanwhile", async () => {
    useAppData.mockReturnValue(appData());
    getMemory.mockResolvedValueOnce({ facts: ["a", "b"] }).mockResolvedValueOnce({ facts: ["a", "b", "c"] });
    render(<MCP />);
    fireEvent.change(await screen.findByLabelText("Memory facts"), { target: { value: "a\nd" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(putMemory).toHaveBeenCalledWith(["a", "c", "d"]));
  });
});

describe("Tools — reminders", () => {
  // Epoch seconds of a local date-time.
  const ts = (...a) => new Date(...a).getTime() / 1000;
  const row = (id, kind, text, due_ts, extra = {}) => ({ id, kind, text, due_ts, days: null, device: "kitchen", created_ts: 0, ...extra });
  const rows = async () => {
    const list = await screen.findAllByTestId("scheduled-row");
    return list.map((r) => within(r));
  };

  it("lists each row with its kind, speaker, text and due time, a timer's time left and an alarm's days", async () => {
    useAppData.mockReturnValue(appData({ devices: [{ name: "kitchen", label: "Kitchen" }] }));
    getReminders.mockResolvedValueOnce({ items: [
      row(1, "reminder", "купить молоко", ts(2030, 0, 15, 7, 30, 15)),
      row(2, "timer", "пицца", Date.now() / 1000 + 270.4, { device: "bedroom" }),
      row(3, "alarm", null, ts(2030, 0, 16, 7, 0, 0), { days: [1, 5] }),
      row(4, "command", "выключи свет", ts(2030, 0, 17, 22, 0, 0)),
    ] });
    render(<MCP />);
    const [reminder, timer, alarm, command] = await rows();
    expect(reminder.getByText("Reminder")).toBeInTheDocument();
    expect(reminder.getByText("Kitchen")).toBeInTheDocument();
    expect(reminder.getByLabelText("Text").value).toBe("купить молоко");
    // jsdom normalizes the input's value ("…T07:30:15.000"), so compare the time it holds.
    expect(new Date(reminder.getByLabelText("Due time").value).getTime() / 1000).toBe(ts(2030, 0, 15, 7, 30, 15));
    expect(timer.getByText("bedroom")).toBeInTheDocument();
    expect(timer.getByText("4m 30s left at load")).toBeInTheDocument();
    expect(alarm.getByText("Alarm")).toBeInTheDocument();
    expect(alarm.getAllByRole("button", { pressed: true }).map((b) => b.textContent)).toEqual(["Mon", "Fri"]);
    expect(alarm.getByText("Repeats on these days")).toBeInTheDocument();
    expect(command.getByText("Command")).toBeInTheDocument();
    expect(command.queryByRole("button", { name: "Mon" })).toBeNull();
    expect(screen.getByText("Reminders").closest("details")).toContainElement(screen.getAllByTestId("scheduled-row")[0]);
  });

  it("Save sends a PATCH per edited row and a DELETE per deleted one, then reloads the list", async () => {
    const data = appData();
    useAppData.mockReturnValue(data);
    getReminders.mockResolvedValueOnce({ items: [
      row(1, "reminder", "a", ts(2030, 0, 15, 7, 30, 15)),
      row(2, "alarm", null, ts(2030, 0, 16, 7, 0, 0), { days: [1] }),
      row(3, "command", "выключи свет", ts(2030, 0, 17, 22, 0, 0)),
      row(4, "timer", null, ts(2030, 0, 18, 9, 0, 0)),
    ] });
    render(<MCP />);
    const [reminder, alarm, command] = await rows();
    fireEvent.change(reminder.getByLabelText("Text"), { target: { value: "b" } });
    fireEvent.change(reminder.getByLabelText("Due time"), { target: { value: "2030-01-16T08:00:05" } });
    fireEvent.click(alarm.getByRole("button", { name: "Mon" }));
    expect(alarm.getByText("Rings once")).toBeInTheDocument();
    fireEvent.click(command.getByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.queryByTestId("save-bar")).toBeNull());
    expect(patchReminder.mock.calls).toEqual([[1, { text: "b", due_ts: ts(2030, 0, 16, 8, 0, 5) }], [2, { days: null }]]);
    expect(deleteReminder.mock.calls).toEqual([[3]]);
    expect(getReminders).toHaveBeenCalledTimes(2);
    expect(data.patch).not.toHaveBeenCalled();
  });

  it("an edit put back and a Delete undone leave the page clean; an empty list says so", async () => {
    useAppData.mockReturnValue(appData());
    getReminders.mockResolvedValueOnce({ items: [row(1, "reminder", "a", ts(2030, 0, 15, 7, 30, 15))] });
    render(<MCP />);
    const [reminder] = await rows();
    fireEvent.change(reminder.getByLabelText("Text"), { target: { value: "b" } });
    fireEvent.click(reminder.getByRole("button", { name: "Delete" }));
    expect(screen.getByTestId("save-bar")).toBeInTheDocument();
    fireEvent.change(reminder.getByLabelText("Text"), { target: { value: "a" } });
    fireEvent.click(reminder.getByRole("button", { name: "Undo" }));
    expect(screen.queryByTestId("save-bar")).toBeNull();
    cleanup();
    render(<MCP />);
    expect(await screen.findByText("Nothing is scheduled.")).toBeInTheDocument();
  });
});

describe("a switched-off item fades its content, not its buttons", () => {
  it("speaker (its Enabled switch stays unfaded; a click elsewhere on the row opens its dialog)", () => {
    useAppData.mockReturnValue(appData({ devices: [{ name: "kitchen", enabled: false }] }));
    render(<Devices />);
    expect(faded(screen.getByText("kitchen"))).toBe(true);
    expect(faded(screen.getByRole("switch"))).toBe(false);
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
