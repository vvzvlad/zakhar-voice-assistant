// @vitest-environment jsdom
// Unit tests for playAudioResponse — BLOB FALLBACK path only. jsdom has no
// MediaSource, so the MSE branch is unreachable here (canStreamMse is false) and
// every Response routes through resp.blob() + new Audio(objectURL), exactly the
// pre-streaming behaviour. The MSE path is browser-integration code and is left
// out of unit tests by design.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { playAudioResponse } from "../streamAudio.js";

// Audio stub: records instances, lets play() resolve immediately, and exposes
// pause() as a spy plus manual event firing for ended/error.
class FakeAudio {
  constructor() {
    this.listeners = {};
    this.pause = vi.fn();
    FakeAudio.instances.push(this);
  }
  addEventListener(ev, cb) { this.listeners[ev] = cb; }
  play() { return FakeAudio.playImpl(); }
}

// A Response whose Content-Type is audio/wav (piper) so MSE is never even probed,
// and whose blob() resolves with a stub. No `body` is provided, which also forces
// the fallback (canStreamMse needs resp.body).
function wavResponse(blob = { wav: true }) {
  return {
    headers: { get: () => "audio/wav" },
    blob: vi.fn(async () => blob),
  };
}

beforeEach(() => {
  FakeAudio.instances = [];
  FakeAudio.playImpl = () => Promise.resolve();
  vi.stubGlobal("Audio", FakeAudio);
  URL.createObjectURL = vi.fn(() => "blob:wav-clip");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("playAudioResponse — blob fallback", () => {
  it("buffers the body, plays it and fires onPlaying", async () => {
    const onPlaying = vi.fn();
    const resp = wavResponse();
    playAudioResponse(resp, { onPlaying });

    // blob() + createObjectURL + Audio happen on the microtask after blob resolves.
    await vi.waitFor(() => expect(onPlaying).toHaveBeenCalledTimes(1));
    expect(resp.blob).toHaveBeenCalledTimes(1);
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(FakeAudio.instances).toHaveLength(1);
  });

  it("fires onEnded and revokes the object URL when the clip ends", async () => {
    const onEnded = vi.fn();
    playAudioResponse(wavResponse(), { onEnded });
    await vi.waitFor(() => expect(FakeAudio.instances).toHaveLength(1));

    FakeAudio.instances[0].listeners.ended();
    expect(onEnded).toHaveBeenCalledTimes(1);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:wav-clip");
  });

  it("fires onError on an audio 'error' event and frees the URL", async () => {
    const onError = vi.fn();
    playAudioResponse(wavResponse(), { onError });
    await vi.waitFor(() => expect(FakeAudio.instances).toHaveLength(1));

    FakeAudio.instances[0].listeners.error();
    expect(onError).toHaveBeenCalledWith("Audio playback failed");
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:wav-clip");
  });

  it("fires onError when play() rejects", async () => {
    FakeAudio.playImpl = () => Promise.reject(new Error("blocked"));
    const onPlaying = vi.fn();
    const onError = vi.fn();
    playAudioResponse(wavResponse(), { onPlaying, onError });

    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith("Audio playback failed"));
    expect(onPlaying).not.toHaveBeenCalled();
  });

  it("stop() pauses the audio, revokes the URL and suppresses later callbacks", async () => {
    const onEnded = vi.fn();
    const onError = vi.fn();
    const ctl = playAudioResponse(wavResponse(), { onEnded, onError });
    await vi.waitFor(() => expect(FakeAudio.instances).toHaveLength(1));

    ctl.stop();
    expect(FakeAudio.instances[0].pause).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:wav-clip");

    // A late terminal event after stop() must not call back through.
    FakeAudio.instances[0].listeners.ended();
    FakeAudio.instances[0].listeners.error();
    expect(onEnded).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it("stop() before the blob resolves cancels playback (no Audio is created)", async () => {
    const onPlaying = vi.fn();
    // A blob() that never resolves during this tick: stop() lands first.
    let resolveBlob;
    const resp = {
      headers: { get: () => "audio/wav" },
      blob: vi.fn(() => new Promise((r) => { resolveBlob = r; })),
    };
    const ctl = playAudioResponse(resp, { onPlaying });
    ctl.stop();
    resolveBlob({});
    await Promise.resolve();
    await Promise.resolve();
    expect(FakeAudio.instances).toHaveLength(0);
    expect(onPlaying).not.toHaveBeenCalled();
  });
});
