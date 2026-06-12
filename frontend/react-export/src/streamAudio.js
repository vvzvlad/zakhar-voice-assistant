// Browser-media integration for the panel's "Test voice" preview. Given the raw
// Response from /api/tts/test, start playback as early as possible:
//
//   - MSE STREAMING for MP3 (audio/mpeg) on browsers with a usable MediaSource:
//     the response body is piped chunk-by-chunk into a SourceBuffer, so audio
//     begins before synthesis finishes.
//   - BLOB FALLBACK otherwise (piper's audio/wav, or a browser without a usable
//     MediaSource): the whole body is buffered into a Blob and played at once —
//     this reproduces the pre-streaming behaviour.
//
// Framework-agnostic (no React): returns a controller `{ stop }`. Exactly ONE
// terminal callback (onEnded or onError) fires per playback, and stop() suppresses
// both. jsdom has no MediaSource, so only the blob path is unit-testable.

// playAudioResponse(resp, { onPlaying, onEnded, onError }) -> { stop }
export function playAudioResponse(resp, { onPlaying, onEnded, onError } = {}) {
  // Strip any "; charset=..." suffix so MediaSource.isTypeSupported sees a bare
  // MIME (e.g. "audio/mpeg"), which is what it expects.
  const mime = (resp.headers.get("Content-Type") || "").split(";")[0].trim();

  // One-shot terminal guard: ensures a single onEnded/onError, and that stop()
  // silences any later event. Held object URLs are revoked exactly once.
  let done = false;
  let audio = null;
  let reader = null;
  let objectUrl = null; // blob URL or MediaSource URL still awaiting revocation

  const revokeUrl = () => {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
  };

  // Cancel the input stream so the underlying connection is released even when the
  // consumer does not call stop() (e.g. a SourceBuffer error fires onError) or when
  // stop() lands before the reader exists. Once getReader() ran the body is locked,
  // so prefer the reader; otherwise cancel resp.body directly. All guarded: the blob
  // path never creates a reader, and a Response may have no cancelable body.
  const cancelInput = () => {
    if (reader) {
      try { reader.cancel(); } catch { /* ignore */ }
    } else if (resp.body && typeof resp.body.cancel === "function") {
      try { resp.body.cancel(); } catch { /* ignore */ }
    }
  };

  // Terminal helpers: first call wins; both free the blob URL (the MediaSource
  // URL is freed earlier, right after sourceopen).
  const finishEnded = () => {
    if (done) return;
    done = true;
    cancelInput();
    revokeUrl();
    if (onEnded) onEnded();
  };
  const finishError = (msg) => {
    if (done) return;
    done = true;
    cancelInput();
    revokeUrl();
    if (onError) onError(msg || "Audio playback failed");
  };

  // Wire the <audio> terminal events shared by both strategies.
  const wireAudio = (el) => {
    audio = el;
    el.addEventListener("ended", () => finishEnded());
    el.addEventListener("error", () => finishError("Audio playback failed"));
  };

  // Kick off audio.play(); resolve -> onPlaying, reject -> onError.
  const startPlayback = (el) => {
    const p = el.play();
    if (p && typeof p.then === "function") {
      p.then(() => { if (!done && onPlaying) onPlaying(); },
             () => finishError("Audio playback failed"));
    } else if (!done && onPlaying) {
      // Older browsers where play() returns undefined: assume it started.
      onPlaying();
    }
  };

  const canStreamMse =
    resp.body &&
    typeof MediaSource !== "undefined" &&
    mime &&
    MediaSource.isTypeSupported(mime);

  if (canStreamMse) {
    // --- MSE streaming path (audio/mpeg) ------------------------------------
    const mediaSource = new MediaSource();
    const el = new Audio();
    wireAudio(el);
    objectUrl = URL.createObjectURL(mediaSource);
    el.src = objectUrl;

    let sb = null;
    const queue = []; // pending Uint8Array chunks awaiting a non-updating SourceBuffer
    let streamDone = false; // reader exhausted

    // End the MediaSource once the reader is drained and the queue is flushed.
    const maybeEndStream = () => {
      if (streamDone && queue.length === 0 && sb && !sb.updating &&
          mediaSource.readyState === "open") {
        try { mediaSource.endOfStream(); } catch { /* already ended/closed */ }
      }
    };

    // MSE requires sequential appends: only appendBuffer when idle, and drive the
    // queue forward from the buffer's "updateend" event.
    const pump = () => {
      if (!sb || sb.updating || queue.length === 0) { maybeEndStream(); return; }
      const chunk = queue.shift();
      try {
        sb.appendBuffer(chunk);
      } catch (e) {
        finishError("Audio playback failed");
        return;
      }
    };

    const readLoop = async () => {
      try {
        for (;;) {
          const { done: rdone, value } = await reader.read();
          if (rdone) { streamDone = true; pump(); break; }
          if (value && value.byteLength) { queue.push(value); pump(); }
        }
      } catch {
        // Reader cancelled (stop) or a mid-stream network/synthesis failure.
        if (!done) finishError("Audio playback failed");
      }
    };

    mediaSource.addEventListener("sourceopen", () => {
      // The object URL has served its purpose (bound the element to this source);
      // free it now so it is never leaked.
      revokeUrl();
      if (done) return;
      try {
        sb = mediaSource.addSourceBuffer(mime);
      } catch {
        finishError("Audio playback failed");
        return;
      }
      sb.addEventListener("updateend", pump);
      sb.addEventListener("error", () => finishError("Audio playback failed"));
      reader = resp.body.getReader();
      readLoop();
    }, { once: true });

    startPlayback(el);
  } else {
    // --- Blob fallback path (audio/wav, or no usable MediaSource) ------------
    // Buffer the whole body, then play it as a single object URL — identical to
    // the pre-streaming behaviour.
    resp.blob().then((blob) => {
      if (done) return; // stopped while buffering
      const el = new Audio();
      wireAudio(el);
      objectUrl = URL.createObjectURL(blob);
      el.src = objectUrl;
      startPlayback(el);
    }).catch(() => finishError("Audio playback failed"));
  }

  return {
    // Idempotent: silence all future callbacks, stop playback, cancel the input
    // stream and free any held object URL. Never fires onEnded/onError.
    stop() {
      if (done) {
        // Already terminal: still make sure nothing is left holding resources.
        cancelInput();
        if (audio) { try { audio.pause(); } catch { /* ignore */ } }
        revokeUrl();
        return;
      }
      done = true;
      cancelInput();
      if (audio) { try { audio.pause(); } catch { /* ignore */ } }
      revokeUrl();
    },
  };
}
