// Browser-media integration for the panel's "Test voice" preview. Given the raw
// Response from /api/tts/test (always audio/wav), the whole body is buffered into a
// Blob and played at once.
//
// Framework-agnostic (no React): returns a controller `{ stop }`. Exactly ONE
// terminal callback (onEnded or onError) fires per playback, and stop() suppresses
// both.

// playAudioResponse(resp, { onPlaying, onEnded, onError }) -> { stop }
export function playAudioResponse(resp, { onPlaying, onEnded, onError } = {}) {
  // One-shot terminal guard: ensures a single onEnded/onError, and that stop()
  // silences any later event. The held object URL is revoked exactly once.
  let done = false;
  let audio = null;
  let objectUrl = null; // blob URL still awaiting revocation

  const revokeUrl = () => {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
  };

  // Terminal helpers: first call wins; both free the blob URL.
  const finishEnded = () => {
    if (done) return;
    done = true;
    revokeUrl();
    if (onEnded) onEnded();
  };
  const finishError = (msg) => {
    if (done) return;
    done = true;
    revokeUrl();
    if (onError) onError(msg || "Audio playback failed");
  };

  resp.blob().then((blob) => {
    if (done) return; // stopped while buffering
    const el = new Audio();
    audio = el;
    el.addEventListener("ended", () => finishEnded());
    el.addEventListener("error", () => finishError("Audio playback failed"));
    objectUrl = URL.createObjectURL(blob);
    el.src = objectUrl;
    // Kick off play(); resolve -> onPlaying, reject -> onError.
    const p = el.play();
    if (p && typeof p.then === "function") {
      p.then(() => { if (!done && onPlaying) onPlaying(); },
             () => finishError("Audio playback failed"));
    } else if (!done && onPlaying) {
      // Older browsers where play() returns undefined: assume it started.
      onPlaying();
    }
  }).catch(() => finishError("Audio playback failed"));

  return {
    // Idempotent: silence all future callbacks, stop playback and free any held
    // object URL. Never fires onEnded/onError.
    stop() {
      done = true;
      if (audio) { try { audio.pause(); } catch { /* ignore */ } }
      revokeUrl();
    },
  };
}
