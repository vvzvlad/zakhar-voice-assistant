#!/usr/bin/env python3
"""Extract the pre-Dense bottleneck embedding (Flatten output) from a mixednet
model for a set of wav clips. For each clip we slide the non-streaming window,
take the window with MAX trigger-prob (= where the streaming KWS would fire), and
return that window's (prob, embedding). This mirrors the 2nd-stage verifier at
deployment: KWS fires -> verifier reads the same bottleneck -> tiny classifier.

CLI: extract_embeddings.py <weights.h5> <out.npz> <label:0|1> <wav_dir_or_filelist> [max_clips]
Reusable: import extract_dir(weights, paths) -> (probs[N], embs[N,D]).
"""
import sys, os, glob, types, logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
import tensorflow as tf
from microwakeword import mixednet
from microwakeword.model_train_eval import load_config
from microwakeword.audio.audio_utils import generate_features_for_clip

CFG = "/home/claude/zakhar-mww/training_parameters_v8.yaml"
STUDENT = dict(fcf=32, fck=5, pw="64,64,64,64,64", rep="1, 1, 1, 1, 1",
               mk="[5], [7,11], [9,15], [17,23], [29]", res="0,0,0,0,0")

def _flags(arch):
    f = types.SimpleNamespace()
    f.model_name = "mixednet"; f.training_config = CFG; f.stride = 3
    f.train = 1; f.restore_checkpoint = 0; f.use_weights = "last_weights"; f.verbosity = logging.ERROR
    f.test_tf_nonstreaming = f.test_tflite_nonstreaming = 0
    f.test_tflite_nonstreaming_quantized = f.test_tflite_streaming = f.test_tflite_streaming_quantized = 0
    f.first_conv_filters = arch["fcf"]; f.first_conv_kernel_size = arch["fck"]
    f.pointwise_filters = arch["pw"]; f.repeat_in_block = arch["rep"]
    f.mixconv_kernel_sizes = arch["mk"]; f.residual_connection = arch["res"]
    f.max_pool = 0; f.spatial_attention = 0; f.pooled = 0  # v8 defaults
    return f

_MODEL = {}
def _get(weights, arch=STUDENT):
    key = (weights, id(arch))
    if key in _MODEL: return _MODEL[key]
    fl = _flags(arch); cfg = load_config(fl, mixednet)
    shape = cfg["training_input_shape"]; bs = cfg["batch_size"]
    m = mixednet.model(fl, shape, bs); m.load_weights(weights)
    emb = tf.keras.Model(m.input, [m.layers[-1].output, m.layers[-2].output])  # prob, embedding
    win = shape[0]
    _MODEL[key] = (emb, win)
    return _MODEL[key]

def load16k(p):
    d, sr = sf.read(p, dtype="int16")
    if d.ndim > 1: d = d[:, 0]
    return d.astype(np.float32) / 32768.0, sr

def extract_dir(weights, paths, arch=STUDENT, batch=256):
    emb, win = _get(weights, arch)
    probs, embs = [], []
    for p in paths:
        try:
            d, sr = load16k(p)
            if sr != 16000: continue
            spec = generate_features_for_clip(d, step_ms=10)
            spec = spec.astype(np.float32) * 0.0390625 if np.issubdtype(spec.dtype, np.integer) else spec.astype(np.float32)
            if len(spec) < win:  # front-pad with silence (= training's truncate_start)
                spec = np.concatenate([np.zeros((win-len(spec), spec.shape[1]), spec.dtype), spec])
            # sliding windows (stride 3 -> deploy cadence); cap windows per clip
            idx = list(range(0, len(spec) - win + 1, 3))
            X = np.stack([spec[i:i+win] for i in idx]).astype(np.float32)
            pr, em = emb.predict(X, batch_size=batch, verbose=0)
            pr = pr.reshape(-1)
            k = int(np.argmax(pr))
            probs.append(float(pr[k])); embs.append(em[k].reshape(-1))
        except Exception:
            continue
    return np.array(probs, np.float32), np.array(embs, np.float32)

if __name__ == "__main__":
    weights, out, label, src = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    mx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    if os.path.isdir(src): paths = sorted(glob.glob(os.path.join(src, "*.wav")))
    else: paths = [l.strip() for l in open(src) if l.strip()]
    if mx: paths = paths[:mx]
    pr, em = extract_dir(weights, paths)
    np.savez(out, probs=pr, embs=em, label=np.full(len(pr), label, np.int8))
    print(f"{out}: {len(pr)} clips, emb_dim={em.shape[1] if em.size else 0}, mean_prob={pr.mean():.3f}")
