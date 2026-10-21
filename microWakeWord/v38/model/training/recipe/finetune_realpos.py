#!/usr/bin/env python3
"""
T8 finetune: continue-train the FOLDED v28 deploy head on operator's real short-«захар» positives.
Mirrors the proven nr2/finetune_deploy.py (standard head, gentle BCE, backbone already good), but:
  - reads a SYNTHESIZED config (adds the real-pos feature dir to the mix) instead of the hardcoded one,
  - saves to a scratchpad model dir (never overwrites repo artifacts),
  - inits from the v28 folded deploy weights (deployable standard head — no re-fold needed after).
Arch flags set explicitly to the deploy values so the export tflite matches the incumbent exactly.

Usage:
    finetune_realpos.py <synth_cfg.yaml> <OUT_model_dir> <steps> [DRY=0]
DRY=1 -> run only ~60 steps, print loss + output-logit distribution on a positive & a negative batch
         (ml-researcher #11: verify the DISTRIBUTION before committing a long run). Trains nothing else.

Run as `claude` via venv python:  PYTHONPATH=<root>/micro-wake-word venv/bin/python finetune_realpos.py ...
"""
import sys, os, types, logging
import numpy as np, tensorflow as tf
from microwakeword.model_train_eval import load_config
import microwakeword.mixednet as mixednet
import microwakeword.data as input_data

ROOT = "/home/claude/zakhar-mww"
INIT_WEIGHTS = f"{ROOT}/trained_models/zakhar_angmax_deploy/last_weights.weights.h5"  # folded v28

def flags(cfg_path):
    f = types.SimpleNamespace()
    f.model_name = "mixednet"; f.training_config = cfg_path; f.stride = 3
    f.train = 1; f.restore_checkpoint = 0; f.use_weights = "last_weights"
    f.verbosity = logging.ERROR
    f.max_pool = 0; f.spatial_attention = 0; f.pooled = 0; f.angular_head = 0
    # explicit arch — identical to angmax_deploy export (deploy_and_eval.sh)
    f.pointwise_filters = "64,64,64,64,64"
    f.repeat_in_block = "1, 1, 1, 1, 1"
    f.mixconv_kernel_sizes = "[5], [7,11], [9,15], [17,23], [29]"
    f.residual_connection = "0,0,0,0,0"
    f.first_conv_filters = 32; f.first_conv_kernel_size = 5
    return f

def main():
    if len(sys.argv) < 4:
        sys.exit("usage: finetune_realpos.py <synth_cfg.yaml> <OUT_model_dir> <steps> [DRY=0]")
    cfg_path, out, steps = sys.argv[1], os.path.abspath(sys.argv[2]), int(sys.argv[3])
    dry = len(sys.argv) > 4 and sys.argv[4] not in ("0", "", "false")
    if os.path.exists(os.path.join(out, "last_weights.weights.h5")):
        sys.exit(f"REFUSE: {out}/last_weights.weights.h5 exists — pick a fresh model dir (no overwrite).")
    if not os.path.isfile(INIT_WEIGHTS):
        sys.exit(f"missing v28 init weights: {INIT_WEIGHTS}")

    fl = flags(cfg_path); cfg = load_config(fl, mixednet)
    cfg["train_dir"] = out; os.makedirs(out, exist_ok=True)
    model = mixednet.model(fl, cfg["training_input_shape"], cfg["batch_size"])
    model.load_weights(INIT_WEIGHTS)
    dp = input_data.FeatureHandler(cfg)
    bs = cfg["batch_size"]; L = cfg["spectrogram_length"]
    # SpecAugment dict — base angmax config carries an EMPTY augmentation_policy, but
    # get_data() requires these keys. Use the proven nr2/finetune_deploy.py values.
    aug = cfg.get("augmentation_policy") or {}
    aug = {"freq_mix_prob": 0.0, "time_mask_max_size": 5, "time_mask_count": 2,
           "freq_mask_max_size": 5, "freq_mask_count": 2, **aug}
    for _k in ("time_mask_max_size", "time_mask_count", "freq_mask_max_size", "freq_mask_count"):
        aug.setdefault(_k, 5 if "size" in _k else 2)
    opt = tf.keras.optimizers.Adam(1e-4)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    if dry:
        steps = min(steps, 60)
        print(f"DRY finetune: {steps} steps, then distribution check", flush=True)
    else:
        print(f"FINE-TUNE real-pos deployable head: {steps} steps lr1e-4 -> {out}", flush=True)

    for i in range(steps):
        X, Y, Wt = dp.get_data("training", batch_size=bs, features_length=L,
                               truncation_strategy="default", augmentation_policy=aug)
        with tf.GradientTape() as tape:
            p = model(X, training=True)
            l = bce(Y, p, sample_weight=Wt)
        g = tape.gradient(l, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        if i % 2000 == 0 and not dry:
            model.save_weights(os.path.join(out, "last_weights.weights.h5"))
            print(f"step {i}/{steps} loss {float(l):.4f}", flush=True)

    if dry:
        # ml-researcher #11: is the head actually separating pos from neg? histogram both.
        for label in ("training",):
            X, Y, Wt = dp.get_data(label, batch_size=256, features_length=L,
                                   truncation_strategy="default", augmentation_policy=aug)
            p = np.asarray(model(X, training=False)).ravel(); y = np.asarray(Y).ravel()
            pos = p[y > 0.5]; neg = p[y <= 0.5]
            def h(a): return "n=%d mean=%.3f p10=%.3f p50=%.3f p90=%.3f" % (
                len(a), a.mean(), np.percentile(a,10), np.percentile(a,50), np.percentile(a,90)) if len(a) else "n=0"
            print(f"DRY_DIST pos: {h(pos)}", flush=True)
            print(f"DRY_DIST neg: {h(neg)}", flush=True)
        print("DRY_DONE — inspect separation before the full run. NOTHING SAVED.", flush=True)
        return
    model.save_weights(os.path.join(out, "last_weights.weights.h5"))
    print("FT_REAL_DONE", flush=True)

if __name__ == "__main__":
    main()
