#!/usr/bin/env python3
"""G2 second-stage verifier: a tiny MLP on the 64-d bottleneck embedding of the
KWS model, run ONLY after a KWS trigger, to reject no-VAD vocal-music false
accepts without hurting true positives.

Train: positives = real device wakes (+ synth), negatives = music windows the
v8 KWS falsely fires on (mined). Eval on DISJOINT pos + DISJOINT eval-music FAs.
Reports the FA-rejection (FAPH cut) at a fixed true-positive cost, + on-device cost.
"""
import sys, os, glob, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
sys.path.insert(0, "/home/claude/zakhar-mww")
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from v9.extract_embeddings import extract_dir
import tensorflow as tf

W8 = "/home/claude/zakhar-mww/trained_models/zakhar_v8/last_weights.weights.h5"

def paths(d): return sorted(glob.glob(os.path.join(d, "*.wav")))

# ---- positives: real device wakes, split disjoint; + synth for train robustness
pos = paths("/home/claude/zakhar-mww/v8/dev_heldout_pos")
pos_tr, pos_ev = pos[:100], pos[100:]
synth = paths("/home/claude/zakhar-mww/v8/pos_clean")[:600]
# ---- negatives: mined music false-accepts (train/eval music SOURCE-disjoint)
neg_tr = paths("/home/claude/zakhar-mww/v9/verif_neg_train") + paths("/home/claude/zakhar-mww/v9/verif_neg_train2")
neg_ev = paths("/home/claude/zakhar-mww/v9/verif_neg_eval") + paths("/home/claude/zakhar-mww/v9/verif_neg_eval2")
print(f"pos_tr {len(pos_tr)} pos_ev {len(pos_ev)} synth {len(synth)} neg_tr {len(neg_tr)} neg_ev {len(neg_ev)}")

def emb(ps):
    _, e = extract_dir(W8, ps); return e

Ep_tr, Ep_ev = emb(pos_tr + synth), emb(pos_ev)
En_tr, En_ev = emb(neg_tr), emb(neg_ev)
D = Ep_tr.shape[1]
# standardize on train
mu = np.concatenate([Ep_tr, En_tr]).mean(0); sd = np.concatenate([Ep_tr, En_tr]).std(0) + 1e-6
def z(x): return (x - mu) / sd
Xtr = np.concatenate([z(Ep_tr), z(En_tr)]); ytr = np.concatenate([np.ones(len(Ep_tr)), np.zeros(len(En_tr))])

clf = tf.keras.Sequential([
    tf.keras.layers.Input((D,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
cw = {0: 1.0, 1: len(En_tr) / max(1, len(Ep_tr))}  # balance
clf.fit(Xtr, ytr, epochs=80, batch_size=64, class_weight=cw, verbose=0)

sp_ev = clf.predict(z(Ep_ev), verbose=0).reshape(-1)   # true-pos scores
sn_ev = clf.predict(z(En_ev), verbose=0).reshape(-1)   # music-FA scores
print(f"\nemb_dim={D}  verifier params={clf.count_params()}")
print("thr   keep_TP(recall)  reject_music_FA   note")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    keep = float((sp_ev >= thr).mean())          # true positives retained
    rej = float((sn_ev < thr).mean())            # music FAs rejected (=FAPH cut)
    print(f"{thr:<5} {keep:0.3f}            {rej:0.3f}")
# pick operating point: highest FA-reject with >=0.98 TP retained
best = None
for thr in np.linspace(0.05, 0.95, 91):
    keep = (sp_ev >= thr).mean()
    if keep >= 0.98:
        best = (float(thr), float(keep), float((sn_ev < thr).mean()))
print(f"\nOP (TP-retain>=0.98): thr={best[0]:.2f} keep_TP={best[1]:.3f} reject_FA={best[2]:.3f}" if best else "no OP >=0.98")
print(f"score margin: TP scores min={sp_ev.min():.3f} mean={sp_ev.mean():.3f} | music-FA scores max={sn_ev.max():.3f} mean={sn_ev.mean():.3f}")
print(f"  -> separation gap (min TP - max FA) = {sp_ev.min()-sn_ev.max():.3f}  (N_pos_ev={len(sp_ev)} N_fa_ev={len(sn_ev)})")

# --- cheap on-device variant: logreg on time-pooled 64-d channel embedding ---
def pool64(E):  # Flatten was (T*64); reshape to (T,64) and mean over time
    T = E.shape[1] // 64
    return E.reshape(E.shape[0], T, 64).mean(1)
from sklearn.linear_model import LogisticRegression
Xp = np.concatenate([pool64(Ep_tr), pool64(En_tr)]); yp = ytr
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xp, yp)
sp2 = lr.predict_proba(pool64(Ep_ev))[:,1]; sn2 = lr.predict_proba(pool64(En_ev))[:,1]
keep2 = float((sp2>=0.5).mean()); rej2 = float((sn2<0.5).mean())
print(f"\nCHEAP logreg on 64-d pooled emb: params={64+1}  keep_TP={keep2:.3f} reject_FA={rej2:.3f} "
      f"gap={sp2.min()-sn2.max():.3f}")
# persist the CHOSEN verifier (cheap logreg on 64-d pooled bottleneck)
np.savez("/home/claude/zakhar-mww/v9/verifier/logreg64.npz",
         w=lr.coef_.reshape(-1).astype(np.float32), b=np.float32(lr.intercept_[0]))
json.dump({"type":"logreg_on_64d_timepooled_bottleneck","params":65,
           "keep_TP":keep2,"reject_music_FA":rej2,"margin":float(sp2.min()-sn2.max()),
           "N_pos_eval":int(len(sp2)),"N_musicFA_eval":int(len(sn2))},
          open("/home/claude/zakhar-mww/v9/verifier/verifier_chosen.json","w"), indent=2)
print("saved logreg64 verifier")
os.makedirs("/home/claude/zakhar-mww/v9/verifier", exist_ok=True)
clf.save("/home/claude/zakhar-mww/v9/verifier/verifier.keras")
np.savez("/home/claude/zakhar-mww/v9/verifier/norm.npz", mu=mu, sd=sd)
json.dump({"emb_dim": int(D), "params": int(clf.count_params()),
           "op_thr": best[0] if best else None, "keep_TP": best[1] if best else None,
           "reject_music_FA": best[2] if best else None},
          open("/home/claude/zakhar-mww/v9/verifier/verifier.json", "w"), indent=2)
print("saved verifier/")
