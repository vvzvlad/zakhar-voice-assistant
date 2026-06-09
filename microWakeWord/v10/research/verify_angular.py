#!/usr/bin/env python3
"""Path-2 test: does a 2nd-stage verifier rescue the angular head's music-FAPH?
Build the angular model (angular_head=1) with s10hn weights, extract the L2-normalised
bottleneck embedding for true wakes + the model's own music false-fires, train a tiny
logreg verifier, and measure whether it rejects HELD-OUT (ambient) music FAs while keeping
true wakes. If yes => angular(recall) + verifier(music-FA cut) is a shippable combination.
"""
import sys, os, glob, types, logging, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
import tensorflow as tf, soundfile as sf
from microwakeword import mixednet
from microwakeword.model_train_eval import load_config
from microwakeword.audio.audio_utils import generate_features_for_clip
from sklearn.linear_model import LogisticRegression

W = "/home/claude/zakhar-mww/trained_models/zakhar_v10_s10hn/last_weights.weights.h5"
CFG = "/home/claude/zakhar-mww/training_parameters_v10_s10hn.yaml"
STUDENT = dict(fcf=32, fck=5, pw="64,64,64,64,64", rep="1,1,1,1,1",
               mk="[5],[7,11],[9,15],[17,23],[29]", res="0,0,0,0,0")
def fl():
    f=types.SimpleNamespace(); f.model_name="mixednet"; f.training_config=CFG; f.stride=3
    f.train=1; f.restore_checkpoint=0; f.use_weights="last_weights"; f.verbosity=logging.ERROR
    f.test_tf_nonstreaming=f.test_tflite_nonstreaming=0
    f.test_tflite_nonstreaming_quantized=f.test_tflite_streaming=f.test_tflite_streaming_quantized=0
    f.first_conv_filters=STUDENT["fcf"]; f.first_conv_kernel_size=STUDENT["fck"]
    f.pointwise_filters=STUDENT["pw"]; f.repeat_in_block=STUDENT["rep"]
    f.mixconv_kernel_sizes=STUDENT["mk"]; f.residual_connection=STUDENT["res"]
    f.max_pool=0; f.spatial_attention=0; f.pooled=0; f.angular_head=1
    return f
flags=fl(); cfg=load_config(flags, mixednet); win=cfg["training_input_shape"][0]
m=mixednet.model(flags, cfg["training_input_shape"], cfg["batch_size"]); m.load_weights(W)
emb=tf.keras.Model(m.input, [m.get_layer("prob").output, m.get_layer("l2norm").output])

def feats(p):
    d,sr=sf.read(p,dtype="int16"); d=d[:,0] if d.ndim>1 else d
    if sr!=16000: return None
    s=generate_features_for_clip(d.astype(np.float32)/32768.0, step_ms=10)
    s=s.astype(np.float32)*0.0390625 if np.issubdtype(s.dtype,np.integer) else s.astype(np.float32)
    if len(s)<win: s=np.concatenate([np.zeros((win-len(s),s.shape[1]),s.dtype),s])
    return s
def embed(paths):
    P=[];E=[]
    for p in paths:
        s=feats(p)
        if s is None: continue
        idx=list(range(0,len(s)-win+1,3)); X=np.stack([s[i:i+win] for i in idx]).astype(np.float32)
        pr,ee=emb.predict(X,batch_size=256,verbose=0); k=int(np.argmax(pr.reshape(-1)))
        P.append(float(pr.reshape(-1)[k])); E.append(ee[k].reshape(-1))
    return np.array(P),np.array(E)
g=lambda d:sorted(glob.glob(d+"/*.wav"))
pos=g("/home/claude/zakhar-mww/v8/dev_heldout_pos")
_,Ep_tr=embed(pos[:100]); _,Ep_ev=embed(pos[100:])
_,En_tr=embed(g("/home/claude/zakhar-mww/v10/avr_fa_train"))
_,En_ev=embed(g("/home/claude/zakhar-mww/v10/avr_fa_eval"))
print(f"pos_tr {len(Ep_tr)} pos_ev {len(Ep_ev)} fa_tr {len(En_tr)} fa_ev(ambient,held-out) {len(En_ev)}")
if len(En_tr)<5 or len(En_ev)<5: print("too few FAs to train/eval"); sys.exit()
X=np.concatenate([Ep_tr,En_tr]); y=np.concatenate([np.ones(len(Ep_tr)),np.zeros(len(En_tr))])
lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(X,y)
sp=lr.predict_proba(Ep_ev)[:,1]; sn=lr.predict_proba(En_ev)[:,1]
for thr in [0.3,0.4,0.5,0.6]:
    print(f"thr {thr}: keep_TP {float((sp>=thr).mean()):.3f}  reject_ambientMusicFA {float((sn<thr).mean()):.3f}")
print(f"margin minTP {sp.min():.3f}  maxFA {sn.max():.3f}")

# persist the deployable verifier (trained on fma FAs, validated on held-out ambient)
import json
np.savez("/home/claude/zakhar-mww/output_v10/candidates/angular_verifier_logreg.npz",
         w=lr.coef_.reshape(-1).astype(np.float32), b=np.float32(lr.intercept_[0]))
json.dump({"on":"angular s10hn l2norm bottleneck (64-d)","params":int(lr.coef_.size+1),
           "keep_TP@0.3":float((sp>=0.3).mean()),"reject_ambientMusicFA@0.3":float((sn<0.3).mean()),
           "margin":float(sp.min()-sn.max()),"N_pos_eval":int(len(sp)),"N_ambientFA_eval":int(len(sn))},
          open("/home/claude/zakhar-mww/output_v10/candidates/angular_verifier.json","w"),indent=2)
print("saved angular verifier")
