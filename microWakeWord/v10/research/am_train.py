#!/usr/bin/env python3
"""TRUE AM-Softmax / CosFace head trainer for binary KWS.

Head (in mixednet via --angular_head): Flatten -> L2Normalize(e) -> Dense(1,no bias) -> sigmoid.
We project the Dense kernel to norm s after every step => raw pre-sigmoid logit = s*cos(theta)
exactly (e and W unit, W scaled by s). Additive COSINE margin on positives in the LOSS only:
    z = s*cos(theta) - s*m*[y==1]  = raw_logit - s*m*y
    loss = weighted BCE(sigmoid(z), y)
Inference graph stays margin-free (standard L2norm+FC+sigmoid) => converts via mWW unchanged.

Usage: am_train.py <train_dir> <steps> <s> <m>   (trains on clean v8 data)
"""
import sys, os, types, logging, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
import tensorflow as tf
from microwakeword import mixednet, data as input_data
from microwakeword.model_train_eval import load_config

CFG = os.environ.get("AM_CFG", "/home/claude/zakhar-mww/training_parameters_v10am.yaml")  # v8 clean data
train_dir = sys.argv[1]; STEPS = int(sys.argv[2]); S = float(sys.argv[3]); M = float(sys.argv[4])
STUDENT = dict(fcf=32, fck=5, pw="64,64,64,64,64", rep="1,1,1,1,1",
               mk="[5],[7,11],[9,15],[17,23],[29]", res="0,0,0,0,0")

def flags():
    f = types.SimpleNamespace()
    f.model_name="mixednet"; f.training_config=CFG; f.stride=3
    f.train=1; f.restore_checkpoint=0; f.use_weights="last_weights"; f.verbosity=logging.ERROR
    f.test_tf_nonstreaming=f.test_tflite_nonstreaming=0
    f.test_tflite_nonstreaming_quantized=f.test_tflite_streaming=f.test_tflite_streaming_quantized=0
    f.first_conv_filters=STUDENT["fcf"]; f.first_conv_kernel_size=STUDENT["fck"]
    f.pointwise_filters=STUDENT["pw"]; f.repeat_in_block=STUDENT["rep"]
    f.mixconv_kernel_sizes=STUDENT["mk"]; f.residual_connection=STUDENT["res"]
    f.max_pool=0; f.spatial_attention=0; f.pooled=0; f.angular_head=1
    return f

fl=flags(); config=load_config(fl, mixednet); config["train_dir"]=train_dir
os.makedirs(train_dir, exist_ok=True)
dp=input_data.FeatureHandler(config)
shape=config["training_input_shape"]; bs=config["batch_size"]
model=mixednet.model(fl, shape, bs)
logit_model=tf.keras.Model(model.input, model.get_layer("cos_logit").output)
Wvar=model.get_layer("cos_logit").kernel    # shape (D,1)

def project_W():
    w=Wvar.numpy(); n=np.linalg.norm(w)+1e-9; Wvar.assign(w/n*S)
project_W()

opt=tf.keras.optimizers.Adam()
steps_list=config["training_steps"]; lrs=config["learning_rates"]
pcw=config["positive_class_weight"]; ncw=config["negative_class_weight"]; EPS=1e-6

@tf.function
def step(x,y,w,lr):
    opt.learning_rate.assign(lr)
    with tf.GradientTape() as tape:
        raw=logit_model(x, training=True)              # = s*cos(theta)
        z=raw - S*M*y                                  # cosine margin on positives (y in {0,1})
        z=tf.clip_by_value(z, -30.0, 30.0)
        p=tf.sigmoid(z)
        per=-(y*tf.math.log(p+EPS)+(1-y)*tf.math.log(1-p+EPS))
        loss=tf.reduce_sum(per*w)/(tf.reduce_sum(w)+EPS)
    g=tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(g, model.trainable_variables))
    return loss

def cum(i):
    c=0
    for k,s in enumerate(steps_list):
        c+=s
        if i<=c: return lrs[k],(pcw[k] if k<len(pcw) else pcw[-1]),(ncw[k] if k<len(ncw) else ncw[-1])
    return lrs[-1],pcw[-1],ncw[-1]

aug={"freq_mix_prob":0.0,"time_mask_max_size":5,"time_mask_count":2,"freq_mask_max_size":5,"freq_mask_count":2}
print(f"AM-Softmax s={S} m={M} | clean v8 data | steps {STEPS}", flush=True)
for i in range(1, STEPS+1):
    lr,pc,nc=cum(i)
    X,Y,W=dp.get_data("training", batch_size=bs, features_length=config["spectrogram_length"],
                      truncation_strategy="default", augmentation_policy=aug)
    Y=Y.reshape(-1,1).astype("float32")
    cw=np.where(Y>0.5, pc, nc).reshape(-1).astype("float32")
    w=(W*cw).astype("float32")
    loss=step(tf.constant(X,tf.float32), tf.constant(Y,tf.float32), tf.constant(w,tf.float32),
              tf.constant(lr,tf.float32))
    project_W()                                        # keep ||W||=s -> raw=s*cos(theta)
    if i%1000==0:
        model.save_weights(os.path.join(train_dir,"last_weights.weights.h5"))
        print(f"step {i}/{STEPS} loss {float(loss):.4f} lr {lr} |W|={float(tf.norm(Wvar)):.2f}", flush=True)
model.save_weights(os.path.join(train_dir,"last_weights.weights.h5"))
print("AM_DONE", flush=True)
