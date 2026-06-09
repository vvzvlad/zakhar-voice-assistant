# G8 — Architecture sweep (BC-ResNet/TC-ResNet vs mixednet) + hyperparam sweep

## BC-ResNet / TC-ResNet port — feasibility assessment

The mWW streaming framework exposes exactly the primitives a BC-ResNet needs:
`microwakeword/layers/`: `stream.py` (streaming ring-buffer wrapper), `delay.py`,
`sub_spectral_normalization.py` (**SubSpectralNorm is a core BC-ResNet block**),
`average_pooling2d.py`, `strided_drop.py`. So a faithful **streaming** BC-ResNet is
portable *in principle*: implement the broadcasted-residual block (1D temporal conv +
broadcast to freq-temporal, SubSpectralNorm) as `Stream(...)`-wrapped cells, like
mixednet's MixConv cells.

**But** it's a non-trivial new model module (`bc_resnet.py`) + converter validation +
arena tuning — several hours of build/debug on its own. Given the ~10–12 h breadth
budget shared across 11 goals and three trainings already cooking, a full port is
**out of scope for this run**; staged here as the highest-value follow-up. What we CAN
do honestly now at equal arena is sweep the mixednet design space (below), which already
covers the levers BC-ResNet would change (residuals, width, temporal context).

From literature (G11): BC-ResNet's edge is the accuracy/compute *frontier* at very low
MACs; at our 32 k-param / 45 KB-arena operating point mixednet is already competitive and
is what the converter fully supports. The sweep tests whether the BC-ResNet-style choices
(residual connections, width, window) help *here* on device-eval.

## Same-arena mixednet sweep (runs on the v9 multi-condition data, gated on device-eval)

| variant | change vs v8 student | hypothesis |
|---------|----------------------|-----------|
| `v9res` | residual_connection 0,0,1,1,1 (v8: all 0) | BC-ResNet-style residuals improve gradient flow / FRR |
| `v9cw`  | heavier positive class weight | trade FAPH headroom for lower FRR |
| `v9win` | shorter clip_duration (1500 ms) | smaller arena; does less context hurt device-recall? |

Each = same convert + device-eval + strat-eval gate as every candidate
(`v9/eval_candidate.sh`). Ship only if it beats v8 device-FRR 0.059 without FAPH regress.

(Results appended as runs complete.)

## Faithful AM-softmax angular head — convertibility CONFIRMED (forward task)

The v9 margin run used a crude logit-shift approximation and regressed. The proper port is a
real angular head (L2-normalise the bottleneck embedding, cosine classifier, additive margin
m≈0.35, scale s≈30). **Convertibility probe (done):** an int8 TFLite with `L2_NORMALIZATION`
before the FC head converts cleanly (op present in the quantized graph; TFLM supports the
L2_NORMALIZATION kernel) → on-device feasible.

Recipe (for the real-data iteration, NOT run now — data is the bottleneck, EV of a loss change
on the same data is low):
1. Add an `angular_head` flag to `mixednet.model`: after Flatten, insert
   `tf.math.l2_normalize(axis=-1)` then `Dense(1, use_bias=False)`; multiply by a fixed scale s.
2. Custom loop: read the pre-sigmoid logit = s·cosθ from the graph; apply margin in the LOSS only
   (subtract m from cosθ for positives), `loss = BCE(sigmoid(s·(cosθ − m·y)), y)`. Inference graph
   stays margin-free (l2norm + FC + logistic) → converts via the standard mWW path with the flag.
3. Gate on device-eval as usual. Most valuable once real far-field/media positives exist (so the
   tighter margin actually separates real hard positives, not synthetic-corrupted ones).
