# V10 angular-head — targeted research (tavily)

- **Margin type:** use **cosine margin (CosFace/AM-Softmax: s·(cosθ − m))**, NOT ArcFace
  angular (cos(θ+m)). Angular margin has an edge case (θ+m>π → margin paradoxically increases
  the logit, needs easy-margin/Taylor hack). Cosine margin is monotone & simpler. Operator
  also specified additive cosine margin. (ArcFace CVPR19; John Trimble writeup.)
- **Normalisation (ArcFace/CosFace):** L2-normalise BOTH features and class weights → logit =
  s·cosθ; embeddings live on a hypersphere of radius s. (s≈15–30; sweep.)
- **QUANTIZATION RISK (key):** Imagination Technologies edge-AI study — **INT8 is NOT
  recommended for reliable ArcFace scoring**; recommend 16-bit fixed-point OR per-channel, and
  note channel-wise ops add HW cost. → Expect a real quant-gap if we INT8 the cosine head.
  Plan: measure full-int8 quant-gap; if large, keep L2-norm + cosine head in **float32**
  (backbone stays int8 per-channel), per the operator's guidance.
- Amazon sub-8-bit QAT (Interspeech) exists but is heavier; not needed unless PTQ gap forces it.
