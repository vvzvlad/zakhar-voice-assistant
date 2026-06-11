# v19 — BEST RECALL (angular, annealed). FRR 3.0% [1.7–5.4] (7× better than standard). NOT a drop-in:
FAPH unfixable in-model (silence 5/h, speech 8/h even with VAD). Needs OFFLINE per-trigger verifier
(firmware 2-stage). verifier_logreg.npz = a verifier trained on real FAs (offline rejects 94–100% of
silence/music/speech FAs at 98.6% TP-keep; was trained on v17 — retrain on v19 embeddings for exact use).
Ship only with the 2-stage verifier integrated.
