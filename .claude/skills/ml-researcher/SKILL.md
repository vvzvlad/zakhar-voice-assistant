---
name: ml-researcher
description: Experiment & evaluation discipline for ML work (esp. on-device KWS / small-data audio). Use BEFORE designing or running any experiment, building or trusting an eval, validating a dataset, or claiming ANY result ("recall went up", "X beats Y", "the fix works", "is X better", "did the fix work"). Portable do/don't rules + concrete methods/thresholds/sources, each grounded in both external best-practice and (⚑) a real bug this project hit. Full sourced reference: best-practices-reference.md. Triggers: «эксперимент», «оценить модель», «recall/FAPH», «датасет», «проверить результат».
---

# Researcher — rules for honest ML experiments & eval

Theme: **the measurement lies until proven clean, and a "win" is the strongest signal you have a bug.**
Be skeptical of your OWN numbers first. `⚑` = a place this exact rule already caught a wrong conclusion
here. Sources + full formulations: [best-practices-reference.md](best-practices-reference.md).

## 1. Leakage-safe eval — the harness is a suspect
- **Split disjoint on EVERY shared key**: speaker/voice, session, source, uuid. Prove `train ∩ eval = ∅` programmatically (log the intersection size, require 0). *(COVYT: same speaker both sides → re-identifies instead of solving the task.)*
- **Derive each model's train set from its ACTUAL config** (feature-dirs/manifests), never from memory. ⚑ *audit assumed v16 used only nr2_pos; really v16/v19/v27 all pull v8 dirs → wrong clean-set.*
- **Post-hoc near-duplicate audit** (audio embeddings/hashes) across train↔eval; **adversarial validation** (train-vs-test classifier, target AUC≈0.5 — >~0.6 = leak/shift, feature-importance names the leaker). ⚑ *~88% leaked eval (shared voices, shared feature-dirs, ±30 s session-neighbor negatives, byte-identical music dupes) → "recall 80%" was ~65%.*
- **No preprocessing/normalization/feature-selection before the split** (fit on train only; sklearn Pipeline; augment only after split).

## 2. Streaming/stateful offline eval
- **Fresh state per clip** for independent examples — never reuse one model instance across clips; verify with **reset vs preserve vs shuffle-order** (robust to shuffle = correct independence). ⚑ *state-leak (ring buffers + sliding-window prob) bled from neighbor clips → faked short-fire 65→23%; invisible on long clips, corrupts short. (Google saw up to 2× accuracy distortion without reset.)*
- **Eval in the SAME streaming mode you deploy** (chunk/lookahead/left-context); cache-aware, not buffered recompute. If state must carry (long-form), model it on train too.

## 3. Honest metrics on small samples
- **Every rate gets a CI**: Wilson for proportions (recall/FRR) `statsmodels…proportion_confint(method='wilson')`; **Poisson exact** for rare counts (FA/hour). ⚑ *"v27 strictly beats v16" was Δrecall=0.000 CI[−.19,.19]; its FAPH "win" = 1 silence+1 music fire, Poisson CI[0.07,16].*
- **Compare models with a PAIRED test** on the shared set — McNemar (`mlxtend`, exact if b+c<25) or paired bootstrap (≥1000 resamples, fixed seed); Bonferroni for multiple. **Overlapping CIs / p>0.05 ⇒ a tie ⇒ keep the incumbent**, don't ship the "winner".
- **Small n → say it, widen the CI, don't generalize from a probe.** ⚑ *16-clip test "drawn keep 81%" → full batch 49%; "short 37%" was n=8 noise (~55–65%).* Track the right unit (per-item ≠ per-voice ≠ per-source).

## 4. False-accept (FAPH) standards
- **Never report detection/FRR without FA/hour** at a fixed operating point ("%" is not a FAR unit). Compare models **at a FIXED FA-rate** (e.g. 1 FA/10 h), then read off miss/recall.
- **Large held-out negative corpus**, **environment-held-out** (silence/music/speech/vacuum/TV separately), FA/hour with Poisson CI; define **latency-window + lockout** so repeated spikes in one segment aren't counted as separate FAs. ⚑ *only radio-FAPH was trustworthy (others session-leaked, <0.35 h); v27's whole FA profile = single-fire counts → all class differences within Poisson noise.*

## 5. Train/serve gap (on-device audio)
- **Validate on REAL device re-recording**, not just synthetic — raw clean data can HURT; RIR/noise/MUSAN aug only closes the gap partially. Capture on the exact serve channel. ⚑ *raw synthetic hurt recall; only device-tract helped; mic_channel=0 is what the model hears.*
- **Validate at the RIGHT stage**: content on the clean SOURCE, degradation on the device-tract — never filter on a cue the tract destroys; account for the device frontend (AEC/AGC/denoise) in training data. ⚑ *device-tract STT can't hear the soft trailing «р» of drawn «захааар» (turbo AND large-v3, evenly across lengths) → validate the source, not the tract.*

## 6. Dataset validation
- **Run `mfa validate` (or equivalent) BEFORE alignment/training**; filter by ASR/alignment confidence; hand-check G2P for the keyword; validate labels at dataset-build, not after training.
- **Know your validator's failure modes before trusting keep/reject — and never "loosen" a filter until it admits garbage.** ⚑ *turbo+strict-«р» false-rejected good clips; the loose fix then admitted truncated «захааа» (no «р») = bad positives. large-v3 recovered only 13% of rejects → it's a tract artifact, not a model choice.*

## 7. Adversarial / red-team the RESULT
- **After any result/eval, spawn skeptic subagents to BREAK it** (leakage, state-leak, dup, optimistic split, sample-size, overfit); default verdict = "refuted unless proven"; re-derive headlines a second way. **The bigger the win, the harder you red-team** — "too good" ≈ leakage, not success. ⚑ *the adversarial subagent pass is the only reason we caught the 88% leak and the v27-tie.*
- Check that any **pretrained/transfer model's train data doesn't overlap your test** (model leakage).

## 8. Generalization / recall ceiling (unseen speakers)
- **More DISTINCT speakers is the strongest, most reliable lever** — error falls ~logarithmically (EER 50→5.2%, 500→1.9%, 1500→1.2%, 3000→0.9%). Diversity > volume-per-speaker. ⚑ *recall floor is a generalization gap (val 98.7% vs held-out ~65%); the fish diverse-voice round is the lever, not more clips of the same.*
- **Margin loss (AAM-softmax/ArcFace) > vanilla softmax** for open-set (~25–30% EER) — but it's **not free** (m/s-sensitive, can lose to tuned triplet; sweep m,s). ⚑ *angular gives the best recall (FRR 3-6%) but isn't deployable as-is (fires on silence at stage-1, inseparable in angular space — 0/39 verifier configs fix it). Lever to revive = retrain angular with heavy silence negs.*
- **Augment (MUSAN/RIR/SpecAugment)**; add **targeted synthetic negatives for confusables** (vowel-group / c-AUC metric).

## 9. Duration / prosody-aware KWS
- **Phoneme/onset matching alone won't separate look-alike words** (add prosody/duration features; duration-penalty vs over-long false fires; keep train/inference duration consistent). ⚑ *the model is an ONSET detector — short «захар» vs drawn «захааар» not separated (~65% short-fire); unsolved.*

## 10. Reproducibility & don't-ship-on-a-lie
- **Seed everything + determinism** (random/np/torch/cuda, `cudnn.deterministic`, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, DataLoader `worker_init_fn`) — accepting PyTorch determinism is incomplete (CPU↔GPU/atomicAdd).
- **Track every run** (git commit + data version&hash + params + metrics + env); version data via DVC; pin env as an artifact. Keep a dated run-log; save manifests/transcriptions/seeds for re-audit; **never ship/commit a "win" without an honest, leakage-safe bench — keep the incumbent until the bench proves a real gain.** ⚑ *.226 kept a NIGHT log + manifests, kept v27 prod, pushed nothing.*

## 11. Footguns the research didn't cover (ours)
- **Verify the output DISTRIBUTION, not "exit 0"** (counts/histogram/spot-listen + dry-run before big runs). ⚑ *`--drawn-tags 1` silently made ALL drawn = a2; default `--model s1` mangled the word — both "ran fine".*
- **Find the real bottleneck before scaling the cheap part.** ⚑ *generation is cheap+parallel, recapture is real-time on ONE device — over-generated 36k ≈ 3 days of recapture.*
- **Measure causal hypotheses before acting.** ⚑ *I asserted the capture window cut the «р»; measuring the envelope showed the word intact — a soft-tract «р», not a cut.*

## Pre-ship checklist
- [ ] speaker/session/source/uuid-disjoint split proven (intersection 0) + adversarial-val AUC≈0.5 + near-dup audit
- [ ] streaming eval in deploy-mode, fresh-state-per-clip verified (reset/preserve/shuffle)
- [ ] every metric has a CI (Wilson/Poisson); model comparison is paired (McNemar/bootstrap), not "by eye"
- [ ] FAPH on a large environment-held-out negative corpus, at a fixed FA-rate, with lockout window
- [ ] dataset validated at the right stage; validator failure-modes known
- [ ] result red-teamed by skeptic agents; any "too good" number leakage-audited
- [ ] output distribution / defaults checked; real bottleneck identified
- [ ] seeds + determinism + env pinned; run fully logged; nothing shipped within noise
