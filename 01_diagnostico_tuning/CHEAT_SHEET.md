# Chapter 4 Cheat Sheet — Tuning & Diagnostics as System Control

*Companion to* **Mastering CatBoost — Pro Edition**, Chapter 4.

## A staged tuning workflow (§4.2)

| Level | Concern | Knobs |
|---|---|---|
| 0 | Sanity & contract | correct `cat_features`, deployment-matched splits, fixed seed; consider `has_time=True` only when the pre-sorted internal-order contract is intended |
| 1 | Learning dynamics | jointly budget `learning_rate` and tree count; $N≈C/η$ is a rough heuristic, not a law; early stopping is one governed selection workflow |
| 2 | Model capacity | start with a small development grid for `depth` and `l2_leaf_reg`; their interaction and optimum are data-dependent |
| 3 | Robustness candidates | compare `bootstrap_type`, `bagging_temperature`/`subsample`, `rsm`, and `random_strength` after the split and primary metric are fixed |

**First-run candidate:** `iterations=2000, learning_rate=0.05, early_stopping_rounds=50, depth=6, l2_leaf_reg=3`. Treat curve diagnoses as hypotheses: compare L2, depth, and learning-rate candidates on development data before automating a wider search.

## The Stability Triangle (§4.3)

Dynamics (LR+tree budget) ↔ structure (depth) ↔ smoothing (L2). These controls interact, but deeper trees do not mechanically require a fixed L2 increase and faster learning does not mechanically require shallower trees; measure the joint surface. A full symmetric depth-$d$ tree has $2^d$ leaf slots; +1 depth doubles those slots but does not guarantee exactly double serialized model size. It adds one comparison per tree at inference.

## The Overfitting Detector (§4.4)

- **Strategy:** overcommit iterations → monitor eval set → abort.
- `early_stopping_rounds` = alias for `od_wait`, **forces `od_type='Iter'`** (discards `od_pval`).
- `Iter` is patience mode. `IncToDec` uses `od_pval`; compare them only when the validation-noise and repeated-look policy justify the choice.
- Set/verify `use_best_model=True` with an `eval_set` so the final model is truncated to the best iteration.
- An apparently premature stop motivates candidates such as more patience or a different learning rate; a flat curve motivates checking optimization, capacity, and data signal. Confirm the diagnosis with repeated development runs.
- Prefer a predeclared, reasonably selection-stable metric for early stopping. Logloss is smooth; AUC is threshold-free/rank-based but not smooth and may be noisy. Report policy-threshold metrics separately.

## Reading the curves (§4.5)

| Pathology | Symptom | Treatment |
|---|---|---|
| Divergence (observed overfit pattern) | train falls while validation curls up | first verify the split/metric, then compare stronger L2, shallower depth, and lower LR under early stopping; lower LR can change the path and budget, so measure rather than assuming it only delays the same outcome |
| Plateau candidate | both curves flat at unsatisfactory error | test capacity, regularization, features, and optimization candidates; the curve alone does not identify one cause |
| Spike (instability) | jagged jumps | lower `learning_rate`; check outliers / custom-loss numerics |
| Identifier/leak suspicion | abrupt development behavior or one identifier-like feature dominates | verify point-in-time availability and split integrity; drop illegitimate IDs; dominance is a review cue, not a 50% rule |
| Quantization steps | staircase loss curve | raise `border_count` (or `per_float_feature_quantization`) |

## The Ladder of Truth (§4.6)

1. `PredictionValuesChange` — fast, structural, biased toward high-cardinality features; triage only.
2. `LossFunctionChange` — model-conditional loss-change approximation requiring a labeled Pool; useful alongside ablations, not a uniquely faithful truth.
3. SHAP — per-prediction explanations; especially efficient for oblivious trees.
4. `type='Interaction'` — pairwise strength; ID × strong-feature interaction ⇒ contextual leak suspect.

## Cookbook triggers (§4.8)

- **ID-leak suspicion:** audit feature legitimacy and split construction first; dropping a pure ID is the control, while CTR complexity is only a capacity knob.
- **Small data:** compare stronger smoothing, shallower candidates, and resampling choices across seeds; no fixed L2/bootstrap/detector recipe is universal.
- **Imbalance:** compare unweighted and cost/weighting candidates on the deployment metric, calibration, and policy cost; freeze the early-stopping metric before search.
- **Forward-deployment time series:** use origin-respecting splits and point-in-time features. `has_time=True` is a candidate internal-order flag for pre-sorted data, not a substitute for the forecast protocol.
- **Latency:** benchmark tree count, depth, feature set, and native batching on target hardware. `border_count` is primarily a training/quantization control with model-dependent indirect inference effects.

## MLOps (§4.7)

`save_snapshot=True, snapshot_file=..., snapshot_interval=600` for fault tolerance (auto-resume with compatible data/params). CPU is the safer choice for regulatory-grade reproducibility; pin seed + version + threads and validate on your hardware. Model too big ⇒ lower `max_ctr_complexity`, use `model_size_reg`.

## Pitfalls

- Fix the data contract, estimand, split, and primary metric before interpreting a stochastic hyperparameter search.
- Lowering `learning_rate` without raising `iterations` ⇒ silent underfitting.
- `bagging_temperature` only with Bayesian; `subsample` only with Bernoulli/Poisson(GPU)/MVS.
- `boosting_type` defaults and support depend on objective/device. Treat Ordered-vs-Plain as a development comparison where supported; inspect `get_all_params()`.
- Defaults (`leaf_estimation_method`, bootstrap, one-hot threshold) are conditional on objective/device/version — always verify resolved values.
