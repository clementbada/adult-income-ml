# Decisions log

Every methodological decision, with its reason and what could invalidate it.
Format of the course: what we did, why, and what could invalidate it. This file feeds the report.

---

## STEP 1 - Problem framing, EDA, leakage audit, split (done together by the three members)
Details, numbers and figures: `notebooks/01_EDA_framing_split.ipynb`. Code: `src/data.py`.

### Framing
- **Unit:** one person (a prospect). **Prediction time:** onboarding / KYC, before the bank knows the actual income.
  **User:** the wealth-management sales team. **Decision:** contact or not, via a threshold on P(income > 50K).
- The model outputs a probability; the threshold is chosen later (STEP 4); the log-loss trains the model.
- **Limitation:** 1994 US census data; a real deployment today would face distribution shift.

### Data facts
- 48,842 rows, 14 variables, prevalence 23.9%. A naive "always <=50K" model gets 76.1% accuracy,
  so accuracy is not our metric: PR-AUC (random baseline = prevalence) plus precision and recall.
- Missing values only in `workclass` (5.7%), `occupation` (5.8%), `native-country` (1.8%). For `workclass` and
  `occupation` they are informative (9.5% >50K when missing vs 24.8% when filled in).
- Strongest signals: marital status / relationship, occupation, education (jump at Bachelors), age (bell-shaped,
  peak 45-55: non-linear), hours worked.
- Sex gap in the data: 30.4% (men) vs 10.9% (women) >50K, mostly a composition effect (married men and married
  women have the same rate, 44.6% vs 44.8%).

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 1 | Target `>50K` = 1; main metric PR-AUC | Imbalance 23.9%: accuracy misleading | - | STEP 3 |
| 2 | Remove the 52 exact duplicates (fnlwgt included) before the split; keep the 6,374 same-profile rows | Same record in train and test = group leakage; same-profile rows are distinct people (1,029 profiles have both classes) | Two distinct people with identical weight and profile (negligible effect) | `src/data.py` |
| 3 | Drop `fnlwgt` | Census survey weight, not a property of the person, never available to a bank | - | STEP 2 Pipeline |
| 4 | Drop `education`, keep `education-num` | Exact one-to-one redundancy; `education-num` keeps the order in one column | A diploma with several codes (checked: none) | STEP 2 Pipeline |
| 5 | Missing categorical values become an explicit `"Missing"` category | Missingness is informative (people outside employment) | Missingness caused by a different process in production | STEP 2 Pipeline |
| 6 | Keep the 244 rows with `capital-gain = 99999` unchanged | Real wealthy people (valid top-code); deleting them biases the data, replacing the value invents an amount | - | becomes `has_capital_gain = 1` in STEP 2 |
| 7 | Replace capital amounts by `has_capital_gain` / `has_capital_loss` | Same-year amounts not known at prediction time; values behave like codes (122 distinct values, all-or-nothing outcomes); exact amounts add +0.08 PR-AUC we cannot justify (0.827 vs 0.749 vs 0.715 without capital, quick CV check with a default LightGBM) | Bank receives the same-year tax return before deciding | STEP 2 Pipeline; compared in STEP 6; STEP 8 |
| 8 | `sex` and `race` are not model inputs; kept aside for the audit | Targeting on sex or race is illegal for a French / EU bank | - | STEP 2 Pipeline, STEP 9 |
| 9 | `relationship`: merge `Husband` and `Wife` into `Spouse` | Keeps the household role (Wife 46.9% vs Husband 44.9%: same signal), removes the most direct sex proxy | Large performance loss (checked in STEP 9) | STEP 2 Pipeline |
| 10 | Fairness audit: `sex` main attribute (metrics + mitigation); `race` per group with sizes, no mitigation, no merging of groups | Small race groups (~80-95 people in test) give noisy metrics; merging would hide opposite effects | - | STEP 9 |
| 11 | 80/20 stratified split, seed 42; 5-fold StratifiedKFold in development; test used once | Same prevalence everywhere; no time order (cross-section) and no group identifier | A timestamp or household identifier | `src/data.py` |

### Notes for STEP 2
`has_capital_gain` / `has_capital_loss`; age bands (bell-shaped relation, for the linear model); hours bands;
group rare `native-country` values (38 of 41 countries below 0.5%).

---

## STEP 2 - Feature engineering (done together by the three members)
Details, numbers and figures: `notebooks/02_feature_engineering.ipynb`. Code: `src/features.py`.
Settings: `config.yaml`, section `features`. Tests: `tests/test_features.py`.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 12 | Two-layer Pipeline: stateless `AdultFeatureBuilder` (row rules), then a learned `ColumnTransformer` (imputation, encoding, scaling) | Anything learned from data must be fitted on training folds only; production receives raw rows | - | `src/features.py` |
| 13 | Two preprocessors: `linear` (one-hot, scaling, bands) and `tree` (ordinal codes, no scaling, no bands) | Linear models cannot cut variables into ranges and are scale-sensitive; trees can and are not | - | `build_preprocessor(cfg, family)` |
| 14 | Implements decisions 3, 4, 7, 8, 9: `fnlwgt`, `education`, `sex`, `race` and the capital amounts never reach the model; `Husband`/`Wife` -> `Spouse` (45.1% >50K) | STEP 1 | - | `AdultFeatureBuilder` |
| 15 | Missing categorical -> `"Missing"` (decision 5); median imputer on numbers as a safeguard | Missingness informative (9.5% vs 24.8%); a production row may be incomplete | Missingness caused by another process in production | `ColumnTransformer` |
| 16 | One-hot for linear, categories < 0.5% of training rows grouped (38 countries = 5.9% of rows), unknown categories tolerated; ordinal codes for trees (`categorical_mask` tells LightGBM which columns are categorical); no target encoding | No order between categories; rare levels give noisy coefficients; target encoding adds a leakage risk for no need (max 42 levels) | Rare countries carrying signal (checkable in STEP 4) | `ColumnTransformer` |
| 17 | Engineered: `has_capital_gain`, `has_capital_loss` (all models); `age_band` (6), `hours_band` (4), linear only; no `net_capital` | Indicators: decision 7. Bands: bell-shaped age, stepped hours. A loss signals wealth (50.0% >50K), a net value would reverse it | Linear model still underfitting age (-> spline) | `AdultFeatureBuilder`, `config.yaml` |
| 18 | `capital` option: `indicators` (default), `amounts` (log1p for linear, raw for trees), `none` | STEP 6 must rebuild the "exact amounts" reference with the same code | - | `config.yaml`, `build_preprocessor` |
| 19 | Evidence (default logistic regression, 5-fold CV, PR-AUC): 0.688 no capital, no band -> 0.729 indicators -> 0.745 with both bands; bands better on 5/5 folds | A feature is kept only if it improves the honest score | - | notebook 02, Part F |

### Limitation
Band edges were chosen after the STEP 1 EDA, which looked at all rows. Edges are round and domain-based (life stages,
part-time below 35 hours), not tuned on the target, and the shape is re-checked on the development set only.

### Notes for STEP 3 and STEP 4
- Class weights or resampling go **after** the preprocessor (resampling requires an `imblearn` Pipeline).
- Logistic regression: `build_preprocessor(cfg, "linear")`; trees: `build_preprocessor(cfg, "tree")`;
  LightGBM (STEP 5): pass the categorical columns given by `categorical_mask(fitted_pre)`.
- Linear reference to beat: PR-AUC 0.745 (default logistic regression).
- STEP 8 (SHAP): with one-hot, sum SHAP values per original variable for the global view.

---

## STEP 3 - Baseline, evaluation module, experiment tracking (done together by the three members)
Details, numbers and figures: `notebooks/03_baseline_evaluation.ipynb`. Code: `src/evaluate.py`, `src/models.py`,
`src/tracking.py`. Settings: `config.yaml`, sections `models`, `evaluation`, `tracking`. Tests: `tests/test_models.py`.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 20 | One evaluation function for every model: PR-AUC (main, random baseline = prevalence), ROC-AUC, log-loss, precision, recall, F1, confusion matrix | Same metrics, same folds, fair comparison (requirement 4); PR-AUC suits a 23.9% minority class (S6) | - | `src/evaluate.py` |
| 21 | Threshold-dependent metrics reported at 0.5 only as a reference | No universal threshold (S6); the operating threshold is chosen in STEP 4 on out-of-fold predictions | - | `config.yaml` (`evaluation`) |
| 22 | Cross-validation keeps out-of-fold predictions; the full Pipeline is refitted on each training fold | No leakage from the validation fold; OOF predictions needed for STEP 4 (threshold) and STEP 7 (calibration) | - | `cross_validate_oof` |
| 23 | Baseline = logistic regression, L2 penalty, C = 1 (default), scaled inputs | Course reference model (S1); L2 keeps correlated coefficients stable (S2); C chosen in STEP 5 | - | `src/models.py`, `config.yaml` (`models`) |
| 24 | Baseline result (5-fold CV, development set): PR-AUC 0.745 +/- 0.012, ROC-AUC 0.901 +/- 0.005, log-loss 0.328; at t = 0.5 precision 0.724, recall 0.595, F1 0.653 | Reference every later model must beat | - | notebook 03 |
| 25 | MLflow logs every run (parameters, per-fold and summary metrics, artifacts) in the local `mlruns/` folder | Requirement 7; needed for the Optuna trials of STEP 5 | Session 10 using another tool | `src/tracking.py` |
| 26 | Business priority = recall: a missed wealthy prospect (FN) costs more than a useless call (FP), so the operating threshold will favour recall over precision | Lost client and fees vs a few minutes of advisor time (STEP 1, A.3); the baseline at 0.5 misses 40% of the >50K prospects | Fixed advisor capacity (then precision / number of calls drives the threshold) | STEP 4 (threshold choice) |

### Notes for STEP 4
At t = 0.5 the baseline misses 40% of the >50K prospects (3,785 of 9,345 on the development set). Decision 26: the
threshold is chosen on OOF predictions with an explicit business cost where a false negative costs more than a false
positive (Session 6); the cost ratio is set and justified in STEP 4. Compare on the same folds: threshold tuning on OOF predictions, then `class_weight="balanced"`, then SMOTE inside an `imblearn` Pipeline
(training folds only). PR-AUC is not expected to change with the threshold; precision, recall and F1 will.

---

## STEP 4 - Class imbalance (lead: Clément)
Details, numbers and figures: `notebooks/04_class_imbalance.ipynb`. Code: `src/imbalance.py`, `src/evaluate.py`
(`threshold_table`, `best_threshold`), `src/models.py` (`class_weight`, `smote` options). Settings: `config.yaml`,
section `imbalance`. Tests: `tests/test_imbalance.py`.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 27 | Business cost = 5 x missed wealthy prospects + 1 x useless calls (X = 5) | Useless call ~ EUR 50 (30 min of advisor time); contacted wealthy prospect ~ EUR 300 expected value (10% x 1% fees on EUR 300k, first year only); ratio ~ 6 rounded down | Different business numbers; fixed advisor capacity | `config.yaml` (`imbalance`) |
| 28 | The threshold minimises the business cost on out-of-fold predictions: **0.17** for the baseline | Course rule: no universal threshold, chosen on development data, never on the test set (S6); flat minimum (0.13-0.20 within 1% of the lowest cost) and stable per fold (0.12-0.19) | - | `best_threshold`, `src/imbalance.py` |
| 29 | Remedy kept: **threshold tuning only** (no class weights, no SMOTE) | Same PR-AUC; almost the same business cost (class weights 0.7% cheaper, consistently on 5/5 folds; SMOTE no gain), but class weights and SMOTE worsen log-loss (0.328 -> 0.40) and SMOTE creates unrealistic one-hot rows; least invasive (S6 hierarchy). The deciding reason is the probabilities, not the cost | Another model where a remedy clearly lowers the cost | `config.yaml` (`imbalance.remedy`) |
| 30 | Sensitivity X = 2, 3, 5, 10: same remedy conclusion for every X; marginal price 3.5 / 5.4 / 7.4 calls per extra wealthy client | Shows the decision does not hinge on the exact assumption | - | notebook 04, Part E |
| 31 | Result (development, out-of-fold): wealthy found 60% -> 90%, calls 20% -> 43% of prospects, precision 72% -> 51%, business cost -40%; PR-AUC unchanged (0.745) | Threshold changes the operating point, not the ranking (S6) | Test-set confirmation (STEP 6) | notebook 04, Part F |
| 32 | Honest check of the threshold choice: threshold chosen on 4 folds, cost measured on the 5th (cross-validated threshold) | Choosing and measuring on the same predictions is optimistic (like an in-sample backtest); measured optimism only ~0.6% (12,752 vs 12,681), thresholds 0.16-0.19 | - | `cross_validated_threshold`, notebook 04 F.2 |

### Notes for the next steps
- **STEP 5:** Random Forest and LightGBM are tuned **without** class weights / `scale_pos_weight`, and their threshold is chosen with the same business cost. Optionally, `scale_pos_weight` (S6) can be checked for LightGBM with the same rule.
- **STEP 6:** the threshold is recomputed for the final model on its own out-of-fold predictions (the 0.17 belongs to the baseline), then applied once to the test set.
- **STEP 7:** keeping the threshold rather than class weights preserves the probabilities, so calibration starts from honest probabilities.
- **STEP 9:** lowering the threshold to 0.17 doubles the number of people contacted; check the selection rate by sex and race on the final model.

---

## STEP 5 - Three models, Optuna tuning, MLflow (lead: Lucas)
Details, numbers and figures: `notebooks/05_models_tuning.ipynb`. Code: `src/models.py` (`build_random_forest`,
`build_lightgbm`, `ToCategory`, `build_model`), `src/tuning.py`, `src/tracking.py` (`nested` option). Settings:
`config.yaml`, sections `models` and `tuning`. Results: `reports/tuning/`. Tests: `tests/test_models.py`,
`tests/test_tuning.py`. Run: `python -m src.tuning` (about 30 min on 4 cores).

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 33 | Random Forest: the categorical encoding (one-hot or ordinal) is a hyperparameter chosen by Optuna; LightGBM: native categorical columns (pandas 'category') | sklearn's RF reads ordinal codes as ordered numbers; the better encoding depends on the other settings (Part B); LightGBM groups categories natively | - | `src/models.py`, `config.yaml` |
| 34 | Optuna (TPE sampler, seed 42, 10 random start trials, median pruner after 2 folds) maximises the mean PR-AUC over the 5 folds; 20 / 40 / 60 trials; search ranges widened until no best value sits on an edge | PR-AUC is the main metric (decision 1); the cost depends on the threshold, chosen afterwards (decision 28); pruning saves about a third of the time | A best value on the edge of a range | `src/tuning.py`, `config.yaml` (`tuning`) |
| 35 | Candidate final model: **LightGBM** (PR-AUC 0.752 vs 0.745, cost -2.4%, better on 5/5 folds), threshold 0.15 at X = 5 | Consistent paired gain over the tuned logistic regression; the Random Forest brings no regular gain | A paired comparison on new data that reverses the gain; the choice is then frozen before the test set (decision 39: the test set measures, it never chooses) | `reports/tuning/`, notebook 05 |
| 36 | LightGBM: threshold only, no `is_unbalance` / `scale_pos_weight` | Same PR-AUC, no regular cost gain (2 folds better, 3 worse), log-loss 0.32 -> 0.39 | - | notebook 05, Part E |
| 37 | Nested CV (5 outer folds, Optuna on 3 inner folds) confirms the scores: optimism at most 0.002 PR-AUC, same ranking | Requirement 6: the reported score must not be optimistic (S5) | - | `nested_cv`, notebook 05, Part F |
| 38 | Every Optuna trial (complete or pruned) is an MLflow child run under one parent run per model, plus a final run and a nested-CV run per model | Requirement 7: every run, parameters, metrics and artifacts, for all three models | - | `src/tuning.py`, `src/tracking.py` |

### Results (development set, same 5 folds as STEPS 1-4)
| Model | PR-AUC | ROC-AUC | Log-loss | Threshold (X = 5) | Recall | Precision | Cost | Honest cost | Nested PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression (C = 3.5) | 0.745 | 0.901 | 0.328 | 0.17 | 90.5% | 50.7% | 12,674 | 12,748 | 0.745 |
| Random Forest (one-hot, 350 trees, depth 20) | 0.746 | 0.903 | 0.326 | 0.18 | 91.0% | 50.3% | 12,610 | 12,679 | 0.744 |
| LightGBM (350 rounds, lr 0.024, 21 leaves) | **0.752** | **0.905** | **0.323** | 0.15 | 91.6% | 50.3% | **12,376** | **12,490** | **0.752** |

Paired differences with the tuned logistic regression (fold by fold): LightGBM +0.0074 +/- 0.0037 PR-AUC (better on
5/5 folds), cost -60 per fold (lower on 5/5 folds); Random Forest +0.0014 PR-AUC (4/5), cost -13 per fold (3/5).

### Notes for the next steps
- **STEP 6:** refit the LightGBM Pipeline with `reports/tuning/best_params.yaml` on the whole development set and
  evaluate it **once** on the test set, with the tuned logistic regression as the reference. The threshold (0.15)
  must be **applied by the delivered model** (for example `FixedThresholdClassifier`, scikit-learn >= 1.5, or the
  threshold saved next to the Pipeline); otherwise `predict()` still answers at 0.5.
- **STEP 6:** also report the "amounts" reference (exact capital amounts, decision 7): the tree models gain little
  here partly because these amounts were removed.
- **STEP 7:** LightGBM keeps unweighted probabilities (decision 36), so calibration starts from its out-of-fold
  predictions in `reports/tuning/oof_predictions.csv`.
- **Early stopping (Session 4)** is not used: Optuna tunes `n_estimators` together with `learning_rate`, which
  plays the same role (small learning rate + more rounds, Part C.3 of notebook 05). Early stopping would need a third
  split inside each training fold; it is an option if the training time becomes a problem.
- **Nested CV budget:** the nested CV uses fewer trials (10 / 15 / 20) and 3 inner folds, so it estimates a slightly
  cheaper version of the tuning procedure; with a flat top of the search space (Part C.3) the difference is small.
- **Decision 29 (STEP 4):** with the same paired comparison as here, class weights lowered the cost on 5/5 folds
  (about -17 per fold, 0.7%) for the logistic regression. "Threshold only" stays the right choice, but the reason is
  the probabilities (log-loss 0.33 -> 0.40), not "same cost".

---

## STEP 6 - Final model, evaluated once on the test set (lead: Eliott)
Details, numbers and figures: `notebooks/06_final_test_evaluation.ipynb`. Code: `src/final.py`. Settings: `config.yaml`,
section `final`. Results: `reports/final/`. Tests: `tests/test_final.py`. Run: `python -m src.final` (about 1 minute).

Course rule (Session 5, "CV tuning + test set"): tune with CV -> freeze the modelling choices -> evaluate once on the
untouched test set.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 39 | Final model frozen **before** opening the test set: LightGBM with the STEP 5 hyperparameters. The logistic regression is measured on test as a reference only. Replaces the fallback rule of decision 35 ("no gain on test -> keep the logistic regression") | Choosing between models on the test set would turn it into a selection tool and make the reported score optimistic (S5) | - | `config.yaml` (`final`), `src/final.py` |
| 40 | The threshold is part of the delivered model: call when P(>50K) >= **0.15** (STEP 4 rule, X = 5, on LightGBM's own out-of-fold predictions); stored in `reports/final/final_model.yaml` | `predict()` answers at 0.5: on test it would call 2,051 people instead of 4,238 and miss far more wealthy prospects (S6) | - | `predict_decision` |
| 41 | Final Pipeline refitted on the whole development set (80%), then evaluated **once** on the test set | Each CV model saw only 4/5 of the development set; nothing is changed after the test measure | - | `run_final` |
| 42 | Test results: LightGBM PR-AUC **0.763**, ROC-AUC 0.907, log-loss 0.319; finds 2,143 of 2,336 wealthy (91.7%), calls 43% of people, precision 50.6%, cost 3,060. Logistic regression: 0.750 / 0.902 / 0.326, cost 3,114 (+1.7%) | Same metrics as STEPS 3-5 (requirement 4) | - | `reports/final/test_results.csv` |
| 43 | CV estimates were honest: test PR-AUC inside CV mean +/- 2 std for both models (+1.2 and +0.4 std), same ranking, same recall | No leakage, no overfitting of the 5 folds by Optuna. Test slightly above CV: the final model is trained on 100% of the development set (each CV model on 80%), plus sampling noise of one test set. The LightGBM advantage rests on the paired CV comparison (better on 5/5 folds, decision 36); the test only confirms it | - | `reports/final/dev_vs_test.csv` |
| 44 | Cost of decision 7 measured on test: exact capital amounts give PR-AUC 0.832 and cost 2,775 (-9%); indicators kept | Same-year amounts not available before contact (decision 7) | Bank receives the previous year's tax return | notebook 06, Part E |

### Notes for STEPS 7 to 9
- **STEP 7 (calibration):** test probabilities of the final model in `reports/final/test_predictions.csv` (column
  `lightgbm`); reliability curve + Brier score from them, no refit needed.
- **STEP 8 (SHAP):** rebuild the final model with `fit_on_development(cfg, "lightgbm", load_best_params(cfg, "lightgbm"), X_dev, y_dev)`.
- **STEP 9 (fairness):** test decisions = `test_predictions.csv["lightgbm"] >= 0.15`; `sex` and `race` are in `X_test` (same index).

---

## STEP 7 - Calibration diagnostics of the final model (lead: Eliott)
Details, numbers and figures: `notebooks/07_calibration.ipynb`. Code: `src/calibration.py`. Settings: `config.yaml`,
section `calibration`. Results: `reports/calibration/`. Tests: `tests/test_calibration.py`. Run: `python -m src.calibration`
(a few seconds, no refit: reads the probabilities saved in STEPS 5 and 6).

Course rules (Session 7): ranking and calibration are different questions; reliability diagram + Brier + ECE together;
learn any calibration map on predictions the model did not train on; compare before / after on unseen data; check segments.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 45 | Diagnosis on the final model's test probabilities (STEP 6): reliability diagram (10 equal-width bins, counts, histogram), Brier score, ECE; the development out-of-fold curve is shown next to it | Requirement 9; a point built on few people is weak evidence (S7) | - | `src/calibration.py` |
| 46 | **No recalibration.** Platt and isotonic are learned on development out-of-fold predictions and compared with the same 5 folds (map learned on 4 folds, measured on the 5th); rule: keep one only if it lowers the Brier score on >= 4/5 folds. Brier rather than ECE because the ECE depends on the binning. Both do so on 1/5 folds (isotonic lowers the ECE on 3/5 only); Platt learns A = 1.02, B = 0.01 (nothing to correct); isotonic lowers PR-AUC (0.752 -> 0.743) | Same "regular gain" logic as STEPS 4-5; the test set does not choose | Data shift (new population) | `choose_method`, `config.yaml` |
| 47 | Result on test: Brier **0.102** (constant 23.9% model: 0.182), ECE **0.013** (development: 0.007), mean predicted 24.1% vs 23.9% observed. No systematic over- or under-confidence; local test gaps (45-55% bins over-estimated by 4-6 points, 85% bin under-estimated by 5 points) under 2 points on development and in opposite directions. Probabilities trusted as they are for the 0.15 decision | LightGBM trained on log-loss, no class weights / SMOTE (decisions 29, 36) | 1994 data: re-check on recent data before real use | notebook 07, Parts B-E, H |
| 48 | Cross-check of the threshold: with calibrated probabilities, call when 1 x (1 - p) <= 5 x p, i.e. p >= 1/6 = 0.167, consistent with 0.15 found on LightGBM's out-of-fold predictions (cost within 1% of the minimum from 0.12 to 0.18); 0.15 kept | S7 link between calibrated probabilities and business costs | - | `bayes_threshold` |
| 49 | Calibration by sex on test: men 31.0% predicted vs 30.4% observed, women 10.8% vs 11.5% (small, opposite errors) | S7: overall calibration can hide segment errors | - | notebook 07, Part G |

### Note for STEP 9
Women's probabilities are slightly too low on average: check whether wealthy women are found as often as wealthy men
at the threshold 0.15 (recall by sex), and discuss calibration within groups as one of the fairness views.

---

## STEP 8 - Diagnostics and explainability of the final model (lead: Clément)
Details, numbers and figures: `notebooks/08_explainability.ipynb`. Code: `src/explain.py`. Settings: `config.yaml`,
section `explain`. Results: `reports/explain/`. Tests: `tests/test_explain.py`. Run: `python -m src.explain` (about 2 minutes).

Course rules (Session 8): a global score hides where the model fails -> slice-based evaluation with size and prevalence,
weakest segment defined by the decision metric, then error analysis; gain and permutation importance answer different
questions; SHAP explains the model (global and local) on a stated scale and is not causal; PDP shows the average learned
relationship and can query unrealistic points; the model card names the weakest segment.
The model is the frozen final model of STEP 6; the test set only describes it, nothing is chosen here.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 50 | 19 meaningful slices on the test set (age bands, diploma, married / not married, hours bands, sex) with n, prevalence, found / missed, precision, recall, PR-AUC at 0.15. Weakest = lowest **recall** among slices with >= 100 wealthy people -> **not married**: recall 67% (224 of 333 found), 109 of the 193 misses (56%) | A miss costs 5 calls (X = 5); tiny slices are noisy (S8). 17-25 year olds (recall 47%, 45 wealthy) and no high-school diploma (51%, 75 wealthy) shown but not conclusive | Larger test set | `slice_table`, `weakest_slice` |
| 51 | Error analysis of the weakest slice: missed wealthy single people have less education (10.4 vs 13.1 years), fewer hours (41 vs 49), rarely a manager / professional job (28% vs 76%) or a capital gain (13% vs 42%); mean predicted 7%. Reported, not corrected (the model is frozen; a lower threshold for 94%-non-wealthy single people would multiply useless calls) | S8: inspect errors in the weakest slice | - | `error_analysis` |
| 52 | Global importance reported with **permutation importance** on the test set (PR-AUC drop, 20 shuffles, raw columns): education-num 0.094, age 0.077, occupation 0.058, relationship 0.056, capital-gain 0.055; `sex`, `race`, `fnlwgt`, `education` = 0. Gain shown for comparison (relationship 1st with 29% of gain); `relationship` + `marital-status` read as one household signal (correlated substitutes) | Measured on new data with our metric; gain is a training statistic (S8) | - | `permutation_table`, `gain_importance` |
| 53 | SHAP with TreeExplainer on the log-odds scale (additivity checked, max error 1.8e-14): mean |SHAP| (relationship 0.68, age 0.61, marital-status 0.59, education 0.44, occupation 0.39), beeswarm for numerical inputs, mean SHAP per category; three waterfalls chosen by a fixed rule (clear call ~90%, borderline call at 0.15, median missed wealthy single person) | Requirement 8; S8 warns to state the output scale | - | `shap_values`, `local_cases` |
| 54 | PDP (2,000 development rows) for age (bell shape, peak about 29% at 47-52), education (rises from 9 years) and hours (rises from 35 to 50 hours) | S8: shape of the learned relationship; same bell as STEP 1 | Unrealistic combinations (S8) | notebook 08 F |
| 55 | Drivers make business sense (career stage, education, job, hours, investments). Household signal: plausible for 1994 data, partial proxy for sex, and source of the main weakness. All explanations read as model behaviour, not causes. Model card in notebook 08, Part H | S8 | Use outside the 1994 US census population | notebook 08 G-H |

---

## STEP 9 - Fairness and bias audit (lead: Clément)
Details, numbers and figures: `notebooks/09_fairness.ipynb`. Code: `src/fairness.py`. Settings: `config.yaml`,
section `fairness`. Results: `reports/fairness/`. Tests: `tests/test_fairness.py`. Run: `python -m src.fairness` (about 2 minutes).

Course rules (Session 9): measure by group -> choose a fairness definition from the cost of each error, before looking at the
results -> mitigate (pre-, in- or post-processing) -> re-audit on held-out data; report the trade-off, not only one point.
"Before" = the STEP 6 model on the test set; the mitigation is chosen on the development set only; "after" is measured once on test.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 56 | Audit with Fairlearn `MetricFrame` by `sex` at the business threshold: size, base rate, selection rate, TPR (recall), FPR, PPV (precision); one number per definition (recall gap, equalized-odds gap, selection-rate ratio, precision gap); 95% bootstrap interval for the recall gap (1,000 resamples of the test set) | S9 audit table; bootstrap idea from S4 to show the gap is not noise | - | `group_metrics`, `gap_summary`, `bootstrap_tpr_gap` |
| 57 | Definition prioritised: **equal opportunity** (equal recall for wealthy women and wealthy men), chosen from the cost structure before the results: the costly error is a missed wealthy prospect (X = 5). Demographic parity (ignores who is wealthy; base rates 11.5% vs 30.4%), equalized odds and predictive parity are reported, not targeted | S9: choose the definition from the cost of the decision, before seeing what the model satisfies | A bank policy based on equal access (demographic parity) | notebook 09 A |
| 58 | Before (test): recall 81% women (310 of 383) vs 94% men (1,833 of 1,953), gap **12.9 points** (95% interval 8.8-16.7); development out-of-fold gap 10.7; selection-rate ratio 0.35 (four-fifths rule fails); precision 48% vs 51% (predictive parity almost holds, as the STEP 7 calibration by sex suggested) | Same model, same threshold as STEP 6 | - | `reports/fairness/metrics_by_sex_test.csv` |
| 59 | Source of the gap: **proxy variables** (marital status, household role) and historical bias, not `sex` itself (not an input): inside each marital group the development gap is 0.3-2.9 points (married 93.8% vs 96.7%, not married 63.4% vs 63.7%) | S9: fairness through unawareness fails when other inputs reconstruct the attribute; links to STEP 8 (weakest segment = not married) | - | `recall_within_strata`, notebook 09 C |
| 60 | `race`: reported by group with sizes, no mitigation (recall 85-92%; only 10-113 wealthy people in the minority groups of the test set) | Decision 10: tiny groups give noisy metrics | Larger data | `reports/fairness/metrics_by_race_test.csv` |
| 61 | Mitigation, **in-processing**: training loss reweighted by label and group, weight `lam` on wealthy women, 1 elsewhere; same Pipeline and hyperparameters; one threshold for everybody (re-chosen with the STEP 4 rule). `lam` chosen on the development set with the same 5 folds: smallest `lam` whose out-of-fold recall gap is <= 3 points -> **lam = 6** (gap 2.5, cost +4.1%, PR-AUC 0.752 -> 0.736, threshold 0.24). Not post-processing (`ThresholdOptimizer` needs `sex` at decision time, excluded by decision 8); not `ExponentiatedGradient` (random mixture of models, constraint checked on training predictions) | S9 in-processing loss w(y, a); `sex` used only to build training weights; cross-validated like every other choice | A different gap target (business / ethical choice; the trade-off curve gives the price of each option) | `tradeoff_curve`, `choose_lambda`, `config.yaml` (`fairness`) |
| 62 | After (test, measured once): recall 87% women (335) vs 92% men (1,792), gap **4.3 points** (95% interval 0.9-7.5, no overlap with before); +25 wealthy women found, -41 wealthy men; cost 3,060 -> 3,221 (+5.3%); PR-AUC 0.763 -> 0.753; ECE 0.013 -> 0.062 (probabilities no longer calibrated); precision gap 3 -> 12 points (the S9 theorem: with different base rates, equal opportunity moves away from predictive parity) | S9: report fairness and operating metrics together, disclose what was traded away | Small test group of wealthy women (383): one point of their recall = 4 people | `reports/fairness/before_after_test.csv` |
| 63 | Recommendation: the mitigated model, because equal opportunity matches the harm we care about and its price is moderate and disclosed; conditions: read its probabilities as a ranking (or recalibrate on out-of-fold predictions, STEP 7 method), and compliance approval for collecting `sex` for training only. The STEP 6 model stays the reference for STEPS 6-8 | S9: the operating point is a business and ethical decision | Compliance refuses the use of `sex` in training | notebook 09 G-H |
