# Decisions log

Every methodological decision, with its reason and what could invalidate it.
Format of the course: what we did, why, and what could invalidate it. This file feeds the report.

---

## STEP 1 - Problem framing, EDA, leakage audit, split (lead: Clément)
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
| 7 | Replace capital amounts by `has_capital_gain` / `has_capital_loss` | Same-year amounts not known at prediction time; values behave like codes (122 distinct values, all-or-nothing outcomes); exact amounts add +0.08 PR-AUC we cannot justify (0.825 vs 0.748 vs 0.714 without capital, quick CV check) | Bank receives the same-year tax return before deciding | STEP 2 Pipeline; compared in STEP 6; STEP 8 |
| 8 | `sex` and `race` are not model inputs; kept aside for the audit | Targeting on sex or race is illegal for a French / EU bank | - | STEP 2 Pipeline, STEP 9 |
| 9 | `relationship`: merge `Husband` and `Wife` into `Spouse` | Keeps the household role (Wife 46.9% vs Husband 44.9%: same signal), removes the most direct sex proxy | Large performance loss (checked in STEP 9) | STEP 2 Pipeline |
| 10 | Fairness audit: `sex` main attribute (metrics + mitigation); `race` per group with sizes, no mitigation, no merging of groups | Small race groups (~80-95 people in test) give noisy metrics; merging would hide opposite effects | - | STEP 9 |
| 11 | 80/20 stratified split, seed 42; 5-fold StratifiedKFold in development; test used once | Same prevalence everywhere; no time order (cross-section) and no group identifier | A timestamp or household identifier | `src/data.py` |

### Notes for STEP 2
`has_capital_gain` / `has_capital_loss`; age bands (bell-shaped relation, for the linear model); hours bands;
group rare `native-country` values (38 of 41 countries below 0.5%).

---

## STEP 2 - Feature engineering (lead: Eliott)
Details, numbers and figures: `notebooks/02_feature_engineering.ipynb`. Code: `src/features.py`.
Settings: `config.yaml`, section `features`. Tests: `tests/test_features.py`.

### Decisions
| # | Decision | Why | What could invalidate it | Where |
|---|---|---|---|---|
| 12 | Two-layer Pipeline: stateless `AdultFeatureBuilder` (row rules), then a learned `ColumnTransformer` (imputation, encoding, scaling) | Anything learned from data must be fitted on training folds only; production receives raw rows | - | `src/features.py` |
| 13 | Two preprocessors: `linear` (one-hot, scaling, bands) and `tree` (ordinal codes, no scaling, no bands) | Linear models cannot cut variables into ranges and are scale-sensitive; trees can and are not | - | `build_preprocessor(cfg, family)` |
| 14 | Implements decisions 3, 4, 7, 8, 9: `fnlwgt`, `education`, `sex`, `race` and the capital amounts never reach the model; `Husband`/`Wife` -> `Spouse` (45.1% >50K) | STEP 1 | - | `AdultFeatureBuilder` |
| 15 | Missing categorical -> `"Missing"` (decision 5); median imputer on numbers as a safeguard | Missingness informative (9.5% vs 24.8%); a production row may be incomplete | Missingness caused by another process in production | `ColumnTransformer` |
| 16 | One-hot for linear, categories < 0.5% of training rows grouped (38 countries = 5.9% of rows), unknown categories tolerated; ordinal codes for trees (`categorical_mask` for HistGradientBoosting); no target encoding | No order between categories; rare levels give noisy coefficients; target encoding adds a leakage risk for no need (max 42 levels) | Rare countries carrying signal (checkable in STEP 4) | `ColumnTransformer` |
| 17 | Engineered: `has_capital_gain`, `has_capital_loss` (all models); `age_band` (6), `hours_band` (4), linear only; no `net_capital` | Indicators: decision 7. Bands: bell-shaped age, stepped hours. A loss signals wealth (50.0% >50K), a net value would reverse it | Linear model still underfitting age (-> spline) | `AdultFeatureBuilder`, `config.yaml` |
| 18 | `capital` option: `indicators` (default), `amounts` (log1p for linear, raw for trees), `none` | STEP 6 must rebuild the "exact amounts" reference with the same code | - | `config.yaml`, `build_preprocessor` |
| 19 | Evidence (default logistic regression, 5-fold CV, PR-AUC): 0.688 no capital, no band -> 0.729 indicators -> 0.745 with both bands; bands better on 5/5 folds | A feature is kept only if it improves the honest score | - | notebook 02, Part F |

### Limitation
Band edges were chosen after the STEP 1 EDA, which looked at all rows. Edges are round and domain-based (life stages,
part-time below 35 hours), not tuned on the target, and the shape is re-checked on the development set only.

### Notes for STEP 3 and STEP 4
- Class weights or resampling go **after** the preprocessor (resampling requires an `imblearn` Pipeline).
- Logistic regression: `build_preprocessor(cfg, "linear")`; trees: `build_preprocessor(cfg, "tree")`;
  `HistGradientBoostingClassifier(categorical_features=categorical_mask(fitted_pre))`.
- Linear reference to beat: PR-AUC 0.745 (default logistic regression).
- STEP 8 (SHAP): with one-hot, sum SHAP values per original variable for the global view.
