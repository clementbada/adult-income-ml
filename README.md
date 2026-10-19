# Adult Income: identifying high-income prospects for wealth management

**Course:** Supervised Learning & Model Engineering (MSc Data for Finance, Mines Paris PSL x Albert School)
**Dataset:** Adult / Census Income, OpenML `name="adult"`, `version=2` (48,842 people, 1994 US census)
**Group:** Clément, Eliott Monneraie, Lucas Stefani

## Business problem
A private bank wants to rank prospects by their probability of earning more than 50K USD per year, in
order to offer wealth-management services. The model outputs P(income > 50K | profile); a decision
threshold turns it into "contact / do not contact". Because the decision is about people, the project
includes a fairness audit on sex (main) and race (secondary).

## Project structure
```
config.yaml          all settings (seed, split ratios, paths, model defaults, Optuna search spaces)
requirements.txt     pinned Python libraries
DECISIONS.md         log of every methodological decision (feeds the report)
src/                 main implementation (Python modules, one per pipeline stage)
  data.py            STEP 1: load data, remove exact duplicates, stratified 80/20 split
  features.py        STEP 2: feature builder + preprocessing Pipeline (linear / tree)
  evaluate.py        STEP 3: metrics (PR-AUC, ROC-AUC, log-loss, precision, recall, F1) + cross-validation
  models.py          STEP 3 / 5: the three model Pipelines (logistic regression, Random Forest, LightGBM)
  tracking.py        STEP 3: MLflow experiment tracking
  imbalance.py       STEP 4: class-imbalance remedies (threshold, class weights, SMOTE) + business cost
  tuning.py          STEP 5: Optuna tuning of the 3 models, MLflow logging of every trial, comparison, nested CV
  final.py           STEP 6: refit the frozen final model on the development set, evaluate it ONCE on the test set
  calibration.py     STEP 7: reliability diagram, Brier score, ECE, Platt / isotonic check
  explain.py         STEP 8: slices and error analysis, gain vs permutation importance, SHAP, PDP
  fairness.py        STEP 9: fairness audit by sex (Fairlearn MetricFrame), in-processing mitigation, before / after
notebooks/           supplementary material: exploration and figures, one notebook per step
tests/               automated checks (pytest)
reports/figures/     figures used in the report
reports/tuning/      STEP 5 result tables (best hyperparameters, trials, comparison, nested CV, out-of-fold predictions)
reports/final/       STEP 6 test results, test probabilities (reused in STEPS 7 and 9), frozen final settings
reports/calibration/ STEP 7 calibration tables
reports/explain/     STEP 8 slices, importances, partial dependence tables
reports/fairness/    STEP 9 metrics by sex and race, trade-off curve, before / after table
data/                downloaded and split data (created by src/data.py, not committed)
```

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run
```bash
python -m src.data      # download the data, remove duplicates, create the dev/test split
python -m src.features  # fit every preprocessing variant on the dev set (sanity check)
python -m src.models    # STEP 3: cross-validate the baseline and log it in MLflow
python -m src.imbalance # STEP 4: compare the imbalance remedies and choose the threshold
python -m src.tuning    # STEP 5: tune and compare the 3 models (about 30-40 min on a laptop)
python -m src.tuning --n-trials 5 --skip-nested   # STEP 5 quick check (a few minutes)
python -m src.final     # STEP 6: refit the final model, evaluate it ONCE on the test set (~1 min)
python -m src.calibration  # STEP 7: reliability diagram, Brier, ECE, Platt / isotonic check (seconds)
python -m src.explain    # STEP 8: slices, gain vs permutation, SHAP, PDP (~2 min)
python -m src.fairness   # STEP 9: fairness audit, mitigation chosen on development, before / after on test (~2 min)
mlflow ui               # browse the logged runs at http://127.0.0.1:5000
pytest                  # run the tests
```

## Results
Final model: **LightGBM** (Optuna hyperparameters, threshold 0.15 chosen with the business cost on the development set),
evaluated once on the untouched test set (9,758 people, 23.9% at >50K):

| Model | PR-AUC | ROC-AUC | Log-loss | Recall | Precision | Business cost |
|---|---|---|---|---|---|---|
| LightGBM (final) | 0.763 | 0.907 | 0.319 | 91.7% | 50.6% | 3,060 |
| Logistic regression (reference) | 0.750 | 0.902 | 0.326 | 90.5% | 51.3% | 3,114 |

Details: `reports/final/`, `notebooks/06_final_test_evaluation.ipynb`, `DECISIONS.md` (STEP 6).

Calibration (STEP 7): Brier score 0.102 (constant model 0.182), ECE 0.013; no recalibration needed (Platt A = 1.02, B = 0.01).
Details: `reports/calibration/`, `notebooks/07_calibration.ipynb`.

Explainability (STEP 8): main drivers are education, age, household role, occupation, capital gain and hours;
`sex` and `race` have zero importance (removed by the Pipeline). Weakest segment: people who are not married
(recall 67% vs 92% overall). Details: `reports/explain/`, `notebooks/08_explainability.ipynb` (model card in Part H).

Fairness (STEP 9): definition prioritised = equal opportunity (equal recall for wealthy women and men). The final model
finds 81% of wealthy women vs 94% of wealthy men (gap 12.9 points), mainly through marital status and household role.
In-processing mitigation (training weight 6 on wealthy women, chosen on the development set): gap 4.3 points on test,
for +5.3% business cost, PR-AUC 0.753 and less calibrated probabilities. Details: `reports/fairness/`,
`notebooks/09_fairness.ipynb`.

## Use of AI assistants
To be completed in the report (where and how AI assistants were used).
