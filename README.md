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
config.yaml          all settings (seed, split ratios, paths, later: search spaces)
requirements.txt     pinned Python libraries
DECISIONS.md         log of every methodological decision (feeds the report)
src/                 main implementation (Python modules, one per pipeline stage)
  data.py            STEP 1: load data, remove exact duplicates, stratified 80/20 split
  features.py        STEP 2: feature builder + preprocessing Pipeline (linear / tree)
  evaluate.py        STEP 3: metrics (PR-AUC, ROC-AUC, log-loss, precision, recall, F1) + cross-validation
  models.py          STEP 3: baseline logistic regression Pipeline
  tracking.py        STEP 3: MLflow experiment tracking
  imbalance.py       STEP 4: class-imbalance remedies (threshold, class weights, SMOTE) + business cost
notebooks/           supplementary material: exploration and figures, one notebook per step
tests/               automated checks (pytest)
reports/figures/     figures used in the report
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
mlflow ui               # browse the logged runs at http://127.0.0.1:5000
pytest                  # run the tests
```

## Results
To be completed (STEP 6 onwards).

## Use of AI assistants
To be completed in the report (where and how AI assistants were used).
