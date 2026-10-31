# Healthcare ML Lab

The original portfolio dashboard remains at `index.html`. The new lab is at [lab/index.html](../lab/index.html).

## What is new

- Three real, browser-executable models: linear regression, decision tree and gradient boosting.
- Scenario controls, model comparisons and age/BMI sensitivity curves.
- One-at-a-time feature replacement explanations, explicitly distinct from SHAP.
- Holdout density, residual histograms, permutation importance and grouped error tables.
- Local CSV validation, batch predictions and downloadable results.
- Reproducible Python training and independent Python/JavaScript inference tests.

## Reproduce the included experiment

Use Python 3.11+ and Node 20+ for the test commands.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download `insurance.csv` from this pinned [public source revision](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/d20658ec6d336af2d4ddb5fd72b6f677dd46136e/insurance.csv) and save it under `data/insurance.csv`. The raw CSV is not redistributed in this repository. Consult its source for dataset usage terms.

```bash
python pipeline.py --data data/insurance.csv --source "Public insurance dataset"
python -m http.server 8000
```

Open `http://localhost:8000/lab/`. The browser needs HTTP to load the JSON model export; opening the HTML using `file://` is insufficient.

Default seed: 42. Test fraction: 0.2. Duplicate source records are removed before splitting. Categorical encoding is female=0/male=1 and no=0/yes=1. Inputs are age, sex, BMI and smoking status. The target is log10(charges). Models are trained on the training split only.

The lab uses a newly executed experiment. It matches the historical dashboard's rounded R² results on this source, but does not claim to reproduce every original dissertation artifact. The JSON records input-file SHA-256 and package versions; newline/BOM differences change the file hash. Included exports were generated with scikit-learn 1.8.0. Requirements allow a compatible range; exact numerical results may vary by version.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
node --test tests/lab.test.js
```

The pipeline independently verifies exported model predictions against scikit-learn on every input row before writing a report. The committed JavaScript fixtures cover continuous inputs, categorical combinations and tree-threshold boundaries. The Python tests use generated fixtures only, not medical records.

After retraining, regenerate cross-runtime fixtures:

```bash
python tests/make_parity.py
node --test tests/lab.test.js
```

## Interpretation

R² and MSE are evaluated on log10(charges). Dollar MAE uses back-transformed predictions. Simply exponentiating a log prediction does not yield an unbiased expected dollar cost.

Sensitivity curves vary a single input with all other inputs held constant. Feature replacement effects compare the current estimate with an estimate using one training reference value. Effects overlap, need not add to the prediction and are neither causal effects nor SHAP values.

Permutation importance measures the reduction in holdout R² after shuffling one feature. The lab displays the mean and standard deviation across 10 permutations. It is model reliance, not a causal or clinical ranking.

Grouped errors are descriptive and depend on subgroup sample sizes. A single holdout split provides limited evidence; external validation, repeated evaluation and subgroup review would be needed for stronger generalisation claims.

## Local CSV explorer

Required columns: `age,sex,bmi,smoker`. Optional `charges` must be positive if included. Age must be a whole number inside the trained range, and BMI must be finite and inside its trained range. Sex values are female/male; smoker values are no/yes. Headers and categorical values are case-insensitive. Extra columns are ignored.

The browser processes up to 5 MB and 10,000 records locally, retaining duplicate model rows and rejecting invalid records. Invalid-row counts and the first 10 error messages are shown. Export includes every valid record, its record number and three predictions. Inputs are never transmitted; the static page makes only its initial asset requests.

## Source documentation

- [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [Permutation importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)
