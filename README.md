# QSAR/QSPR Modeling Studio

A Streamlit application for building, validating, saving, loading, and visualizing QSAR/QSPR regression models from Excel workbooks.

## Features

- Excel upload with sheet selection for descriptors `X` and endpoint `y`
- Endpoint column selection and row alignment by workbook row index or first-column sample IDs
- Optional SMILES column selection for structure lookup and hover context in PCA/Williams plots
- Automatic removal of endpoint leakage columns and common report/model artifact columns accidentally present in descriptor sheets
- Dataset preview, missing-value diagnostics, endpoint statistics, and descriptor statistics
- Exploratory PCA screening before modeling with eigenvalues, explained variance, selectable PC axes, hoverable sample IDs, and outlier exclusion history
- Leakage-aware preprocessing fitted on the training set only
- Endpoint transformations: none, `log10(y)`, and `-log10(y)`
- Random split with optional binned-y stratification
- Sorted endpoint split that keeps minimum and maximum endpoint samples in training
- Scaling options: none, `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- Regression models: MLR, PCR, PLS, SVR, Random Forest
- Feature selection: none, manual, variance threshold, SelectKBest, RFE, and a built-in genetic algorithm
- Cross-validation, train/test validation metrics, ranked model table, and diagnostic warnings
- Single-split GA candidate model search with different descriptor subsets, progress feedback, and configurable top-N retention
- Plots: endpoint histogram, interactive observed vs predicted with sample ID hover, residuals, CV scores, interactive Williams plot with sample ID hover, distance-based AD, PCA AD, PCR variance, RF importance, GA progress
- Export of reports, predictions, selected descriptors, plots, and complete joblib model bundles
- Exported reports include samples excluded during PCA/outlier review
- Load saved models and predict new compound descriptor workbooks with applicability-domain assessment

## Project Structure

```text
app.py
modules/
  applicability_domain.py
  data_loader.py
  evaluation.py
  export.py
  feature_selection.py
  model_io.py
  models.py
  plots.py
  pca_screening.py
  preprocessing.py
  splitting.py
requirements.txt
tests/
  smoke_test.py
```

## Run Locally

Fast path on macOS/Linux:

```bash
./run_local.sh
```

Then open:

```text
http://127.0.0.1:8501
```

Manual setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If your shell does not have `python`, use `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional structure rendering from SMILES uses RDKit:

```bash
.venv/bin/python -m pip install -r requirements-rdkit.txt
```

Without RDKit the app still works and shows SMILES text, but molecule images are not rendered.

## Data Format

Use an Excel workbook with one sheet containing molecular descriptors and one sheet containing the endpoint. The descriptor sheet should contain mostly numeric descriptor columns. The endpoint sheet may contain multiple columns; choose the response column in the app.

If compounds have IDs, place them in the first column of both sheets and enable `Use first column as sample ID`. Otherwise, rows are aligned by their Excel row order after empty rows and columns are removed.

## Scientific Notes

Preprocessing that depends on descriptor distributions is fitted only on `X_train` and then applied to `X_test` and future prediction data. Scaling is part of the sklearn model pipeline, so cross-validation refits the scaler inside each fold. The test set remains external to model fitting and descriptor-selection fitting.

The Williams plot uses leverage from the selected descriptor matrix and standardized residual limits of `+/- 3`. The distance-based applicability-domain plot is an Insubria-style proxy based on distance to the training-set centroid in standardized descriptor space.

## Smoke Test

```bash
python tests/smoke_test.py
```
