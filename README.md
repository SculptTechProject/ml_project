# California Housing Regression

Minimal end-to-end regression pipeline on the California Housing dataset.

## Features
- Load California Housing dataset from scikit-learn.
- Expand features using degree-2 polynomial features (no bias).
- Split into train and test sets.
- Grid-search over `learning_rate` and `max_iter` for `HistGradientBoostingRegressor`.
- Compute R² and RMSE (manual fallback for compatibility).
- Save all trained models to `models/` directory (cleared on start).
- Identify best model and plot `y_test vs y_pred`.
- Export run summary to `training_results_summary.csv`.

## Quickstart

```bash
# create and activate a virtual environment (example uses Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate

# install required packages
pip install scikit-learn joblib pandas matplotlib

# run training (assumes the main script is named train.py)
python train.py
```

## Output structure
- `models/`: cleaned at the start of each run; contains saved `.joblib` models for each parameter combination.
- `training_results_summary.csv`: CSV with metrics (learning_rate, max_iter, R², RMSE) sorted by R².
- Best model path is printed to stdout during execution.

## Example: load and predict with a saved model
```python
from joblib import load
best = load("models/model_iter250_lr0_1.joblib")  # adjust name to the real best
# input must be transformed the same way (PolynomialFeatures degree=2, include_bias=False)
# assume `x_new_poly` is prepared:
pred = best.predict(x_new_poly)
```

## Suggested commit message
```
feat: add regression pipeline with poly features, grid search, visualization, and model persistence
```
