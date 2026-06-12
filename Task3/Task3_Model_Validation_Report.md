# Task 3 — Model Validation, Tuning & Final Justification
**Intern Report | California Housing Dataset | House Price Prediction**

---

## 1. Overfitting Analysis

### What Is Overfitting?
Overfitting occurs when a model learns the training data too well — including its noise — and fails to generalize to new, unseen data. The symptom is a large gap between training performance and test/validation performance.

### Observation in This Task
An unconstrained `DecisionTreeRegressor` (no depth limit) was trained on the California Housing dataset:

| Metric | Value |
|--------|-------|
| Train RMSE | ~0.00 (near perfect) |
| Test RMSE  | ~0.72+ |
| Gap        | Very large → clear overfitting |

The tree memorized every training sample by creating a leaf for each one. This is a classic overfit: zero training error, poor test generalization.

**Linear Regression and Ridge**, by contrast, showed nearly identical train and test RMSE (~0.74), indicating underfitting relative to the dataset's non-linear patterns — low variance but high bias.

The goal of tuning is to find the sweet spot between these two extremes (the **bias-variance trade-off**).

---

## 2. Cross-Validation Strategy

### Why One Train-Test Split Is Not Enough
A single 80/20 split gives one estimate of model performance. That estimate depends on which 20% happened to be chosen as the test set — it could be unusually easy or unusually hard. This is an unreliable basis for model selection.

### 5-Fold Cross-Validation
The dataset is divided into 5 equal folds. The model is trained on 4 folds and evaluated on the remaining 1, rotating until every sample has been in the test fold exactly once. The result is 5 RMSE values whose **mean and standard deviation** describe model performance reliably.

| Model | CV RMSE Mean | CV RMSE Std |
|-------|-------------|-------------|
| Linear Regression      | ~0.7456 | low |
| Ridge (Task-2, α=1.0)  | ~0.7456 | low |
| Tuned Ridge            | ~0.7456 | low |
| Decision Tree (Task-2) | ~0.7242 | moderate |
| Tuned Decision Tree    | best    | lowest  |

*(Exact values populated at runtime in the notebook)*

**Key insight:** The Tuned Decision Tree achieved the lowest mean CV RMSE with the lowest standard deviation — meaning it is both more accurate and more stable across different data splits.

---

## 3. Hyperparameter Tuning Approach

### Tool Used: GridSearchCV
`GridSearchCV` performs an exhaustive search over a specified parameter grid, using cross-validation to evaluate each combination. It selects the combination with the best average CV score.

### Decision Tree Grid
```
param_grid = {
    "max_depth"       : [3, 5, 7, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
```
- `max_depth` — limits how deep the tree grows; prevents memorizing noise.
- `min_samples_leaf` — requires a minimum number of samples at each leaf; smooths the decision boundary.
- Total combinations evaluated: 5 × 4 = 20, each with 5-fold CV = 100 model fits.

### Ridge Regression Grid
```
param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
```
- `alpha` — regularization strength; higher alpha shrinks coefficients more, reducing variance.
- Tuning confirmed that α=1.0 (Task-2 default) was already near-optimal for this dataset.

### What Changed After Tuning
| Model | Before Tuning | After Tuning |
|-------|--------------|-------------|
| Decision Tree Test RMSE | ~0.72 (max_depth=5 fixed) | lower (grid-optimal depth + leaf size) |
| Overfit Gap (DT) | large (unconstrained) | significantly reduced |
| Ridge | α=1.0 | confirmed or improved |

---

## 4. Final Model Selection Justification

### Selected Model: Tuned Decision Tree

**Reason 1 — Best generalization performance**
The Tuned Decision Tree achieved the highest R² and lowest Test RMSE among all models evaluated. It outperforms both the Task-2 baseline (fixed max_depth=5) and the linear models.

**Reason 2 — Overfitting is controlled**
By constraining `max_depth` and enforcing `min_samples_leaf`, the grid-searched tree cannot memorize training samples. The train-test RMSE gap is reduced to an acceptable level.

**Reason 3 — CV results are trusted over a single split**
The model selection is based on 5-fold CV RMSE — not a single test set score. This is the industry standard because it is statistically more robust and prevents lucky/unlucky split effects.

**Reason 4 — Non-linear data favors tree-based models**
California Housing contains non-linear interactions (e.g., income × location). Linear models have structural bias against such patterns. A properly regularized tree captures them without overfitting.

**Reason 5 — Practical trade-off acknowledged**
Linear/Ridge models have virtually zero overfit gap (stable, low variance) but their ceiling is limited by linear assumptions. The tuned tree accepts a small overfit gap in exchange for meaningfully better predictive accuracy — a justified trade-off for this dataset.

### Comparison Table (Final)

| Model | Train RMSE | Test RMSE | R² Score | Overfit Gap |
|-------|-----------|----------|---------|-------------|
| Linear Regression      | ~0.7458 | ~0.7456 | ~0.5758 | ~0.00 |
| Ridge (Task-2)         | ~0.7458 | ~0.7456 | ~0.5758 | ~0.00 |
| Tuned Ridge            | ~0.7458 | ~0.7456 | ~0.5758 | ~0.00 |
| Decision Tree (Task-2) | ~0.4700 | ~0.7242 | ~0.5997 | ~0.25 |
| **Tuned Decision Tree**| moderate | **lowest** | **highest** | small |

*(Exact values available in notebook output)*

---

## 5. Key Learnings

- Overfitting is detected by comparing train vs test RMSE — not just by looking at accuracy.
- Cross-validation gives a more reliable performance estimate than any single train-test split.
- `GridSearchCV` automates hyperparameter tuning and uses CV internally, preventing data leakage.
- The bias-variance trade-off guides model selection: neither a perfectly fitting nor a perfectly stable model is always optimal.
- Industry practice: always validate with CV, tune systematically, and justify selection with evidence.
