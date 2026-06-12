import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── Step 2: Load dataset ──────────────────────────────────────────────────────
data = fetch_california_housing(as_frame=True)
df   = pd.concat([data.data, data.target.rename("HousePrice")], axis=1)
print("Dataset shape:", df.shape)
print(df.head(3).to_string(), "\n")

# ── Step 3: Scale & Split ─────────────────────────────────────────────────────
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── Step 4: Overfitting detection ─────────────────────────────────────────────
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
train_rmse = np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))
test_rmse  = np.sqrt(mean_squared_error(y_test,  tree.predict(X_test)))

print("=" * 55)
print("STEP 4 - Overfitting Detection (Unconstrained Decision Tree)")
print(f"  Train RMSE : {train_rmse:.4f}")
print(f"  Test  RMSE : {test_rmse:.4f}")
print(f"  Gap        : {test_rmse - train_rmse:.4f}  <- large gap = overfitting\n")

# ── Step 5: Cross-Validation ──────────────────────────────────────────────────
def cv_rmse(model, X, y, cv=5):
    scores = cross_val_score(model, X, y,
                             scoring="neg_root_mean_squared_error", cv=cv)
    return -scores

models_baseline = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=1.0),
    "Decision Tree"     : DecisionTreeRegressor(random_state=42),
}

print("=" * 55)
print("STEP 5 - 5-Fold Cross-Validation (Baseline Models)")
print(f"{'Model':<22} {'CV RMSE Mean':>14} {'CV RMSE Std':>12}")
print("-" * 50)
for name, model in models_baseline.items():
    s = cv_rmse(model, X_scaled, y)
    print(f"{name:<22} {s.mean():>14.4f} {s.std():>12.4f}")
print()

# ── Step 6a: GridSearchCV — Decision Tree ────────────────────────────────────
print("=" * 55)
print("STEP 6a - GridSearchCV: Decision Tree (running...)")
dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid={"max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [1, 5, 10, 20]},
    scoring="neg_root_mean_squared_error",
    cv=5, n_jobs=-1
)
dt_grid.fit(X_train, y_train)
print("  Best params :", dt_grid.best_params_)
print("  Best CV RMSE:", round(-dt_grid.best_score_, 4), "\n")

# ── Step 6b: GridSearchCV — Ridge ────────────────────────────────────────────
print("=" * 55)
print("STEP 6b - GridSearchCV: Ridge Regression (running...)")
ridge_grid = GridSearchCV(
    Ridge(),
    param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    scoring="neg_root_mean_squared_error",
    cv=5, n_jobs=-1
)
ridge_grid.fit(X_train, y_train)
print("  Best params :", ridge_grid.best_params_)
print("  Best CV RMSE:", round(-ridge_grid.best_score_, 4), "\n")

# ── Step 7: Full evaluation table ────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred    = model.predict(X_te)
    tr_rmse = np.sqrt(mean_squared_error(y_tr, model.predict(X_tr)))
    te_rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2      = r2_score(y_te, pred)
    return {"Model": name, "Train RMSE": round(tr_rmse, 4),
            "Test RMSE": round(te_rmse, 4), "R2 Score": round(r2, 4)}

rows = [
    evaluate("Linear Regression",       LinearRegression(),                              X_train, y_train, X_test, y_test),
    evaluate("Ridge (Task-2)",           Ridge(alpha=1.0),                                X_train, y_train, X_test, y_test),
    evaluate("Tuned Ridge",              ridge_grid.best_estimator_,                      X_train, y_train, X_test, y_test),
    evaluate("Decision Tree (Task-2)",   DecisionTreeRegressor(max_depth=5, random_state=42), X_train, y_train, X_test, y_test),
    evaluate("Tuned Decision Tree",      dt_grid.best_estimator_,                         X_train, y_train, X_test, y_test),
]

summary = pd.DataFrame(rows).set_index("Model")
summary["Overfit Gap"] = (summary["Test RMSE"] - summary["Train RMSE"]).round(4)

print("=" * 55)
print("STEP 7 - Model Comparison Summary")
print(summary.to_string(), "\n")

# ── Step 8: Plots ─────────────────────────────────────────────────────────────
all_models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    ridge_grid.best_estimator_,
    DecisionTreeRegressor(max_depth=5, random_state=42),
    dt_grid.best_estimator_,
]
best_idx   = summary["R2 Score"].values.argmax()
best_name  = summary.index[best_idx]
best_mdl   = all_models[best_idx]
best_mdl.fit(X_train, y_train)
y_pred = best_mdl.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar: train vs test RMSE
ax = axes[0]
x, w = np.arange(len(summary)), 0.35
ax.bar(x - w/2, summary["Train RMSE"], w, label="Train RMSE", color="steelblue")
ax.bar(x + w/2, summary["Test RMSE"],  w, label="Test RMSE",  color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(summary.index, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("RMSE")
ax.set_title("Train vs Test RMSE (Overfit Gap)")
ax.legend()

# Scatter: actual vs predicted
axes[1].scatter(y_test, y_pred, alpha=0.3, s=10, label="Predictions")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r-", label="Ideal")
axes[1].set_xlabel("Actual")
axes[1].set_ylabel("Predicted")
axes[1].set_title(f"Actual vs Predicted - {best_name}")
axes[1].legend()

plt.tight_layout()
plt.savefig("task3_validation_analysis.png", dpi=150)
plt.close()
print("Chart saved: task3_validation_analysis.png")

# ── Step 9: CV on all tuned models ───────────────────────────────────────────
tuned_models = {
    "Linear Regression"      : LinearRegression(),
    "Ridge (Task-2)"          : Ridge(alpha=1.0),
    "Tuned Ridge"             : ridge_grid.best_estimator_,
    "Decision Tree (Task-2)" : DecisionTreeRegressor(max_depth=5, random_state=42),
    "Tuned Decision Tree"    : dt_grid.best_estimator_,
}

print("\n" + "=" * 55)
print("STEP 9 - Cross-Validation Summary (All Models)")
print(f"{'Model':<26} {'CV RMSE Mean':>14} {'CV RMSE Std':>12}")
print("-" * 54)
for name, model in tuned_models.items():
    s = cv_rmse(model, X_scaled, y)
    print(f"{name:<26} {s.mean():>14.4f} {s.std():>12.4f}")

# ── Step 10: Save best model ──────────────────────────────────────────────────
joblib.dump(best_mdl, "best_model_task3.pkl")
joblib.dump(scaler,   "scaler_task3.pkl")
print(f"\nSaved: {best_name} -> best_model_task3.pkl")
print("Saved: scaler_task3.pkl")
print("\nTask-3 complete.")
