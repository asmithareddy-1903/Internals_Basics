import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("MLOPs_Lab_CIE/data/training_data.csv")

X = df.drop("report_turnaround_hours", axis=1)
y = df["report_turnaround_hours"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Metrics function
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    return mae, rmse, r2, mape

mlflow.set_experiment("pathscan-report-turnaround-hours")

results = []

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        mae, rmse, r2, mape = evaluate(model, X_test, y_test)

        # Log params & metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })
        mlflow.set_tag("experiment_type", "baseline_comparison")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

# Find best model
best_model = min(results, key=lambda x: x["rmse"])

# Save JSON
output = {
    "experiment_name": "pathscan-report-turnaround-hours",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best_model["rmse"]
}

import json
with open("MLOPs_Lab_CIE/results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")