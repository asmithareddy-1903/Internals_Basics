import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("MLOPs_Lab_CIE/data/training_data.csv")

X = df.drop("report_turnaround_hours", axis=1)
y = df["report_turnaround_hours"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

model = GradientBoostingRegressor(random_state=42)

mlflow.set_experiment("pathscan-report-turnaround-hours")

with mlflow.start_run(run_name="tuning-pathscan"):

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=5,
        cv=3,
        scoring="neg_mean_squared_error",
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_params(search.best_params_)
    mlflow.log_metrics({
        "best_mae": mae,
        "best_rmse": rmse
    })

# Save JSON
import json

output = {
    "search_type": "random",
    "n_folds": 3,
    "total_trials": 5,
    "best_params": search.best_params_,
    "best_mae": mae,
    "best_cv_mae": abs(search.best_score_),
    "parent_run_name": "tuning-pathscan"
}

with open("MLOPs_Lab_CIE/results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 completed")