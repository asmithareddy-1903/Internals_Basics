import pandas as pd
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("MLOPs_Lab_CIE/data/training_data.csv")

X = df.drop("report_turnaround_hours", axis=1)
y = df["report_turnaround_hours"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model
model = GradientBoostingRegressor()

# Hyperparameter space
param_dist = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1]
}

# Random Search
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=5,                # EXACTLY 5 trials
    scoring="neg_mean_absolute_error",
    cv=3,                    # 3 folds
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

# Test MAE
y_pred = best_model.predict(X_test)
best_mae = mean_absolute_error(y_test, y_pred)

# CV MAE (IMPORTANT FIX)
best_cv_mae = abs(search.best_score_)

# Save results
results = {
    "search_type": "random",
    "n_folds": 3,
    "total_trials": 5,
    "best_params": search.best_params_,
    "best_mae": best_mae,
    "best_cv_mae": best_cv_mae,
    "parent_run_name": "tuning-pathscan"
}

with open("MLOPs_Lab_CIE/results/step2_s2.json", "w") as f:
    json.dump(results, f, indent=4)

print("Task 2 completed successfully!")