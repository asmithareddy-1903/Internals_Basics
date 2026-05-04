import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load datasets
train_df = pd.read_csv("MLOPs_Lab_CIE/data/training_data.csv")
new_df = pd.read_csv("MLOPs_Lab_CIE/data/new_data.csv")

# Combine data
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# Features and target
X = train_df.drop("report_turnaround_hours", axis=1)
y = train_df["report_turnaround_hours"]

X_combined = combined_df.drop("report_turnaround_hours", axis=1)
y_combined = combined_df["report_turnaround_hours"]

# SAME TEST SET (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Champion model (old)
champion_model = GradientBoostingRegressor(random_state=42)
champion_model.fit(X_train, y_train)

champion_preds = champion_model.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, champion_preds))

# Retrained model (new data)
retrained_model = GradientBoostingRegressor(random_state=42)
retrained_model.fit(X_combined, y_combined)

retrained_preds = retrained_model.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, retrained_preds))

# Improvement
improvement = champion_rmse - retrained_rmse

# Decision rule
threshold = 0.3

if improvement >= threshold:
    action = "promoted"
else:
    action = "kept_champion"

# Save JSON
output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": float(champion_rmse),
    "retrained_rmse": float(retrained_rmse),
    "improvement": float(improvement),
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "rmse"
}

with open("MLOPs_Lab_CIE/results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")