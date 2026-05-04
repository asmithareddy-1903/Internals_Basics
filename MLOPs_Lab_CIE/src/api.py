from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

app = FastAPI()

# Load dataset and train model
df = pd.read_csv("MLOPs_Lab_CIE/data/training_data.csv")

X = df.drop("report_turnaround_hours", axis=1)
y = df["report_turnaround_hours"]

model = GradientBoostingRegressor(random_state=42)
model.fit(X, y)

# Input schema with validation
class InputData(BaseModel):
    sample_count: int = Field(..., ge=1)
    test_complexity: int = Field(..., ge=1)
    lab_technician_count: int = Field(..., ge=1)
    is_urgent: int = Field(..., ge=0, le=1)

# Health endpoint
@app.get("/health")
def health():
    return {
        "alive": True,
        "service": "PathScan report_turnaround_hours API"
    }

# Prediction endpoint
@app.post("/forecast")
def forecast(data: InputData):
    try:
        features = np.array([[ 
            data.sample_count,
            data.test_complexity,
            data.lab_technician_count,
            data.is_urgent
        ]])

        prediction = model.predict(features)[0]

        return {"prediction": float(prediction)}

    except Exception:
        raise HTTPException(status_code=422, detail="Invalid input")