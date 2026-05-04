# MLOps Lab CIE — PathScan Report Turnaround Time Prediction

## 1. Overview

This project implements an end-to-end MLOps pipeline to predict report turnaround time in a pathology lab using machine learning techniques. The workflow includes model training, hyperparameter tuning, API deployment, and retraining.

---

## 2. Tasks Implemented

### 2.1 Task 1 — Model Training and Comparison

Two models were trained and evaluated:

* Random Forest Regressor
* Gradient Boosting Regressor

Evaluation metrics used:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Mean Absolute Percentage Error (MAPE)

The best model was selected based on the lowest RMSE.

Output file: `results/step1_s1.json`

---

### 2.2 Task 2 — Hyperparameter Tuning

Hyperparameter tuning was performed using:

* Method: Random Search
* Number of trials: 5
* Cross-validation: 3-fold

The best parameters were selected based on MAE.

Output file: `results/step2_s2.json`

---

### 2.3 Task 3 — FastAPI Model Serving

The selected model was deployed using FastAPI.

Endpoints:

* `GET /health` — returns service status
* `POST /forecast` — accepts input features and returns prediction

The API runs on port 9000 and uses Pydantic for input validation.

Output file: `results/step3_s4.json`

---

### 2.4 Task 4 — Retraining Pipeline

The retraining process includes:

* Combining `training_data.csv` and `new_data.csv`
* Retraining the selected model
* Comparing retrained model with the existing model using RMSE

Promotion rule:

* If RMSE improves by at least 0.3, the model is promoted
* Otherwise, the existing model is retained

Output file: `results/step4_s8.json`

---

## 3. Project Structure

```id="struct01"
MLOPs_Lab_CIE/
│
├── data/
├── logs/
├── models/
├── results/
├── src/
│   ├── train.py
│   ├── tune.py
│   ├── api.py
│   ├── retrain.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Setup Instructions

Create and activate a virtual environment:

```bash id="setup01"
python3 -m venv venv
source venv/bin/activate
```

Install required dependencies:

```bash id="setup02"
pip install -r requirements.txt
```

---

## 5. Execution Steps

Train the model:

```bash id="run01"
python MLOPs_Lab_CIE/src/train.py
```

Run hyperparameter tuning:

```bash id="run02"
python MLOPs_Lab_CIE/src/tune.py
```

Start the API server:

```bash id="run03"
uvicorn MLOPs_Lab_CIE.src.api:app --reload --port 9000
```

---

## 6. API Usage

Health check:

```id="api01"
GET http://127.0.0.1:9000/health
```

Prediction endpoint:

```id="api02"
POST http://127.0.0.1:9000/forecast
```

Sample input:

```json id="api03"
{
  "sample_count": 26,
  "test_complexity": 2,
  "lab_technician_count": 4,
  "is_urgent": 1
}
```

---

## 7. Technologies Used

* Python
* Scikit-learn
* FastAPI
* Pandas
* NumPy
* Uvicorn

---

## 8. Conclusion

This project demonstrates a complete machine learning lifecycle, including model development, optimization, deployment, and retraining within an MLOps framework.
