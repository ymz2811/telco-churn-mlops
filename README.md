# Telco Customer Churn — MLOps Pipeline

End-to-end ML project: training with experiment tracking, a REST inference API, and Kubernetes deployment.

**Dataset**: [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — 7,000 customers, 19 features, binary churn label.

## Stack

| Layer | Tools |
|---|---|
| Modelling | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Serving | FastAPI, Uvicorn |
| Containerisation | Docker |
| Orchestration | Kubernetes (minikube) |

## Results



## Project Structure

```
├── src/
│   ├── preprocess.py   # ColumnTransformer pipeline (scaling + encoding)
│   └── train.py        # MLflow experiment across 4 model configs
├── api/
│   ├── main.py         # FastAPI app with /predict and /health endpoints
│   └── schemas.py      # Pydantic request/response models
├── k8s/
│   ├── deployment.yaml # 2-replica Deployment with health probes
│   └── service.yaml    # LoadBalancer Service
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train — runs 4 experiments, registers best model

```bash
python src/train.py
```

### 3. Inspect runs in MLflow UI

```bash
mlflow ui --port 5000
# open http://localhost:5000
```

### 4. Run the API locally

```bash
uvicorn api.main:app --reload
# open http://localhost:8000/docs
```

### 5. Build and run with Docker

```bash
docker build -t churn-api:latest .
docker run --rm -p 8000:8000 churn-api:latest
```

### 6. Deploy to Kubernetes (minikube)

```bash
minikube start
minikube image load churn-api:latest
kubectl apply -f k8s/
minikube service churn-api-service
```

## API Usage

**POST** `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85, "TotalCharges": 958.2
  }'
```

**Response:**

```json
{
  "churn_probability": 0.7431,
  "churn_prediction": true,
  "risk_level": "high"
}
```
