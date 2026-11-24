# MP138 --- ATM Cash Demand Prediction (Patna)

**SCSE3040 MLOps Mini Project**  
**Group ID:** G053  
**Problem ID:** MP138  
**Faculty:** Avantika Saklani  
**Batch:** 74  

---

## Group Members

| Enrolment No. | Name | Role |
| :--- | :--- | :--- |
| **S24CSEU2785** | Ansh Sharma | Model & MLflow Lead |
| **S24CSEU2799** | Amit Bhardwaj | Data & DVC Lead |
| **S24CSEU2769** | Rakesh Kumar | Service & Docker Lead |
| **S24CSEU2792** | Aditya Srivastava | Ops, K8s & Monitoring Lead |

---

## 1. Project Overview

Patna Cooperative Bank manages cash logistics for ATMs located across Patna. Overstocking causes idle cash holding costs, while understocking leads to ATM cashouts and service disruptions. This project implements an end-to-end MLOps pipeline to predict daily cash withdrawals (`withdrawal_thousand_rs`) using regression models evaluated via Mean Absolute Error (MAE).

The complete production pipeline covers:
- **Data Engineering & Versioning:** DVC tracking, data anomaly detection (resolving unit mixup, sentinels, duplicates, and target typos).
- **Experiment Tracking & Model Registry:** MLflow SQLite tracking (`sqlite:///mlflow.db`) with multiple model iterations, SHA-256 integrity validation, and `@champion` aliasing.
- **Microservice API:** Production-grade FastAPI serving `POST /predict` and `GET /health` with Pydantic request validation and structured JSON logging.
- **Containerization & Deployment:** Docker image `mp138-api:1.0` and production Kubernetes manifests (`k8s/deployment.yaml`, `k8s/service.yaml`) configured with 3 replicas and health probes.
- **Continuous Integration:** Automated GitHub Actions workflow running unit and API tests on every push.
- **Drift Monitoring & Automated Retraining:** Population Stability Index (PSI) analysis across temporal partitions and policy-driven model promotion.

---

## 2. Repository Layout

```
├── .dvc/                           # DVC configuration
├── .github/workflows/ci.yml        # GitHub Actions CI workflow
├── app/
│   ├── __init__.py
│   └── main.py                     # FastAPI application with /predict and /health
├── data/
│   ├── raw/
│   │   ├── train.csv.dvc           # DVC pointer for train.csv
│   │   ├── later.csv.dvc           # DVC pointer for later.csv
│   │   └── holdout_features.csv    # Holdout features dataset
├── k8s/
│   ├── deployment.yaml             # Kubernetes Deployment (mp138-api, 3 replicas)
│   └── service.yaml                # Kubernetes Service (mp138-svc, port 80 -> 8000)
├── models/
│   └── model.joblib                # Champion serialized pipeline model
├── predictions/
│   └── holdout_predictions.csv     # Final holdout inference predictions
├── reports/
│   └── findings.yaml               # Evaluated findings and audit report
├── src/
│   └── atm_demand/
│       ├── __init__.py
│       ├── data.py                 # Data loading, validation, and cleaning logic
│       ├── train.py                # MLflow training pipeline and model registry
│       ├── evaluate.py             # Evaluation routines and holdout prediction
│       └── monitor.py              # PSI calculation and retraining logic
├── tests/
│   ├── test_data.py                # Data pipeline unit tests
│   ├── test_model.py               # Model training and prediction tests
│   └── test_api.py                 # FastAPI TestClient endpoint and validation tests
├── Dockerfile                      # Production Docker container definition
├── .dockerignore                   # Docker build ignore file
├── findings.yaml                   # Findings report at root
├── mlflow.db                       # SQLite MLflow experiment and registry database
├── params.yaml                     # Centralized pipeline hyperparameters & paths
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Project documentation
```

---

## 3. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd G053_MP138
   ```

2. **Set up Python Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 4. Running the Pipeline

### Data Cleaning and MLflow Experiment Tracking
```bash
python -m src.atm_demand.train
```

### Model Monitoring and Retraining
```bash
python -m src.atm_demand.monitor
```

### Holdout Predictions
```bash
python -m src.atm_demand.evaluate
```

### Running Tests
```bash
pytest -v
```

---

## 5. API Service Execution

Start the local FastAPI service:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints
- `GET /health`: Health check endpoint returning `{"status": "ok"}`
- `POST /predict`: Predicts ATM daily cash withdrawal in thousand rupees

#### Sample Request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nearby_shops": 45,
    "days_to_salary": 12,
    "footfall_index": 68.5,
    "distance_to_market_km": 2.4,
    "avg_ticket_rs": 2800.0,
    "other_atms_1km": 5,
    "atm_type": "standalone",
    "area": "Residential"
  }'
```

#### Sample Response:
```json
{
  "withdrawal_thousand_rs": 612.45
}
```

---

## 6. Docker & Kubernetes Deployment

### Docker Build and Run
```bash
docker build -t mp138-api:1.0 .
docker run -p 8000:8000 mp138-api:1.0
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
