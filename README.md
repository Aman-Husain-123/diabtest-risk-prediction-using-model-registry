# 🩺 Diabetes Risk Predictor — MLflow Model Registry Demo

> An end-to-end MLOps pipeline demonstrating **MLflow experiment tracking**, **hyperparameter tuning**, **model versioning**, **stage lifecycle management**, and a **Flask web UI** — all on a local SQLite-backed MLflow registry powered by the Pima Indians Diabetes Dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [What's Happening Behind the Scenes](#-whats-happening-behind-the-scenes)
- [MLflow Storage & Registry (Cloud-Ready Section)](#-mlflow-storage--registry-cloud-ready-section)
- [Screenshots](#-screenshots)
- [Complete Run — Step by Step](#-complete-run--step-by-step)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [API Reference](#-api-reference)

---

## 🎯 Project Overview

This project implements a **full MLOps lifecycle** for a diabetes risk classification model:

1. **Train** — RandomForestClassifier with GridSearchCV hyperparameter tuning, all tracked in MLflow with nested runs (parent + 20 child runs per hyperparameter combo)
2. **Register** — Best model is registered in the MLflow Model Registry with version tags and description
3. **Promote** — Model version is transitioned through lifecycle stages (Staging → Production)
4. **Infer** — Model is loaded directly from the registry and used for predictions
5. **Serve** — A Flask web application provides a polished UI for real-time inference

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **ML Framework** | scikit-learn 1.8.0 |
| **Experiment Tracking** | MLflow 3.13.0 |
| **Model Registry** | MLflow Model Registry (SQLite backend) |
| **Hyperparameter Tuning** | GridSearchCV (5-fold CV, 20 combinations) |
| **Web Framework** | Flask |
| **Data** | pandas, numpy |
| **Storage Backend** | SQLite (`mlflow.db`) + local filesystem (`mlruns/`) |
| **Language** | Python 3.13 |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE                                  │
│   Pima Indians Diabetes Dataset (GitHub raw CSV via pandas)         │
│   768 samples · 8 features · Binary classification (0/1)           │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE  (train.py)                    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  MLflow Parent Run  (experiment: diabetes-rf-hp)            │    │
│  │                                                             │    │
│  │   GridSearchCV (5-fold CV)                                  │    │
│  │   ├── n_estimators: [2, 25, 100, 150, 200]                  │    │
│  │   └── max_depth:    [None, 3, 6, 7]                         │    │
│  │   = 20 combinations × 5 folds = 100 fits                   │    │
│  │                                                             │    │
│  │   ┌──────────────────────────────────────────────────────┐  │    │
│  │   │  20 MLflow Child Runs (nested=True)                  │  │    │
│  │   │  Each logs: params + mean_test_score (accuracy)      │  │    │
│  │   └──────────────────────────────────────────────────────┘  │    │
│  │                                                             │    │
│  │   Parent Run also logs:                                     │    │
│  │   ├── best_params  (log_params)                             │    │
│  │   ├── best_score   (log_metric: accuracy)                   │    │
│  │   ├── train/test datasets (log_input)                       │    │
│  │   ├── train.py source code (log_artifact)                   │    │
│  │   ├── model signature (infer_signature)                     │    │
│  │   ├── sklearn model (log_model → mlruns/1/models/)          │    │
│  │   └── tag: author=nitish                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────┬───────────────────────────────────────────────────┘
                  │  model_id: m-xxxx...
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│               MODEL REGISTRY  (register.py)                        │
│                                                                     │
│   mlflow.register_model(f"models:/{model_id}", "diabetes-rf-hp")   │
│                                                                     │
│   MLflowClient.update_model_version()  ← adds description          │
│   MLflowClient.set_model_version_tag() ← experiment, day tags      │
│                                                                     │
│   Registered Model: diabetes-rf-hp                                  │
│   ├── Version 1  (archived)                                         │
│   ├── Version 2  (archived)                                         │
│   └── Version 3  ← current                                          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│            STAGE LIFECYCLE  (stage_transition.py)                   │
│                                                                     │
│   Staging  →  Production  (archive_existing_versions=True)          │
│                                                                     │
│   Uses: client.transition_model_version_stage()                     │
│   (deprecated in MLflow 2.9+; modern alternative: aliases)         │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
          ┌───────┴──────────┐
          ▼                  ▼
┌─────────────────┐  ┌───────────────────────────────────────────────┐
│  CLI INFERENCE  │  │         FLASK WEB UI  (app.py)                │
│  inference.py   │  │                                               │
│                 │  │  GET  /         → renders templates/index.html │
│  Loads model    │  │  POST /predict  → returns JSON prediction      │
│  from registry  │  │                                               │
│  via pyfunc     │  │  Model loaded ONCE at startup from registry   │
│                 │  │  Input: pd.DataFrame with named columns        │
│  Output: [0/1]  │  │  Output: { prediction, label, note }          │
└─────────────────┘  └───────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
model-registry-demo-practice/
│
├── train.py               # Training + MLflow experiment tracking
├── register.py            # Model registration in MLflow registry
├── stage_transition.py    # Stage lifecycle management (Staging/Production)
├── inference.py           # CLI inference from registered model
├── app.py                 # Flask web app (serves UI + /predict API)
│
├── templates/
│   └── index.html         # Premium dark-themed prediction UI
│
├── mlruns/                # MLflow tracking artifacts (auto-generated)
│   └── 1/                 # Experiment ID for "diabetes-rf-hp"
│       ├── <run_id>/      # One folder per MLflow run
│       │   └── artifacts/ # Logged artifacts (train.py source)
│       └── models/        # Stored model artifacts (MLflow 3.x layout)
│           └── m-xxxx/
│               └── artifacts/
│                   ├── model.pkl
│                   ├── MLmodel
│                   ├── conda.yaml
│                   ├── python_env.yaml
│                   └── requirements.txt
│
└── mlflow.db              # SQLite database for MLflow registry metadata
```

---

## 🔍 What's Happening Behind the Scenes

### 1. `train.py` — Training & Tracking

```python
mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run(description="Best hyperparameter trained RF model") as parent:
    grid_search.fit(X_train, y_train)

    # Log every combination as a child run
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])

    # Log best model to registry-compatible storage
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest", signature=signature)
```

- **GridSearchCV** tests 20 hyperparameter combinations × 5 CV folds = **100 model fits**
- **Nested runs**: Each combination gets its own child run with params + CV accuracy
- **Model signature**: Enforces typed input schema at inference time (`long` for ints, `double` for floats)
- **MLflow 3.x storage**: Models are stored under `mlruns/1/models/m-<uuid>/` (not inside run artifacts), assigned a unique `model_id`

### 2. `register.py` — Model Registry

```python
# MLflow 3.x uses model ID URI, not runs:/<run_id>/<artifact_name>
model_uri = f"models:/{model_id}"
result = mlflow.register_model(model_uri, "diabetes-rf-hp")

client.update_model_version(name=model_name, version=result.version,
    description="RandomForest model for Pima Indians Diabetes prediction.")
client.set_model_version_tag(name=model_name, version=result.version,
    key="experiment", value="diabetes prediction")
```

- **Model URI format changed in MLflow 3.x**: The correct format is `models:/<model_id>` rather than `runs:/<run_id>/<artifact_name>` because the model artifact is no longer nested inside the run directory
- **Version auto-increments**: Each call to `register_model` creates a new version number
- **Tags & descriptions** are attached to each version for governance/auditability

### 3. `stage_transition.py` — Lifecycle Stages

```python
client.transition_model_version_stage(
    name="diabetes-rf-hp",
    version=3,
    stage="Production",
    archive_existing_versions=True  # Automatically archives previous Production
)
```

- Moves version 3 from **Staging → Production**
- `archive_existing_versions=True` ensures only one version is active in Production at a time
- **Note**: Stage-based lifecycle is deprecated in MLflow 2.9+; the modern approach uses **model aliases** (`client.set_registered_model_alias(name, "champion", 3)`)

### 4. `inference.py` — CLI Prediction

```python
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Must pass a DataFrame with NAMED columns — model signature enforces this
data = pd.DataFrame([[1, 85, 66, 29, 0, 26.6, 0.351, 31]],
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', ...])
print(model.predict(data))  # → [0] (Not Diabetic)
```

- Loads model from the **MLflow Model Registry** by name + version
- **Critical**: A raw numpy array fails schema enforcement because the model was trained on a DataFrame. Column names must match exactly.

### 5. `app.py` — Flask Inference API

```python
# Model loaded ONCE at server startup (not per-request)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([[...]], columns=[...])
    prediction = model.predict(input_df)
    return jsonify({"prediction": int(prediction[0]), "label": "Diabetic"/"Not Diabetic"})
```

- Model is loaded **once at startup** into memory for fast per-request inference
- Accepts JSON body, returns JSON response
- Frontend JavaScript calls `/predict` asynchronously and renders result without page reload

---

## ☁️ MLflow Storage & Registry (Cloud-Ready Section)

This project currently runs on a **local SQLite backend**, but the exact same code works with cloud backends by changing a single environment variable.

### Current: Local SQLite Backend

```
MLFLOW_TRACKING_URI = sqlite:///mlflow.db   (default when using mlflow.db)
Artifact store     = ./mlruns/               (local filesystem)
```

### Cloud Deployment Options

| Cloud | Tracking Server | Artifact Store | How to Switch |
|---|---|---|---|
| **AWS** | RDS PostgreSQL | S3 Bucket | `MLFLOW_TRACKING_URI=postgresql://...` + `MLFLOW_S3_ENDPOINT_URL` |
| **Azure** | Azure SQL / Blob metadata | Azure Blob Storage | `MLFLOW_TRACKING_URI=azureml://...` |
| **GCP** | Cloud SQL (PostgreSQL) | GCS Bucket | `MLFLOW_TRACKING_URI=postgresql://...` + `ARTIFACT_ROOT=gs://...` |
| **MLflow Cloud** | Databricks Managed MLflow | DBFS / S3 | `MLFLOW_TRACKING_URI=databricks` |

**Zero code changes needed** — only environment variables change:

```bash
# Switch to AWS S3 + RDS
export MLFLOW_TRACKING_URI="postgresql://user:pass@rds-endpoint/mlflowdb"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://my-mlflow-bucket/artifacts"

# Then run exactly the same scripts
python train.py
python register.py
python stage_transition.py
python app.py
```

### MLflow 3.x Model Storage Layout

In MLflow 3.x (this project uses **3.13.0**), the artifact storage layout changed significantly:

```
mlruns/
└── <experiment_id>/
    ├── <run_id>/
    │   └── artifacts/        ← run artifacts (source code, etc.)
    │       └── train.py
    └── models/               ← NEW: models stored separately from runs
        └── m-<model_uuid>/
            └── artifacts/
                ├── MLmodel   ← contains model_id, run_id, flavor info
                ├── model.pkl ← serialized sklearn model
                ├── conda.yaml
                └── requirements.txt
```

This separation means the model URI changed from `runs:/<run_id>/random_forest` (MLflow 2.x) to `models:/<model_id>` (MLflow 3.x).

---

## 📸 Screenshots

### Flask Web UI — Prediction Form

![Flask Diabetes Predictor UI](ss_flask_empty.png)

### Flask Web UI — Prediction Result (Diabetic)

![Prediction Result - Diabetic Risk Detected](ss_flask_result.png)

### MLflow Model Registry — Multiple Versions

The MLflow UI at `http://127.0.0.1:5000` shows:
- **4 logged model runs** with accuracies ranging from **0.776 to 0.783**
- Model versions v1, v2, v3 registered under `diabetes-rf-hp`
- Version 3 promoted to **Production** stage

---

## 🚀 Complete Run — Step by Step

### Prerequisites

```bash
pip install mlflow scikit-learn pandas flask
```

### Step 1 — Start MLflow Tracking Server

```bash
# Start MLflow UI with local SQLite backend (maintenance mode bypass)
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 to see the MLflow dashboard.

### Step 2 — Train the Model

```bash
python train.py
```

**What happens:**
- Downloads Pima Indians Diabetes Dataset from GitHub
- Splits 80/20 train/test
- Runs GridSearchCV: 5 depth values × 4 estimator counts = 20 combinations × 5-fold CV = **100 model fits**
- Creates 1 parent MLflow run + 20 nested child runs
- Logs params, metrics, dataset references, source code artifact, and model
- Prints best params and best CV accuracy (~0.78)

**Example output:**
```
Fitting 5 folds for each of 20 candidates, totalling 100 fits
[CV] END max_depth=None, n_estimators=2; total time=   0.0s
...
{'max_depth': 7, 'n_estimators': 25}
0.7834199653471945
```

### Step 3 — Register the Model

```bash
python register.py
```

**What happens:**
- Reads the model ID from the latest training run's MLmodel file
- Registers the model under the name `diabetes-rf-hp` in the MLflow Model Registry
- Auto-increments version number (v1, v2, v3...)
- Adds description: "RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset"
- Attaches tags: `experiment=diabetes prediction`, `day=sat`

**Example output:**
```
Registered model 'diabetes-rf-hp' already exists. Creating a new version of this model...
Created version '3' of model 'diabetes-rf-hp'.
Model registered with name: diabetes-rf-hp and version: 3
Added tags to model diabetes-rf-hp version 3
```

### Step 4 — Promote to Production

```bash
python stage_transition.py
```

**What happens:**
- Transitions model version 3 from Staging → Production
- Archives any existing Production version automatically

**Example output:**
```
Model version 3 transitioned to Production
```

### Step 5 — CLI Inference (optional)

```bash
python inference.py
```

**Example output:**
```
[0]   # 0 = Not Diabetic, 1 = Diabetic
```

### Step 6 — Launch Flask Web UI

```bash
python app.py
```

Open http://127.0.0.1:5001 in your browser.

**What happens at startup:**
- Flask loads model version 3 from the MLflow registry into memory
- Serves the dark-themed prediction UI
- Listens for POST requests on `/predict`

**Making a prediction:**
1. Fill in the 8 clinical fields
2. Click **Run Prediction**
3. Result appears instantly: ✅ *Not Diabetic* or ⚠️ *Diabetic — Positive Risk Detected*

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Name** | Pima Indians Diabetes Dataset |
| **Source** | [GitHub (npradaschnor)](https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv) |
| **Rows** | 768 |
| **Features** | 8 |
| **Target** | `Outcome` (0 = No Diabetes, 1 = Diabetes) |
| **Train split** | 614 samples (80%) |
| **Test split** | 154 samples (20%) |

### Features

| Feature | Type | Description |
|---|---|---|
| `Pregnancies` | int | Number of times pregnant |
| `Glucose` | int | Plasma glucose concentration (2h oral glucose tolerance test) |
| `BloodPressure` | int | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | int | Triceps skin fold thickness (mm) |
| `Insulin` | int | 2-hour serum insulin (mu U/ml) |
| `BMI` | float | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | float | Genetic influence score |
| `Age` | int | Age in years |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| **Algorithm** | RandomForestClassifier |
| **Tuning** | GridSearchCV, 5-fold cross-validation |
| **Best params** | `max_depth=7, n_estimators=25` (run 2) |
| **Best CV accuracy** | ~78.3% |
| **Serialization** | cloudpickle (sklearn flavor) |
| **MLflow flavor** | `mlflow.sklearn` |
| **Model size** | ~215 KB (v1, v3) to ~3.4 MB (v with n_estimators=200) |

### Registered Versions

| Version | Best Params | CV Accuracy | Stage |
|---|---|---|---|
| v1 | `max_depth=None, n_estimators=200` | 0.7769 | Archived |
| v2 | `max_depth=7, n_estimators=25` | 0.7834 | Archived |
| v3 | `max_depth=7, n_estimators=25` | 0.7834 | **Production** |

---

## 🔌 API Reference

### `POST /predict`

Accepts JSON, returns JSON prediction.

**Request:**
```json
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "dpf": 0.627,
  "age": 50
}
```

**Response (Diabetic):**
```json
{
  "prediction": 1,
  "label": "Diabetic",
  "confidence_note": "Based on Pima Indians Diabetes Dataset"
}
```

**Response (Not Diabetic):**
```json
{
  "prediction": 0,
  "label": "Not Diabetic",
  "confidence_note": "Based on Pima Indians Diabetes Dataset"
}
```

**Error Response:**
```json
{
  "error": "description of what went wrong"
}
```

---

## ⚠️ Known Issues & Notes

- **`artifact_path` deprecation**: MLflow 3.x shows a warning that `artifact_path` is deprecated; use `name` instead in `log_model()`
- **`transition_model_version_stage` deprecation**: Stages are deprecated since MLflow 2.9. Migrate to **aliases** for new projects: `client.set_registered_model_alias(name, "champion", version)`
- **Integer schema warning**: MLflow warns that integer columns can't represent `NaN` at inference time. Cast integer columns to `float64` if your inference data may contain missing values
- **Pickle security**: MLflow warns about cloudpickle serialization. For production, use the `skops` format instead

---

## 📄 License

This project is for educational/demo purposes using the publicly available Pima Indians Diabetes Dataset.
