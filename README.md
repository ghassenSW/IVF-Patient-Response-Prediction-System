# IVF Patient Response Prediction

Machine learning system for predicting ovarian response to IVF treatment based on patient characteristics. Uses probabilistic classification to stratify patients into low, optimal, or high response categories.

## 🌐 Live Demo

**Try the live application**: [https://ivf-patient-response-prediction-system.onrender.com](https://ivf-patient-response-prediction-system.onrender.com)

![Application Screenshot](screenshot.png)

> **Note**: The application may take 30 seconds to wake up on first visit (free tier hosting).

## Overview

- **Goal**: Predict patient response to ovarian stimulation for treatment optimization
- **Model**: Calibrated Gradient Boosting Classifier with SHAP explainability
- **Performance**: 86.1% accuracy with reliable probability estimates
- **Interface**: REST API + Web UI for clinical use
- **Key Features**: AMH, follicle count, AFC, age, cycle number, protocol type

## Project Structure

```
├── README.md
├── requirements.txt
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── preprocessing
│   │   ├── clean_dataset.py
│   │   └── feature_engineering.py
│   │
│   ├── model
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── saved_models
│   │
│   ├── api
│   │   └── app.py
│   └── ui
│       └── [UI files]
│
└── tests
    └── test_model.py
```

## Quick Start

### Option 1: Use the Live Web Application

Visit **[https://ivf-patient-response-prediction-system.onrender.com](https://ivf-patient-response-prediction-system.onrender.com)** to use the application directly without any setup!

### Option 2: Run Locally

1. **Clone the repository**
```bash
git clone <repository-url>
cd "tanit ai"
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start the application**
```bash
python src/api/app.py
```

5. **Open in browser**: http://localhost:8000

## Usage

### 1. Data Preprocessing

Clean and prepare the dataset:
```bash
python src/preprocessing/clean_dataset.py
python src/preprocessing/feature_engineering.py
```

Run exploratory data analysis (optional):
```bash
python src/preprocessing/eda.py
```

### 2. Model Training

Train models with calibration:
```bash
python src/model/train.py
```

Trains three models:
- Random Forest
- Gradient Boosting (best performance)
- Logistic Regression

### 3. Model Evaluation

Evaluate performance on test set:
```bash
python src/model/evaluate.py
```

Generates:
- Confusion matrices
- Calibration curves
- Per-class metrics
- Confidence analysis

### 4. Model Explainability

Analyze with SHAP:
```bash
python src/model/explainability.py
```

Creates visualizations:
- Feature importance bar chart
- Beeswarm plots per class
- Dependence plots for top features
- Individual prediction waterfall

### 5. Make Predictions

Run inference:
```bash
python src/model/predict.py
```

## Interface Usage

### Option 1: Web UI (Recommended for Clinicians)

**Start the API server:**
```bash
python src/api/start_api.py
```

**Open the UI:**
- Open `src/ui/index.html` in your web browser
- Fill in patient information with **real clinical values**:
  - **Cycle Number**: Treatment attempt (1-10)
  - **Age**: Patient age in years (18-50, typical: 24-45)
  - **AMH**: Anti-Müllerian Hormone level in ng/mL (0-10, typical: 0.1-6.5)
  - **Number of Follicles**: Follicle count (0-50, typical: 1-46)
  - **E2 Day 5**: Estradiol level in pg/mL (0-5000, typical: 29-5000)
  - **AFC**: Antral Follicle Count (0-50, typical: 3-30)
  - **Protocol**: Select one stimulation protocol
- Click "Predict Response" to see results

**Note**: The system automatically normalizes input values before prediction.

**API Documentation (Swagger UI):**
Visit `http://localhost:8000/docs` for interactive API testing

**Programmatic API Access:**
```python
import requests

# Use real clinical values - API handles normalization automatically
patient_data = {
    "cycle_number": 1,
    "Age": 32,
    "AMH": 2.5,
    "n_Follicles": 12,
    "E2_day5": 300,
    "AFC": 15,
    "Protocol_agonist": False,
    "Protocol_fixed antagonist": False,
    "Protocol_flexible antagonist": True
}

response = requests.post("http://localhost:8000/predict", json=patient_data)
print(response.json())
```

## Model Performance

**Gradient Boosting (Calibrated)**

| Metric | Score |
|--------|-------|
| Accuracy | 86.1% |
| Precision | 86.7% |
| F1-Score | 86.2% |

**Per-Class Performance:**
- Low Response: 90% accuracy
- Optimal Response: 84% accuracy  
- High Response: 84% accuracy

**Top Predictive Features:**
1. Number of follicles (n_Follicles)
2. Anti-Müllerian Hormone level (AMH)
3. Antral follicle count (AFC)

## Features

### Probabilistic Predictions
- Probability for each response class
- Calibrated for reliable estimates
- Confidence scores included

### Model Interpretability
- SHAP analysis (global + local)
- Feature importance rankings
- Individual prediction breakdowns

### Clinical Integration
- RESTful API
- Normalized inputs
- Clinical interpretations

## Technical Details

**Data Processing:**
- Missing value imputation (median)
- Protocol standardization
- One-hot encoding
- StandardScaler normalization

**Training:**
- 80/20 train-test split
- Stratified sampling
- Sigmoid calibration
- 5-fold cross-validation

## Dependencies

Core:
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+
- matplotlib 3.7+
- seaborn 0.12+
- shap 0.44.1

API (optional):
- fastapi 0.104+
- uvicorn 0.24+
- pydantic 2.5+

## 🧪 Testing

Run unit tests to verify the setup:

```bash
# Install testing dependencies
pip install pytest

# Run all tests
pytest tests/ -v

## Testing

Run unit tests:
```bash
pytest tests/test_model.py -v
```

## Clinical Interpretation

**Response Categories:**
- **Low Response**: Fewer follicles retrieved, may require protocol adjustment
- **Optimal Response**: Balanced follicular response with good pregnancy outcomes
- **High Response**: Excellent follicular yield, monitor for OHSS risk

## Bibliography

1. La Marca A, et al. "Anti-Müllerian hormone as a predictive marker in assisted reproductive technology." Human Reproduction Update, 2010.
2. Broekmans FJ, et al. "A systematic review of tests predicting ovarian reserve and IVF outcome." Human Reproduction Update, 2006.
3. Ferraretti AP, et al. "ESHRE consensus on the definition of 'poor response' to ovarian stimulation." Human Reproduction, 2011.
4. Lundberg SM, Lee SI. "A unified approach to interpreting model predictions (SHAP)." NIPS 2017.

## Author

Ghassen - IVF Response Prediction Project
