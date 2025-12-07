"""
FastAPI application for IVF patient response prediction
Provides endpoints for model inference with probability outputs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict
import joblib
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from model.dataset import get_response_labels

app = FastAPI(
    title="IVF Response Prediction API",
    description="Predict patient ovarian response to IVF treatment",
    version="1.0.0"
)

# Enable CORS for UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for development and production
        "https://*.onrender.com",  # Render deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None
feature_names = None
scaler = None

@app.on_event("startup")
async def load_model():
    """Load trained model and feature configuration"""
    global model, feature_names, scaler
    
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "src" / "model" / "saved_models"
    data_dir = project_root / "data" / "processed"
    
    model = joblib.load(models_dir / "gradient_boosting.pkl")
    feature_info = joblib.load(models_dir / "feature_info.pkl")
    feature_names = feature_info['feature_names']
    scaler = joblib.load(data_dir / "scaler.pkl")
    
    print("Model and scaler loaded successfully!")


class PatientFeatures(BaseModel):
    """Input features for prediction (real clinical values)"""
    cycle_number: int = Field(..., description="Treatment cycle number (1, 2, 3, etc.)", ge=1, le=20)
    Age: float = Field(..., description="Patient age in years", ge=18, le=60)
    AMH: float = Field(..., description="Anti-Müllerian hormone level (ng/mL)", ge=0.0, le=20.0)
    n_Follicles: int = Field(..., description="Number of follicles", ge=0, le=100)
    E2_day5: float = Field(..., description="Estradiol level day 5 (pg/mL)", ge=0, le=10000)
    AFC: int = Field(..., description="Antral follicle count", ge=0, le=2000)
    Protocol_agonist: bool = Field(False, description="Using agonist protocol", alias="Protocol_agonist")
    Protocol_fixed_antagonist: bool = Field(False, description="Using fixed antagonist protocol", alias="Protocol_fixed antagonist")
    Protocol_flexible_antagonist: bool = Field(False, description="Using flexible antagonist protocol", alias="Protocol_flexible antagonist")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Prediction output with probabilities"""
    prediction: str = Field(..., description="Predicted response class")
    confidence: float = Field(..., description="Confidence of prediction (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    clinical_note: str = Field(..., description="Clinical interpretation")


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "IVF Response Prediction API",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientFeatures):
    """
    Predict patient ovarian response to IVF treatment
    Accepts real clinical values and normalizes them automatically
    
    Returns prediction with probabilities and clinical interpretation
    """
    try:
        # Prepare numerical features for normalization (in correct order)
        numerical_features = np.array([
            patient.cycle_number,
            patient.Age,
            patient.AMH,
            patient.n_Follicles,
            patient.E2_day5,
            patient.AFC
        ]).reshape(1, -1)
        
        # Normalize numerical features
        normalized_features = scaler.transform(numerical_features)[0]
        
        # Prepare complete feature vector with protocol one-hot encoding
        # Model expects spaces in protocol names
        feature_dict = {
            'cycle_number': normalized_features[0],
            'Age': normalized_features[1],
            'AMH': normalized_features[2],
            'n_Follicles': normalized_features[3],
            'E2_day5': normalized_features[4],
            'AFC': normalized_features[5],
            'Protocol_agonist': int(patient.Protocol_agonist),
            'Protocol_fixed antagonist': int(patient.Protocol_fixed_antagonist),
            'Protocol_flexible antagonist': int(patient.Protocol_flexible_antagonist)
        }
        
        feature_vector = []
        for name in feature_names:
            feature_vector.append(feature_dict[name])
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Get prediction
        probabilities = model.predict_proba(feature_vector)[0]
        predicted_class = probabilities.argmax()
        
        labels = get_response_labels()
        prediction = labels[predicted_class]
        
        # Clinical interpretation
        if prediction == 'high':
            clinical_note = "Expected good ovarian response. Monitor for OHSS risk."
        elif prediction == 'optimal':
            clinical_note = "Expected balanced response with good outcomes."
        else:
            clinical_note = "Expected lower response. Consider protocol adjustment."
        
        return PredictionResponse(
            prediction=prediction,
            confidence=float(probabilities[predicted_class]),
            probabilities={
                'low': float(probabilities[0]),
                'optimal': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            clinical_note=clinical_note
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "Gradient Boosting Classifier (Calibrated)",
        "features": feature_names,
        "classes": list(get_response_labels().values()),
        "status": "ready"
    }


# Mount static files and serve UI
ui_dir = Path(__file__).parent.parent / "ui"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")
    
    @app.get("/ui")
    async def serve_ui():
        """Serve the web UI"""
        return FileResponse(str(ui_dir / "index.html"))
    
    @app.get("/")
    async def root():
        """Redirect to UI or show API info"""
        return FileResponse(str(ui_dir / "index.html"))
else:
    @app.get("/")
    async def root():
        """API root endpoint"""
        return {
            "message": "IVF Response Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "predict": "/predict",
                "model_info": "/model/info",
                "docs": "/docs"
            }
        }


if __name__ == "__main__":
    import uvicorn
    
    # Try different ports if 8000 is busy
    ports = [8000, 8001, 8002, 8080]
    
    for port in ports:
        try:
            print(f"\nTrying to start server on port {port}...")
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
            break
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                print(f"Port {port} is busy, trying next...")
                continue
            else:
                raise
