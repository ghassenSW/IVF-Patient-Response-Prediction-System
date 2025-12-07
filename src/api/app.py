"""
FastAPI application for IVF patient response prediction
Provides endpoints for model inference with probability outputs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import joblib
import numpy as np
from pathlib import Path
import sys

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
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None
feature_names = None

@app.on_event("startup")
async def load_model():
    """Load trained model and feature configuration"""
    global model, feature_names
    
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "src" / "model" / "saved_models"
    
    model = joblib.load(models_dir / "gradient_boosting.pkl")
    feature_info = joblib.load(models_dir / "feature_info.pkl")
    feature_names = feature_info['feature_names']
    
    print("Model loaded successfully!")


class PatientFeatures(BaseModel):
    """Input features for prediction (normalized values)"""
    cycle_number: float = Field(..., description="Treatment cycle number (normalized)")
    Age: float = Field(..., description="Patient age (normalized)")
    AMH: float = Field(..., description="Anti-Müllerian hormone level (normalized)")
    n_Follicles: float = Field(..., description="Number of follicles (normalized)")
    E2_day5: float = Field(..., description="Estradiol level day 5 (normalized)")
    AFC: float = Field(..., description="Antral follicle count (normalized)")
    Protocol_agonist: bool = Field(False, description="Using agonist protocol", alias="Protocol_agonist")
    Protocol_fixed_antagonist: bool = Field(False, description="Using fixed antagonist protocol", alias="Protocol_fixed antagonist")
    Protocol_flexible_antagonist: bool = Field(False, description="Using flexible antagonist protocol", alias="Protocol_flexible antagonist")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "cycle_number": 0.0,
                "Age": -0.5,
                "AMH": 1.2,
                "n_Follicles": 0.8,
                "E2_day5": 0.3,
                "AFC": 1.5,
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
    
    Returns prediction with probabilities and clinical interpretation
    """
    try:
        # Prepare feature vector - manually map to exact feature names
        # Model expects spaces in protocol names
        feature_dict = {
            'cycle_number': patient.cycle_number,
            'Age': patient.Age,
            'AMH': patient.AMH,
            'n_Follicles': patient.n_Follicles,
            'E2_day5': patient.E2_day5,
            'AFC': patient.AFC,
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
