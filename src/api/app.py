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
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Global variables for model
model = None
feature_names = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, feature_names, scaler
    
    try:
        # Get the project root
        # Local: /path/to/project/src/api/app.py -> project root is 2 levels up
        # Render: /opt/render/project/src/src/api/app.py -> project root is /opt/render/project/src
        current_file = Path(__file__).resolve()
        
        # Check if we're on Render (has /opt/render/project/src in path)
        if '/opt/render/project/src' in str(current_file):
            # On Render: project root is /opt/render/project/src
            project_root = Path('/opt/render/project/src')
        else:
            # Local development: go up 2 levels from api dir
            project_root = current_file.parent.parent.parent
        
        models_dir = project_root / "src" / "model" / "saved_models"
        data_dir = project_root / "data" / "processed"
        
        print(f"Current file: {current_file}")
        print(f"Project root: {project_root}")
        print(f"Loading model from: {models_dir}")
        print(f"Loading scaler from: {data_dir}")
        
        if not models_dir.exists():
            # Debug: list what's actually there
            print(f"Models dir does not exist. Checking parent dirs:")
            if project_root.exists():
                print(f"  Contents of project_root: {list(project_root.iterdir())}")
                src_check = project_root / "src"
                if src_check.exists():
                    print(f"  Contents of src: {list(src_check.iterdir())}")
                    model_check = src_check / "model"
                    if model_check.exists():
                        print(f"  Contents of model: {list(model_check.iterdir())}")
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        if not data_dir.exists():
            print(f"Data dir does not exist: {data_dir}")
            if project_root.exists():
                data_check = project_root / "data"
                if data_check.exists():
                    print(f"  Contents of data: {list(data_check.iterdir())}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        model = joblib.load(models_dir / "gradient_boosting.pkl")
        feature_info = joblib.load(models_dir / "feature_info.pkl")
        feature_names = feature_info['feature_names']
        scaler = joblib.load(data_dir / "scaler.pkl")
        
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="IVF Response Prediction API",
    description="Predict patient ovarian response to IVF treatment",
    version="1.0.0",
    lifespan=lifespan
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
    response_labels = {
        0: "Low Response",
        1: "Optimal Response", 
        2: "High Response"
    }
    
    return {
        "model_type": "Gradient Boosting Classifier (Calibrated)",
        "features": feature_names,
        "classes": list(response_labels.values()),
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
