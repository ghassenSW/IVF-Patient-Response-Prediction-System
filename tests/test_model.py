"""
Unit tests for the IVF response prediction model
Tests model loading, prediction functionality, and API endpoints
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model.dataset import load_processed_data, prepare_features_target, get_response_labels


class TestDataLoading:
    """Test data loading and preparation"""
    
    def test_load_processed_data(self):
        """Test that processed data loads correctly"""
        df = load_processed_data()
        assert len(df) > 0, "Dataset should not be empty"
        assert 'Patient_Response_Encoded' in df.columns, "Target column missing"
    
    def test_prepare_features(self):
        """Test feature preparation"""
        df = load_processed_data()
        X, y, features = prepare_features_target(df)
        
        assert X.shape[0] == len(df), "Feature rows should match dataset"
        assert len(y) == len(df), "Target should match dataset length"
        assert len(features) > 0, "Should have feature names"
    
    def test_response_labels(self):
        """Test response label mapping"""
        labels = get_response_labels()
        assert len(labels) == 3, "Should have 3 response classes"
        assert labels[0] == 'low', "Class 0 should be 'low'"
        assert labels[1] == 'optimal', "Class 1 should be 'optimal'"
        assert labels[2] == 'high', "Class 2 should be 'high'"


class TestModel:
    """Test trained model functionality"""
    
    def test_model_exists(self):
        """Test that trained model file exists"""
        model_path = Path(__file__).parent.parent / "src" / "model" / "saved_models" / "gradient_boosting.pkl"
        assert model_path.exists(), "Trained model should exist"
    
    def test_model_prediction(self):
        """Test model can make predictions"""
        import joblib
        
        model_path = Path(__file__).parent.parent / "src" / "model" / "saved_models" / "gradient_boosting.pkl"
        model = joblib.load(model_path)
        
        # Create sample input
        sample = np.array([[0.0, -0.5, 1.56, 1.33, -0.12, 2.87, 0, 0, 1]])
        
        # Test prediction
        pred = model.predict(sample)
        assert pred.shape[0] == 1, "Should predict for one sample"
        assert pred[0] in [0, 1, 2], "Prediction should be valid class"
        
        # Test probability prediction
        proba = model.predict_proba(sample)
        assert proba.shape == (1, 3), "Should return probabilities for 3 classes"
        assert np.isclose(proba.sum(), 1.0), "Probabilities should sum to 1"


class TestAPI:
    """Test API endpoints"""
    
    def test_api_health(self):
        """Test API health endpoint"""
        import requests
        
        try:
            response = requests.get("http://localhost:8000/", timeout=2)
            assert response.status_code == 200, "Health check should return 200"
            data = response.json()
            assert data['status'] == 'online', "Status should be online"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_api_predict(self):
        """Test API prediction endpoint"""
        import requests
        
        patient_data = {
            "cycle_number": 0.0,
            "Age": -0.5,
            "AMH": 1.56,
            "n_Follicles": 1.33,
            "E2_day5": -0.12,
            "AFC": 2.87,
            "Protocol_agonist": False,
            "Protocol_fixed antagonist": False,
            "Protocol_flexible antagonist": True
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=patient_data, timeout=5)
            assert response.status_code == 200, "Prediction should return 200"
            
            data = response.json()
            assert 'prediction' in data, "Response should contain prediction"
            assert 'confidence' in data, "Response should contain confidence"
            assert 'probabilities' in data, "Response should contain probabilities"
            assert data['prediction'] in ['low', 'optimal', 'high'], "Prediction should be valid"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
