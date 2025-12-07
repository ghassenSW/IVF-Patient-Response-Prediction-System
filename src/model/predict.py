import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from dataset import get_response_labels


def load_model():
    """Load best performing model"""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "src" / "model" / "saved_models"
    
    model = joblib.load(models_dir / "gradient_boosting.pkl")
    feature_info = joblib.load(models_dir / "feature_info.pkl")
    scaler = joblib.load(project_root / "data" / "processed" / "scaler.pkl")
    
    return model, feature_info, scaler


def predict_patient_response(patient_features, model, feature_names):
    """
    Predict patient response with probabilities
    
    Args:
        patient_features: dict with feature values
        model: trained model
        feature_names: list of feature names in correct order
    
    Returns:
        dict with prediction and probabilities
    """
    # Prepare feature vector in correct order
    feature_vector = []
    for name in feature_names:
        if name in patient_features:
            feature_vector.append(patient_features[name])
        else:
            feature_vector.append(0)  # Missing features as 0
    
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    # Predict probabilities
    probabilities = model.predict_proba(feature_vector)[0]
    predicted_class = probabilities.argmax()
    
    labels = get_response_labels()
    
    return {
        'prediction': labels[predicted_class],
        'probabilities': {
            'low': probabilities[0],
            'optimal': probabilities[1],
            'high': probabilities[2]
        },
        'confidence': probabilities[predicted_class]
    }


def print_prediction_report(result):
    """Print formatted prediction report"""
    print("\n--- Prediction Result ---\n")
    
    print(f"Predicted: {result['prediction'].upper()} response")
    print(f"Confidence: {result['confidence']:.1%}\n")
    
    print("Probabilities:")
    for category, prob in result['probabilities'].items():
        bar = "█" * int(prob * 40)
        print(f"  {category:8s} {prob:>5.1%}  {bar}")
    
    # Medical interpretation
    print("\n--- Clinical Note ---\n")
    
    pred = result['prediction']
    conf = result['confidence']
    
    if pred == 'high':
        print("Expected good ovarian response.")
        print("Monitor for OHSS risk.")
    elif pred == 'optimal':
        print("Expected balanced response with good outcomes.")
    else:  # low
        print("Expected lower response.")
        print("Consider protocol adjustment.")
    
    if conf < 0.6:
        print(f"\nNote: Lower confidence ({conf:.1%}) - consider additional assessment.")


def main():
    # Load model
    model, feature_info, scaler = load_model()
    feature_names = feature_info['feature_names']
    
    # Example patient data
    example_patient = {
        'cycle_number': -0.67,  # Normalized value (1st cycle)
        'Age': 0.06,             # Normalized (32 years)
        'AMH': 1.56,             # Normalized (high AMH)
        'n_Follicles': 1.33,
        'E2_day5': -0.12,
        'AFC': 2.87,
        'Protocol_agonist': False,
        'Protocol_fixed antagonist': False,
        'Protocol_flexible antagonist': True
    }
    
    print("\n--- Example Prediction ---\n")
    print("Patient: 32 years, high AMH, many follicles")
    
    # Predict
    result = predict_patient_response(example_patient, model, feature_names)
    
    # Print report
    print_prediction_report(result)
    

if __name__ == "__main__":
    main()
