import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json


def load_processed_data():
    """Load preprocessed data with features and target"""
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "patients_processed.csv"
    
    df = pd.read_csv(data_path)
    return df


def load_preprocessing_artifacts():
    """Load scaler and preprocessing info"""
    project_root = Path(__file__).parent.parent.parent
    artifacts_dir = project_root / "data" / "processed"
    
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    
    with open(artifacts_dir / "preprocessing_info.json", 'r') as f:
        info = json.load(f)
    
    return scaler, info


def prepare_features_target(df):
    """Separate features and target from dataframe"""
    # Get feature columns (exclude patient_id and target columns)
    feature_cols = [col for col in df.columns if col not in ['patient_id', 'Patient Response', 'Patient_Response_Encoded']]
    
    X = df[feature_cols].values
    y = df['Patient_Response_Encoded'].values
    
    return X, y, feature_cols


def get_response_labels():
    """Return response mapping"""
    return {
        0: 'low',
        1: 'optimal', 
        2: 'high'
    }


def main():
    # Load and display data info
    df = load_processed_data()
    print(f"Loaded {len(df)} patients")
    
    X, y, features = prepare_features_target(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures: {features}")
    
    # Response distribution
    labels = get_response_labels()
    print("\nTarget distribution:")
    for i in range(3):
        count = (y == i).sum()
        pct = (count / len(y)) * 100
        print(f"  {labels[i]}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
