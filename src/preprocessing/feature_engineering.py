import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import joblib


def main():
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "processed" / "patients_clean.csv"
    output_dir = project_root / "data" / "processed"
    
    # Load cleaned data
    df = pd.read_csv(input_path)
    
    # Handle missing values with median
    numerical_cols = ['Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC']
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Drop rows with missing target
    df = df.dropna(subset=['Patient Response'])
    
    # One-hot encode protocol
    protocol_dummies = pd.get_dummies(df['Protocol'], prefix='Protocol')
    df = pd.concat([df, protocol_dummies], axis=1)
    
    # Label encode target
    response_mapping = {'low': 0, 'optimal': 1, 'high': 2}
    df['Patient_Response_Encoded'] = df['Patient Response'].map(response_mapping)
    
    # Normalize numerical features
    feature_cols = ['cycle_number', 'Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC']
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Reorder columns
    protocol_cols = [col for col in df.columns if col.startswith('Protocol_')]
    column_order = ['patient_id', 'cycle_number', 'Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC'] + protocol_cols + ['Patient Response', 'Patient_Response_Encoded']
    df_final = df[column_order]
    
    # Save processed data
    output_path = output_dir / "patients_processed.csv"
    df_final.to_csv(output_path, index=False)
    
    # Save scaler
    joblib.dump(scaler, output_dir / "scaler.pkl")
    
    # Save preprocessing info
    feature_info = {
        'feature_columns': feature_cols + protocol_cols,
        'response_mapping': response_mapping,
        'inverse_response_mapping': {v: k for k, v in response_mapping.items()}
    }
    with open(output_dir / "preprocessing_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
