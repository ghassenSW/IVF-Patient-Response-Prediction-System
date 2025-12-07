import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
from dataset import load_processed_data, prepare_features_target, get_response_labels


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train multiple probabilistic models"""
    models = {}
    
    print("\nTraining models...")
    # Random Forest
    print("  - Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # Gradient Boosting
    print("  - Gradient Boosting")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
    gb.fit(X_train, y_train)
    models['gradient_boosting'] = gb
    
    # Logistic Regression
    print("  - Logistic Regression")
    lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    return models


def calibrate_models(models, X_train, y_train):
    """Calibrate models for reliable probability estimates"""
    calibrated_models = {}
    
    print("\nCalibrating probabilities...")
    for name, model in models.items():
        calibrated = CalibratedClassifierCV(model, cv=5, method='sigmoid')
        calibrated.fit(X_train, y_train)
        calibrated_models[name] = calibrated
    
    return calibrated_models


def evaluate_train_accuracy(models, X_train, y_train):
    """Quick evaluation on training data"""
    print("\nAccuracy on training set:")
    for name, model in models.items():
        acc = model.score(X_train, y_train)
        print(f"  {name}: {acc:.2%}")


def save_models(models, feature_names):
    """Save trained models"""
    project_root = Path(__file__).parent.parent.parent
    save_dir = project_root / "src" / "model" / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model_path = save_dir / f"{name}.pkl"
        joblib.dump(model, model_path)
    
    # Save feature names
    feature_info = {'feature_names': feature_names}
    joblib.dump(feature_info, save_dir / "feature_info.pkl")


def main():
    print("\n--- Training Models ---\n")
    
    # Load data
    df = load_processed_data()
    X, y, feature_names = prepare_features_target(df)
    
    print(f"Dataset loaded: {len(df)} patients, {len(feature_names)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Calibrate for probability outputs
    calibrated_models = calibrate_models(models, X_train, y_train)
    
    # Evaluate on training data
    evaluate_train_accuracy(calibrated_models, X_train, y_train)
    
    # Save models
    print("\nSaving models...")
    save_models(calibrated_models, feature_names)
    
    # Save train/test split for evaluation
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, Path(__file__).parent / "saved_models" / "train_test_split.pkl")
    
    print("\nDone! Models saved to src/model/saved_models/")
    print("Run evaluate.py to check performance.")


if __name__ == "__main__":
    main()
