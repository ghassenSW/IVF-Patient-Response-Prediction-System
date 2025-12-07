import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import joblib
from dataset import get_response_labels

sns.set_style("whitegrid")


def load_models_and_data():
    """Load all trained models and test data"""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "src" / "model" / "saved_models"
    
    # Load all three trained models
    models = {
        name: joblib.load(models_dir / f"{name}.pkl")
        for name in ['random_forest', 'gradient_boosting', 'logistic_regression']
    }
    
    # Load test split and feature names
    split_data = joblib.load(models_dir / "train_test_split.pkl")
    feature_info = joblib.load(models_dir / "feature_info.pkl")
    
    return models, split_data, feature_info['feature_names']


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance on test set
    Returns dictionary with metrics and predictions
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    return {
        'model': model_name,
        'accuracy': acc,
        'precision': precision,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    """Create and save confusion matrix heatmap"""
    cm = confusion_matrix(y_test, y_pred)
    labels = get_response_labels()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[labels[i] for i in range(3)],
                yticklabels=[labels[i] for i in range(3)])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{model_name}.png', dpi=300)
    plt.close()


def plot_calibration_curve(y_test, y_proba, model_name, output_dir):
    """
    Plot calibration curve showing reliability of probability estimates
    Compares predicted probabilities vs actual frequencies
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = get_response_labels()
    
    # Plot calibration for each class
    for i in range(3):
        y_binary = (y_test == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, y_proba[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=labels[i])
    
    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.set_title(f'Calibration Curve - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'calibration_{model_name}.png', dpi=300)
    plt.close()


def medical_interpretation(results, y_test):
    """
    Provide clinical interpretation of model results
    Shows per-class accuracy and common misclassifications
    """
    print("\n--- Per-Class Results ---")
    labels = get_response_labels()
    
    # Analyze performance for each response category
    for i, label in labels.items():
        mask = (y_test == i)
        correct = (results['predictions'][mask] == i).sum()
        total = mask.sum()
        
        if total > 0:
            acc = correct / total
            print(f"\n{label.capitalize()}: {correct}/{total} ({acc:.1%})")
            
            # Show where model makes mistakes
            misclassified = results['predictions'][mask]
            for j in range(3):
                if j != i:
                    count = (misclassified == j).sum()
                    if count > 0:
                        print(f"  → {labels[j]}: {count}")
    
    # Analyze prediction confidence
    print("\n--- Confidence ---")
    max_probs = results['probabilities'].max(axis=1)
    print(f"Average: {max_probs.mean():.1%}")
    print(f"High (>80%): {(max_probs > 0.8).sum()} patients")


def main():
    print("\n--- Evaluating Models ---\n")
    
    # Load models and data
    models, split_data, feature_names = load_models_and_data()
    X_test = split_data['X_test']
    y_test = split_data['y_test']
def main():
    """Main evaluation pipeline"""
    print("\n--- Evaluating Models ---\n")
    
    # Load all models and test data
    models, split_data, feature_names = load_models_and_data()
    X_test, y_test = split_data['X_test'], split_data['y_test']
    
    print(f"Testing on {len(X_test)} patients")
    
    # Create output directory for plots
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    all_results = []
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        results = evaluate_model(model, X_test, y_test, model_name)
        all_results.append(results)
        
        print(f"  Accuracy: {results['accuracy']:.1%}")
        print(f"  F1-Score: {results['f1_score']:.1%}")
        
        # Generate visualizations
        plot_confusion_matrix(y_test, results['predictions'], model_name, output_dir)
        plot_calibration_curve(y_test, results['probabilities'], model_name, output_dir)
    
    # Compare all models
    print("\n--- Comparison ---")
    comparison_df = pd.DataFrame(all_results)[['model', 'accuracy', 'f1_score']]
    print(comparison_df.to_string(index=False))
    
    # Identify best model
    best_idx = comparison_df['f1_score'].idxmax()
    best_model = comparison_df.iloc[best_idx]['model']
    print(f"\nBest: {best_model}")
    
    # Detailed analysis of best model
    medical_interpretation(all_results[best_idx], y_test)
    
    # Full classification report
    print(f"\n--- Classification Report ---")
    labels = get_response_labels()
    print(classification_report(y_test, all_results[best_idx]['predictions'],
                                target_names=[labels[i] for i in range(3)]))
    
    print(f"\nPlots: {output_dir}")