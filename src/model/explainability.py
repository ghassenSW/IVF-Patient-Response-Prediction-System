import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import shap
from dataset import get_response_labels

shap.initjs()


def load_best_model():
    """Load the best performing model (Gradient Boosting)"""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "src" / "model" / "saved_models"
    
    model = joblib.load(models_dir / "gradient_boosting.pkl")
    split_data = joblib.load(models_dir / "train_test_split.pkl")
    feature_info = joblib.load(models_dir / "feature_info.pkl")
    
    return model, split_data, feature_info['feature_names']


def explain_model_global(model, X_train, feature_names, output_dir):
    """Global feature importance using SHAP"""
    print("Calculating SHAP values...")
    
    # Use KernelExplainer for calibrated models
    sample_data = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(model.predict_proba, sample_data)
    shap_values = explainer.shap_values(X_train[:100])
    
    # Plot 1: SHAP Summary Plot (bar chart) - Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train[:100], feature_names=feature_names, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Beeswarm plot for each class
    if isinstance(shap_values, list):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        labels = ['Low Response', 'Optimal Response', 'High Response']
        
        for idx, (shap_vals, label) in enumerate(zip(shap_values, labels)):
            # Sort features by mean absolute SHAP
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            sorted_idx = mean_abs_shap.argsort()
            
            for feat_idx in sorted_idx:
                y_pos = list(sorted_idx).index(feat_idx)
                axes[idx].scatter(shap_vals[:, feat_idx], 
                                [y_pos] * len(shap_vals[:, feat_idx]),
                                c=X_train[:100, feat_idx], cmap='coolwarm',
                                alpha=0.6, s=20)
            
            axes[idx].set_yticks(range(len(feature_names)))
            axes[idx].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[idx].set_xlabel('SHAP value')
            axes[idx].set_title(label)
            axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Dependence plots for top 3 features
    if isinstance(shap_values, list):
        shap_values_combined = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
    else:
        shap_values_combined = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_values_combined
    }).sort_values('importance', ascending=False)
    
    # Create dependence plots for top 3 features
    top_features = importance_df.head(3)['feature'].tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, feature in enumerate(top_features):
        feature_idx = feature_names.index(feature)
        
        # Use high response class SHAP values for dependence plot
        if isinstance(shap_values, list):
            shap_vals = shap_values[2]  # High response class
        else:
            shap_vals = shap_values
        
        axes[idx].scatter(X_train[:100, feature_idx], shap_vals[:, feature_idx], 
                         alpha=0.6, s=50, c='steelblue')
        axes[idx].set_xlabel(f'{feature} value')
        axes[idx].set_ylabel('SHAP value')
        axes[idx].set_title(f'SHAP Dependence: {feature}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTop features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return explainer, importance_df


def explain_prediction_local(model, patient_data, feature_names, explainer, output_dir):
    """Explain individual prediction with waterfall plot"""
    # Predict
    proba = model.predict_proba([patient_data])[0]
    pred_class = proba.argmax()
    labels = get_response_labels()
    
    print(f"Predicted: {labels[pred_class]} ({proba[pred_class]:.1%})")
    
    # SHAP explanation
    shap_values = explainer.shap_values(patient_data.reshape(1, -1))
    
    print("\nKey factors:")
    if isinstance(shap_values, list):
        shap_vals = shap_values[pred_class][0]
    else:
        shap_vals = shap_values[0]
    
    # Top 3 factors
    abs_shap = np.abs(shap_vals)
    top_idx = abs_shap.argsort()[-3:][::-1]
    
    for idx in top_idx:
        direction = "↑" if shap_vals[idx] > 0 else "↓"
        print(f"  {feature_names[idx]}: {patient_data[idx]:.2f} {direction}")
    
    # Create waterfall plot for this prediction
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by absolute SHAP value
    sorted_idx = abs_shap.argsort()[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_shap = shap_vals[sorted_idx]
    
    # Create waterfall-style visualization
    colors = ['red' if val < 0 else 'blue' for val in sorted_shap]
    ax.barh(range(len(sorted_features)), sorted_shap, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('SHAP Value (Impact on Prediction)')
    ax.set_title(f'Feature Contributions for {labels[pred_class].upper()} Prediction')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_waterfall_individual.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("\n--- Model Explainability (SHAP) ---\n")
    
    # Load model and data
    model, split_data, feature_names = load_best_model()
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    # Create output directory
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "results" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Global explanation
    print("Feature Importance Analysis:")
    explainer, importance_df = explain_model_global(model, X_train, feature_names, output_dir)
    
    # Local explanation - example patient
    print("\n--- Example Prediction ---\n")
    
    # Pick a high-confidence correct prediction
    probas = model.predict_proba(X_test)
    preds = probas.argmax(axis=1)
    max_probas = probas.max(axis=1)
    
    # Find high-confidence high-response patient
    high_mask = (y_test == 2) & (preds == 2) & (max_probas > 0.8)
    if high_mask.sum() > 0:
        example_idx = np.where(high_mask)[0][0]
        print("Example: High response patient")
        explain_prediction_local(model, X_test[example_idx], feature_names, explainer, output_dir)
    
    print(f"\nPlots saved to: {output_dir}")
    print("  - shap_summary_bar.png")
    print("  - shap_summary_beeswarm.png")
    print("  - shap_dependence_plots.png")
    print("  - shap_waterfall_individual.png")
    
    # Medical insights
    print("\n--- Key Findings ---\n")
    
    top_features = importance_df.head(3)
    print("Most important for prediction:")
    for idx, row in top_features.iterrows():
        print(f"  • {row['feature']}")
    
    print("\nThese features are most useful for patient stratification.")


if __name__ == "__main__":
    main()
