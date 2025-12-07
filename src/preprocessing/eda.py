import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def main():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "patients_clean.csv"
    output_dir = project_root / "results" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    print(f"\nDataset: {df.shape[0]} patients, {df.shape[1]} features\n")
    
    # Analysis 1: Feature correlations
    response_map = {'low': 0, 'optimal': 1, 'high': 2}
    df['Response_Encoded'] = df['Patient Response'].map(response_map)
    
    numerical_cols = ['Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC']
    correlations = df[numerical_cols].corrwith(df['Response_Encoded']).sort_values(ascending=False)
    
    print("Top Predictive Features:")
    for feature, corr in correlations.head(3).items():
        print(f"  {feature}: {corr:.3f}")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df[numerical_cols + ['Response_Encoded']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
    plt.close()
    
    # Analysis 2: AMH vs Response patterns
    df['AMH_Category'] = pd.cut(df['AMH'], bins=[0, 1, 2, 4, 10], 
                                  labels=['Very Low', 'Low', 'Normal', 'High'])
    amh_response = pd.crosstab(df['AMH_Category'], df['Patient Response'], normalize='index') * 100
    
    print("\nAMH Categories vs Response (%):")
    print(amh_response.round(1))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    amh_response.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c', '#ff7f0e'])
    ax.set_title('AMH Level Impact on Patient Response')
    ax.set_xlabel('AMH Category')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Response')
    plt.tight_layout()
    plt.savefig(output_dir / 'amh_patterns.png', dpi=300)
    plt.close()
    
    # Analysis 3: Response distribution (confusion matrix style)
    response_counts = df['Patient Response'].value_counts()
    
    print("\nPatient Response Distribution:")
    for response in ['low', 'optimal', 'high']:
        count = response_counts.get(response, 0)
        pct = (count / len(df)) * 100
        print(f"  {response}: {count} ({pct:.1f}%)")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    response_counts.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c', '#ff7f0e'])
    ax.set_title('Patient Response Categories')
    ax.set_xlabel('Response')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'response_distribution.png', dpi=300)
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}\n")


if __name__ == "__main__":
    main()
