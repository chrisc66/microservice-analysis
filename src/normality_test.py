import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_clean_data(file_path):
    """Load data and clean resolution time column."""
    df = pd.read_csv(file_path)
    # Convert resolution time to numeric, removing any non-numeric values
    df['resolution_time_hours'] = pd.to_numeric(df['resolution_time_hours'], errors='coerce')
    # Remove NaN and negative values
    df = df[df['resolution_time_hours'].notna() & (df['resolution_time_hours'] > 0)]
    return df

def remove_outliers(data):
    """Remove outliers using IQR method."""
    Q1 = data['resolution_time_hours'].quantile(0.25)
    Q3 = data['resolution_time_hours'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a copy of the data without outliers
    data_no_outliers = data[
        (data['resolution_time_hours'] >= lower_bound) & 
        (data['resolution_time_hours'] <= upper_bound)
    ].copy()
    
    outliers_count = len(data) - len(data_no_outliers)
    outliers_percentage = (outliers_count / len(data)) * 100
    
    return data_no_outliers, outliers_count, outliers_percentage, (lower_bound, upper_bound)

def analyze_normality(data, title, output_dir):
    """Analyze normality of resolution times using multiple methods."""
    # Create a copy of the data and remove outliers
    data_no_outliers, outliers_count, outliers_percentage, bounds = remove_outliers(data)
    resolution_times = data_no_outliers['resolution_time_hours']
    
    # Create a figure with 3 subplots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Original distribution
    sns.histplot(data=data['resolution_time_hours'], kde=True, ax=ax1)
    ax1.set_title(f'Original Distribution\n{title}')
    ax1.set_xlabel('Resolution Time (hours)')
    ax1.set_ylabel('Count')
    
    # 2. Distribution without outliers
    sns.histplot(data=resolution_times, kde=True, ax=ax2)
    ax2.set_title(f'Distribution Without Outliers\n{title}\n(Removed {outliers_count} outliers, {outliers_percentage:.1f}%)')
    ax2.set_xlabel('Resolution Time (hours)')
    ax2.set_ylabel('Count')
    
    # 3. Q-Q plot (without outliers)
    stats.probplot(resolution_times, dist="norm", plot=ax3)
    ax3.set_title(f'Q-Q Plot (Without Outliers)\n{title}')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f'normality_analysis_{title.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Statistical tests on data without outliers
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(resolution_times)
    
    # D'Agostino's K^2 test
    k2_stat, k2_p = stats.normaltest(resolution_times)
    
    # Basic statistics
    basic_stats = {
        'original_count': len(data),
        'count_after_outlier_removal': len(resolution_times),
        'outliers_removed': outliers_count,
        'outliers_percentage': outliers_percentage,
        'mean': resolution_times.mean(),
        'median': resolution_times.median(),
        'std': resolution_times.std(),
        'skewness': stats.skew(resolution_times),
        'kurtosis': stats.kurtosis(resolution_times),
        'bounds': bounds
    }
    
    return {
        'shapiro_test': {'statistic': shapiro_stat, 'p_value': shapiro_p},
        'dagostino_test': {'statistic': k2_stat, 'p_value': k2_p},
        'basic_stats': basic_stats
    }

def main():
    # Setup paths
    data_dir = Path('data/processed')
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # List of datasets to analyze
    datasets = {
        'Code Only Bugs': 'code_only_bugs.csv',
        'Config Only Bugs': 'config_only_bugs.csv',
        'Mixed Changes Bugs': 'mixed_changes_bugs.csv',
        'All Bugs': 'bug_data.csv'
    }
    
    # Store results
    results = {}
    
    # Analyze each dataset
    for title, filename in datasets.items():
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found")
            continue
            
        print(f"\nAnalyzing {title}...")
        data = load_and_clean_data(file_path)
        results[title] = analyze_normality(data, title, output_dir)
    
    # Print results
    print("\nNormality Analysis Results (After Outlier Removal):")
    print("=" * 80)
    
    for title, result in results.items():
        print(f"\n{title}:")
        print("-" * 40)
        
        # Basic statistics
        stats = result['basic_stats']
        print(f"Original sample size: {stats['original_count']}")
        print(f"Outliers removed: {stats['outliers_removed']} ({stats['outliers_percentage']:.1f}%)")
        print(f"Final sample size: {stats['count_after_outlier_removal']}")
        print(f"Outlier bounds: [{stats['bounds'][0]:.2f}, {stats['bounds'][1]:.2f}] hours")
        print(f"Mean: {stats['mean']:.2f} hours")
        print(f"Median: {stats['median']:.2f} hours")
        print(f"Standard deviation: {stats['std']:.2f}")
        print(f"Skewness: {stats['skewness']:.2f}")
        print(f"Kurtosis: {stats['kurtosis']:.2f}")
        
        # Shapiro-Wilk test results
        shapiro = result['shapiro_test']
        print(f"\nShapiro-Wilk test:")
        print(f"Statistic: {shapiro['statistic']:.4f}")
        print(f"p-value: {shapiro['p_value']:.4e}")
        print(f"Conclusion: {'Not normal' if shapiro['p_value'] < 0.05 else 'Normal'} distribution (α=0.05)")
        
        # D'Agostino's K^2 test results
        dagostino = result['dagostino_test']
        print(f"\nD'Agostino's K^2 test:")
        print(f"Statistic: {dagostino['statistic']:.4f}")
        print(f"p-value: {dagostino['p_value']:.4e}")
        print(f"Conclusion: {'Not normal' if dagostino['p_value'] < 0.05 else 'Normal'} distribution (α=0.05)")
        
        print("\nVisual analysis plots have been saved to the analysis_results directory.")

if __name__ == "__main__":
    main() 