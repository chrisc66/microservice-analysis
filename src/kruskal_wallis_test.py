import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations

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
    
    return data_no_outliers, (lower_bound, upper_bound)

def perform_kruskal_wallis_test(groups_data):
    """Perform Kruskal-Wallis H-test on the groups."""
    # Extract resolution times for each group
    resolution_times = [group['resolution_time_hours'].values for group in groups_data]
    
    # Perform Kruskal-Wallis H-test
    h_statistic, p_value = stats.kruskal(*resolution_times)
    
    return h_statistic, p_value

def perform_mann_whitney_tests(groups_data, group_names):
    """Perform pairwise Mann-Whitney U tests with Bonferroni correction."""
    results = []
    
    # Get all pairwise combinations
    pairs = list(combinations(range(len(groups_data)), 2))
    
    # Bonferroni correction
    alpha = 0.05
    alpha_corrected = alpha / len(pairs)
    
    for i, j in pairs:
        stat, p_value = stats.mannwhitneyu(
            groups_data[i]['resolution_time_hours'],
            groups_data[j]['resolution_time_hours'],
            alternative='two-sided'
        )
        
        results.append({
            'Group 1': group_names[i],
            'Group 2': group_names[j],
            'Statistic': stat,
            'p-value': p_value,
            'Significant': p_value < alpha_corrected
        })
    
    return results, alpha_corrected

def create_posthoc_heatmap(posthoc_results, group_names, output_dir):
    """Create a heatmap visualization for post-hoc test results."""
    # Create a matrix for p-values
    n_groups = len(group_names)
    p_values = np.ones((n_groups, n_groups))
    
    # Fill the matrix with p-values from the results
    for result in posthoc_results:
        i = group_names.index(result['Group 1'])
        j = group_names.index(result['Group 2'])
        p_values[i, j] = result['p-value']
        p_values[j, i] = result['p-value']  # Mirror the matrix
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        p_values,
        xticklabels=group_names,
        yticklabels=group_names,
        annot=True,
        fmt='.2e',  # Scientific notation for p-values
        cmap='RdYlBu_r',  # Red for significant, blue for non-significant
        vmin=0,
        vmax=0.05,  # Highlight differences up to α=0.05
        square=True
    )
    
    plt.title('Post-hoc Analysis: P-values from Mann-Whitney U Tests\n(Significant if p < 0.0167)', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'posthoc_analysis_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(groups_data, group_names, output_dir):
    """Create visualizations to compare distributions."""
    # Set the style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Prepare data for plotting
    all_data = []
    for group_data, group_name in zip(groups_data, group_names):
        group_df = pd.DataFrame({
            'Resolution Time (hours)': group_data['resolution_time_hours'],
            'Group': group_name
        })
        all_data.append(group_df)
    
    plot_data = pd.concat(all_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Box plot
    sns.boxplot(data=plot_data, x='Group', y='Resolution Time (hours)', ax=ax1)
    ax1.set_title('Resolution Time Distribution by Group', pad=20)
    
    # Rotate x-axis labels
    ax1.set_xticklabels(group_names, rotation=45, ha='right')
    
    # 2. Violin plot
    sns.violinplot(data=plot_data, x='Group', y='Resolution Time (hours)', ax=ax2)
    ax2.set_title('Resolution Time Distribution by Group (Violin Plot)', pad=20)
    
    # Rotate x-axis labels
    ax2.set_xticklabels(group_names, rotation=45, ha='right')
    
    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(output_dir / 'resolution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_descriptive_stats(groups_data, group_names):
    """Calculate descriptive statistics for each group."""
    stats_data = []
    for group_data, group_name in zip(groups_data, group_names):
        stats_dict = {
            'Group': group_name,
            'Count': len(group_data),
            'Mean': group_data['resolution_time_hours'].mean(),
            'Median': group_data['resolution_time_hours'].median(),
            'Std': group_data['resolution_time_hours'].std(),
            'IQR': group_data['resolution_time_hours'].quantile(0.75) - group_data['resolution_time_hours'].quantile(0.25),
            '25th': group_data['resolution_time_hours'].quantile(0.25),
            '75th': group_data['resolution_time_hours'].quantile(0.75)
        }
        stats_data.append(stats_dict)
    
    return pd.DataFrame(stats_data)

def format_stats_output(stats_df):
    """Format the statistics output for better readability."""
    stats_df['Mean'] = stats_df['Mean'].round(2)
    stats_df['Median'] = stats_df['Median'].round(2)
    stats_df['Std'] = stats_df['Std'].round(2)
    stats_df['IQR'] = stats_df['IQR'].round(2)
    stats_df['25th'] = stats_df['25th'].round(2)
    stats_df['75th'] = stats_df['75th'].round(2)
    return stats_df

def main():
    # Setup paths
    data_dir = Path('data/processed')
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Define groups
    groups = {
        'Code Only': 'code_only_bugs.csv',
        'Config Only': 'config_only_bugs.csv',
        'Mixed Changes': 'mixed_changes_bugs.csv'
    }
    
    # Load and prepare data
    groups_data = []
    group_names = []
    
    print("\nLoading and preparing data...")
    for group_name, filename in groups.items():
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found")
            continue
        
        # Load and clean data
        data = load_and_clean_data(file_path)
        print(f"\n{group_name}:")
        print(f"Original sample size: {len(data)}")
        
        # Remove outliers
        data_no_outliers, bounds = remove_outliers(data)
        print(f"Sample size after outlier removal: {len(data_no_outliers)}")
        print(f"Outlier bounds: [{bounds[0]:.2f}, {bounds[1]:.2f}] hours")
        
        groups_data.append(data_no_outliers)
        group_names.append(group_name)
    
    # Calculate and format descriptive statistics
    print("\nDescriptive Statistics:")
    print("=" * 80)
    stats_df = calculate_descriptive_stats(groups_data, group_names)
    stats_df = format_stats_output(stats_df)
    print(stats_df.to_string(index=False))
    
    # Perform Kruskal-Wallis test
    h_statistic, p_value = perform_kruskal_wallis_test(groups_data)
    
    print("\nKruskal-Wallis Test Results:")
    print("=" * 80)
    print(f"H-statistic: {h_statistic:.4f}")
    print(f"p-value: {p_value:.4e}")
    print(f"\nConclusion: There {'is' if p_value < 0.05 else 'is not'} a significant difference")
    print(f"in resolution time distributions between the groups (α=0.05).")
    
    if p_value < 0.05:
        # Perform post-hoc Mann-Whitney U tests
        print("\nPost-hoc Analysis (Mann-Whitney U tests with Bonferroni correction):")
        print("=" * 80)
        posthoc_results, alpha_corrected = perform_mann_whitney_tests(groups_data, group_names)
        
        print(f"\nBonferroni-corrected significance level: α = {alpha_corrected:.4f}")
        print("\nPairwise Comparisons:")
        for result in posthoc_results:
            print(f"\n{result['Group 1']} vs {result['Group 2']}:")
            print(f"U-statistic: {result['Statistic']:.4f}")
            print(f"p-value: {result['p-value']:.4e}")
            print(f"Significant difference: {'Yes' if result['Significant'] else 'No'}")
        
        # Create heatmap for post-hoc results
        create_posthoc_heatmap(posthoc_results, group_names, output_dir)
        print("\nPost-hoc analysis heatmap has been saved to the analysis_results directory.")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(groups_data, group_names, output_dir)
    print("Visualizations have been saved to the analysis_results directory.")

if __name__ == "__main__":
    main() 