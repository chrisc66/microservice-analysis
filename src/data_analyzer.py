"""
Module for analyzing bug data and generating insights about resolution times.
"""

import os
import logging
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro
from typing import Dict, List, Tuple, Optional

from config.config import PROCESSED_DATA_DIR, ANALYSIS_RESULTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BugDataAnalyzer:
    def __init__(self, data_file: str = None):
        """
        Initialize the bug data analyzer.
        
        Args:
            data_file: Path to the bug data CSV file
        """
        if not data_file:
            data_file = os.path.join(PROCESSED_DATA_DIR, 'bug_data.csv')
            
        self.df = pd.read_csv(data_file)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean the data for analysis."""
        # Convert datetime columns
        datetime_cols = ['issue_created_at', 'issue_closed_at', 'pr_merged_at']
        for col in datetime_cols:
            self.df[col] = pd.to_datetime(self.df[col])
            
        # Remove outliers in resolution time (e.g., extremely long times)
        resolution_times = self.df['resolution_time_hours']
        q1 = resolution_times.quantile(0.25)
        q3 = resolution_times.quantile(0.75)
        iqr = q3 - q1
        
        self.df = self.df[
            (resolution_times >= q1 - 1.5 * iqr) &
            (resolution_times <= q3 + 1.5 * iqr)
        ]
        
        # Create categories for analysis
        self.df['primary_change'] = self.df.apply(
            lambda row: 'config' if row['config_files_changed'] > row['app_code_files_changed']
            else 'app_code', axis=1
        )
        
        # Calculate total files changed
        self.df['total_files_changed'] = (
            self.df['config_files_changed'] + 
            self.df['app_code_files_changed'] + 
            self.df['other_files_changed']
        )
        
    def calculate_basic_stats(self) -> Dict:
        """
        Calculate basic statistics about bug resolution times.
        
        Returns:
            Dict: Dictionary containing basic statistics
        """
        stats_dict = {
            'total_bugs': len(self.df),
            'config_bugs': len(self.df[self.df['primary_change'] == 'config']),
            'app_code_bugs': len(self.df[self.df['primary_change'] == 'app_code']),
            'mean_resolution_time': self.df['resolution_time_hours'].mean(),
            'median_resolution_time': self.df['resolution_time_hours'].median(),
            'config_mean_time': self.df[self.df['primary_change'] == 'config']['resolution_time_hours'].mean(),
            'app_code_mean_time': self.df[self.df['primary_change'] == 'app_code']['resolution_time_hours'].mean(),
            'std_resolution_time': self.df['resolution_time_hours'].std(),
            'config_std_time': self.df[self.df['primary_change'] == 'config']['resolution_time_hours'].std(),
            'app_code_std_time': self.df[self.df['primary_change'] == 'app_code']['resolution_time_hours'].std()
        }
        
        return stats_dict

    def test_normality(self) -> Tuple[bool, bool]:
        """
        Test if resolution times follow normal distribution using Shapiro-Wilk test.
        
        Returns:
            Tuple[bool, bool]: (is_config_normal, is_app_normal)
        """
        config_times = self.df[self.df['primary_change'] == 'config']['resolution_time_hours']
        app_times = self.df[self.df['primary_change'] == 'app_code']['resolution_time_hours']
        
        # Shapiro-Wilk test
        _, config_p = shapiro(config_times)
        _, app_p = shapiro(app_times)
        
        return config_p > 0.05, app_p > 0.05

    def perform_statistical_test(self) -> Tuple[float, float, str]:
        """
        Perform appropriate statistical test based on normality.
        
        Returns:
            Tuple[float, float, str]: (statistic, p_value, test_name)
        """
        config_times = self.df[self.df['primary_change'] == 'config']['resolution_time_hours']
        app_times = self.df[self.df['primary_change'] == 'app_code']['resolution_time_hours']
        
        is_config_normal, is_app_normal = self.test_normality()
        
        if is_config_normal and is_app_normal:
            # Use t-test if both are normal
            stat, p_value = stats.ttest_ind(config_times, app_times)
            test_name = "Independent t-test"
        else:
            # Use Mann-Whitney U test if either is non-normal
            stat, p_value = mannwhitneyu(config_times, app_times, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            
        return stat, p_value, test_name

    def calculate_effect_size(self) -> Tuple[float, str]:
        """
        Calculate effect size using Cohen's d.
        
        Returns:
            Tuple[float, str]: (effect_size, interpretation)
        """
        config_times = self.df[self.df['primary_change'] == 'config']['resolution_time_hours']
        app_times = self.df[self.df['primary_change'] == 'app_code']['resolution_time_hours']
        
        # Pooled standard deviation
        n1, n2 = len(config_times), len(app_times)
        var1, var2 = np.var(config_times, ddof=1), np.var(app_times, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(config_times) - np.mean(app_times)) / pooled_se
        
        # Interpret effect size
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
            
        return d, interpretation

    def calculate_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate 95% confidence intervals for mean resolution times.
        
        Returns:
            Dict[str, Tuple[float, float]]: Confidence intervals for each group
        """
        config_times = self.df[self.df['primary_change'] == 'config']['resolution_time_hours']
        app_times = self.df[self.df['primary_change'] == 'app_code']['resolution_time_hours']
        
        # 95% CI for each group
        config_ci = stats.t.interval(
            alpha=0.95,
            df=len(config_times)-1,
            loc=np.mean(config_times),
            scale=stats.sem(config_times)
        )
        
        app_ci = stats.t.interval(
            alpha=0.95,
            df=len(app_times)-1,
            loc=np.mean(app_times),
            scale=stats.sem(app_times)
        )
        
        return {'config': config_ci, 'app_code': app_ci}

    def analyze_size_impact(self) -> Tuple[float, float]:
        """
        Analyze if repository size affects resolution time differences.
        
        Returns:
            Tuple[float, float]: (correlation_coefficient, p_value)
        """
        correlation, p_value = stats.spearmanr(
            self.df['total_files_changed'],
            self.df['resolution_time_hours']
        )
        return correlation, p_value

    def plot_resolution_time_distribution(self):
        """Generate plots comparing resolution times between config and app code changes."""
        plt.figure(figsize=(15, 10))
        
        # Box plot
        plt.subplot(2, 2, 1)
        sns.boxplot(x='primary_change', y='resolution_time_hours', data=self.df)
        plt.title('Bug Resolution Time Distribution')
        plt.xlabel('Primary Change Type')
        plt.ylabel('Resolution Time (hours)')
        
        # Violin plot
        plt.subplot(2, 2, 2)
        sns.violinplot(x='primary_change', y='resolution_time_hours', data=self.df)
        plt.title('Bug Resolution Time Density')
        plt.xlabel('Primary Change Type')
        plt.ylabel('Resolution Time (hours)')
        
        # Q-Q plot for config changes
        plt.subplot(2, 2, 3)
        stats.probplot(
            self.df[self.df['primary_change'] == 'config']['resolution_time_hours'],
            dist="norm",
            plot=plt
        )
        plt.title('Q-Q Plot: Config Changes')
        
        # Q-Q plot for app code changes
        plt.subplot(2, 2, 4)
        stats.probplot(
            self.df[self.df['primary_change'] == 'app_code']['resolution_time_hours'],
            dist="norm",
            plot=plt
        )
        plt.title('Q-Q Plot: App Code Changes')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, 'resolution_time_distribution.png'))
        plt.close()

    def plot_monthly_trends(self):
        """Generate plot showing monthly trends in bug resolution times."""
        # Extract month from created_at
        self.df['month'] = self.df['issue_created_at'].dt.to_period('M')
        
        # Calculate monthly averages and confidence intervals
        monthly_stats = self.df.groupby(['month', 'primary_change'])['resolution_time_hours'].agg([
            'mean',
            'std',
            'count'
        ]).reset_index()
        
        # Calculate 95% CI
        monthly_stats['ci'] = 1.96 * monthly_stats['std'] / np.sqrt(monthly_stats['count'])
        
        # Pivot for plotting
        monthly_means = monthly_stats.pivot(index='month', columns='primary_change', values='mean')
        monthly_ci = monthly_stats.pivot(index='month', columns='primary_change', values='ci')
        
        plt.figure(figsize=(15, 6))
        
        # Plot means with confidence intervals
        for change_type in ['config', 'app_code']:
            plt.plot(range(len(monthly_means)), monthly_means[change_type], 
                    marker='o', label=change_type)
            plt.fill_between(
                range(len(monthly_means)),
                monthly_means[change_type] - monthly_ci[change_type],
                monthly_means[change_type] + monthly_ci[change_type],
                alpha=0.2
            )
        
        plt.title('Monthly Average Bug Resolution Time by Change Type')
        plt.xlabel('Month')
        plt.ylabel('Average Resolution Time (hours)')
        plt.legend(title='Change Type')
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, 'monthly_trends.png'))
        plt.close()

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        # Calculate all statistics
        stats = self.calculate_basic_stats()
        is_config_normal, is_app_normal = self.test_normality()
        stat, p_value, test_name = self.perform_statistical_test()
        effect_size, effect_interpretation = self.calculate_effect_size()
        confidence_intervals = self.calculate_confidence_intervals()
        size_correlation, size_p_value = self.analyze_size_impact()
        
        # Generate plots
        self.plot_resolution_time_distribution()
        self.plot_monthly_trends()
        
        # Create report
        report = f"""
# Bug Resolution Time Analysis Report

## Summary Statistics
- Total bugs analyzed: {stats['total_bugs']}
- Configuration file changes: {stats['config_bugs']} ({stats['config_bugs']/stats['total_bugs']*100:.1f}%)
- Application code changes: {stats['app_code_bugs']} ({stats['app_code_bugs']/stats['total_bugs']*100:.1f}%)

## Resolution Times
- Configuration changes:
  * Mean: {stats['config_mean_time']:.1f} hours
  * Standard deviation: {stats['config_std_time']:.1f} hours
  * 95% CI: [{confidence_intervals['config'][0]:.1f}, {confidence_intervals['config'][1]:.1f}] hours
- Application code changes:
  * Mean: {stats['app_code_mean_time']:.1f} hours
  * Standard deviation: {stats['app_code_std_time']:.1f} hours
  * 95% CI: [{confidence_intervals['app_code'][0]:.1f}, {confidence_intervals['app_code'][1]:.1f}] hours

## Normality Analysis
- Configuration changes: {'Normal' if is_config_normal else 'Non-normal'} distribution
- Application code changes: {'Normal' if is_app_normal else 'Non-normal'} distribution

## Statistical Analysis
- Test used: {test_name}
- Test statistic: {stat:.3f}
- P-value: {p_value:.3f}
- Statistical significance: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)
- Effect size (Cohen's d): {effect_size:.3f}
- Effect size interpretation: {effect_interpretation}

## Repository Size Impact
- Correlation coefficient: {size_correlation:.3f}
- P-value: {size_p_value:.3f}
- Interpretation: {'Significant' if size_p_value < 0.05 else 'Not significant'} correlation between repository size and resolution time

## Key Findings
1. {'Configuration changes tend to be resolved faster than application code changes' 
   if stats['config_mean_time'] < stats['app_code_mean_time'] 
   else 'Application code changes tend to be resolved faster than configuration changes'}
   
2. The difference in resolution times is{' ' if p_value < 0.05 else ' not '}statistically significant

3. The effect size is {effect_interpretation}, indicating {
    'a substantial practical difference' if abs(effect_size) >= 0.8
    else 'a moderate practical difference' if abs(effect_size) >= 0.5
    else 'a small practical difference' if abs(effect_size) >= 0.2
    else 'no practical difference'} between the groups

## Visualizations
Three visualization files have been generated:
1. resolution_time_distribution.png:
   - Box plots showing distribution of resolution times
   - Violin plots showing probability density
   - Q-Q plots for normality assessment
2. monthly_trends.png:
   - Time series of monthly average resolution times
   - 95% confidence intervals shown as shaded regions

## Recommendations
1. {'Consider prioritizing configuration-related bugs as they typically require less time to resolve' 
   if stats['config_mean_time'] < stats['app_code_mean_time'] 
   else 'Consider allocating more resources to configuration-related bugs as they typically require more time to resolve'}
2. Use these insights to better estimate bug resolution times based on the type of change required
3. Consider implementing automated configuration validation to prevent configuration-related bugs
4. {'Consider repository size when estimating resolution times, as it shows a significant correlation' 
   if size_p_value < 0.05 
   else 'Repository size does not significantly impact resolution times'}
"""
        
        # Save report
        os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)
        with open(os.path.join(ANALYSIS_RESULTS_DIR, 'analysis_report.md'), 'w') as f:
            f.write(report)
            
        logger.info("Analysis report generated successfully")

def main():
    """Main function to run the bug data analyzer."""
    try:
        analyzer = BugDataAnalyzer()
        analyzer.generate_report()
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 