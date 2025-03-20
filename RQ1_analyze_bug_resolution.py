import pandas as pd
from scipy.stats import ttest_ind

# Load the dataset
data = pd.read_csv("./data/processed/bug_data.csv")

# Define the fix metrics and target variable
fix_metrics = [
    "config_files_changed",
    "app_code_files_changed",
    "other_files_changed",
    "total_files_changed",
    "config_files_lines_changed",
    "app_code_files_lines_changed",
]
target = "resolution_time_hours"

# Perform t-tests for each fix metric
results = []
for metric in fix_metrics:
    # Split data into two groups: high and low values of the metric
    median_value = data[metric].median()
    group_high = data[data[metric] > median_value][target]
    group_low = data[data[metric] <= median_value][target]

    # Perform t-test
    t_stat, p_value = ttest_ind(group_high, group_low, equal_var=False, nan_policy="omit")

    # Store results
    results.append({"metric": metric, "t_stat": t_stat, "p_value": p_value, "reject_null": p_value < 0.05})

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# Save results to a CSV file
results_df.to_csv("./data/processed/RQ1_t_test_results.csv", index=False)
