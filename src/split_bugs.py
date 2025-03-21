import pandas as pd
import os

# Read the bug data
bug_data = pd.read_csv('data/processed/bug_data.csv')

# Create output directory if it doesn't exist
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

# 1. Config only bugs (has_config_changes = True, has_code_changes = False)
config_only = bug_data[
    (bug_data['has_config_changes'] == True) & 
    (bug_data['has_code_changes'] == False)
]
config_only.to_csv(f'{output_dir}/config_only_bugs.csv', index=False)

# 2. Code only bugs (has_config_changes = False, has_code_changes = True)
code_only = bug_data[
    (bug_data['has_config_changes'] == False) & 
    (bug_data['has_code_changes'] == True)
]
code_only.to_csv(f'{output_dir}/code_only_bugs.csv', index=False)

# 3. Mixed changes bugs (has_config_changes = True, has_code_changes = True)
mixed_changes = bug_data[
    (bug_data['has_config_changes'] == True) & 
    (bug_data['has_code_changes'] == True)
]
mixed_changes.to_csv(f'{output_dir}/mixed_changes_bugs.csv', index=False)

# 4. No changes bugs (has_config_changes = False, has_code_changes = False)
no_changes = bug_data[
    (bug_data['has_config_changes'] == False) & 
    (bug_data['has_code_changes'] == False)
]
no_changes.to_csv(f'{output_dir}/no_changes_bugs.csv', index=False)

# Print summary
print(f"Total bugs: {len(bug_data)}")
print(f"Config only bugs: {len(config_only)}")
print(f"Code only bugs: {len(code_only)}")
print(f"Mixed changes bugs: {len(mixed_changes)}")
print(f"No changes bugs: {len(no_changes)}")

# Verification that all bugs are accounted for
total_categorized = len(config_only) + len(code_only) + len(mixed_changes) + len(no_changes)
print(f"\nVerification:")
print(f"Sum of all categories: {total_categorized}")
print(f"All bugs accounted for: {total_categorized == len(bug_data)}") 