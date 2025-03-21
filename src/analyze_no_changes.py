import pandas as pd
import json
from collections import Counter

# Read the bugs with no changes
no_changes_bugs = pd.read_csv('data/processed/other_bugs.csv')

# Initialize counters
total_bugs = len(no_changes_bugs)
bugs_with_files_changed = 0
file_types_changed = Counter()
bugs_by_type = Counter()
bugs_by_severity = Counter()

# Analyze each bug
for _, bug in no_changes_bugs.iterrows():
    # Count bug types and severities
    bugs_by_type[bug['bug_type']] += 1
    bugs_by_severity[bug['bug_severity']] += 1
    
    # Check if there are actually changed files
    if isinstance(bug['changed_files'], str) and bug['changed_files'] != '[]':
        bugs_with_files_changed += 1
        try:
            changed_files = json.loads(bug['changed_files'])
            for file in changed_files:
                file_types_changed[file['file_type']] += 1
        except json.JSONDecodeError:
            print(f"Error parsing changed_files for bug {bug['issue_number']}")

print(f"\nAnalysis of bugs marked as having no code/config changes:")
print(f"Total bugs analyzed: {total_bugs}")
print(f"Bugs that actually have file changes: {bugs_with_files_changed}")
print(f"\nFile types changed:")
for file_type, count in file_types_changed.most_common():
    print(f"- {file_type}: {count}")
print(f"\nBug types:")
for bug_type, count in bugs_by_type.most_common():
    print(f"- {bug_type}: {count}")
print(f"\nBug severities:")
for severity, count in bugs_by_severity.most_common():
    print(f"- {severity}: {count}")

# Print some suspicious cases
print("\nSuspicious cases (bugs marked as no changes but have files changed):")
for _, bug in no_changes_bugs.iterrows():
    if isinstance(bug['changed_files'], str) and bug['changed_files'] != '[]':
        try:
            changed_files = json.loads(bug['changed_files'])
            if changed_files:
                print(f"\nIssue #{bug['issue_number']}: {bug['issue_title']}")
                print(f"Bug type: {bug['bug_type']}")
                print(f"Bug severity: {bug['bug_severity']}")
                print(f"Changed files:")
                for file in changed_files:
                    print(f"- {file['filename']} ({file['file_type']}): +{file['lines_added']}, -{file['lines_deleted']}")
        except json.JSONDecodeError:
            continue 