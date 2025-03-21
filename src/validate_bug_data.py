import pandas as pd
import requests
import json
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# GitHub API tokens (we'll rotate through these to avoid rate limits)
GITHUB_TOKENS = [
    os.getenv(f"GITHUB_TOKEN_{i}") for i in range(1, 5)
    if os.getenv(f"GITHUB_TOKEN_{i}")
]
current_token_index = 0

# File categorization (matching new_bug_collector.py)
CONFIG_FILES = {'.yml', '.yaml', '.json', '.xml', '.conf', '.config', '.properties', '.ini'}
APP_CODE_FILES = {'.java', '.py', '.js', '.ts', '.go', '.cs', '.cpp', '.rb', '.php'}

def get_next_token():
    global current_token_index
    token = GITHUB_TOKENS[current_token_index]
    current_token_index = (current_token_index + 1) % len(GITHUB_TOKENS)
    return token

def get_github_headers():
    return {
        "Authorization": f"token {get_next_token()}",
        "Accept": "application/vnd.github.v3+json"
    }

def categorize_file(filename):
    """Categorize file as config, application code, or other (matching new_bug_collector.py logic)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in CONFIG_FILES:
        return 'config'
    elif ext in APP_CODE_FILES:
        return 'code'
    else:
        return 'other'

def get_pr_files(repo_name, pr_number):
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}/files"
    response = requests.get(url, headers=get_github_headers())
    
    if response.status_code == 404:
        return None
    
    if response.status_code != 200:
        print(f"Error fetching PR files: {response.status_code}")
        return None
    
    return response.json()

def get_issue_events(repo_name, issue_number):
    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}/events"
    response = requests.get(url, headers=get_github_headers())
    
    if response.status_code != 200:
        print(f"Error fetching issue events: {response.status_code}")
        return None
    
    return response.json()

def analyze_changes(files):
    if not files:
        return False, False, False, {}
    
    has_config_changes = False
    has_code_changes = False
    has_any_changes = len(files) > 0
    
    file_types = {
        'config': 0,
        'code': 0,
        'other': 0
    }
    
    for file in files:
        filename = file['filename']
        file_type = categorize_file(filename)
        file_types[file_type] += 1
        
        if file_type == 'config':
            has_config_changes = True
        elif file_type == 'code':
            has_code_changes = True
    
    return has_config_changes, has_code_changes, has_any_changes, file_types

def update_bug_data():
    # Read the bug data
    bug_data = pd.read_csv('data/processed/bug_data.csv')
    total_bugs = len(bug_data)
    updates = 0
    errors = 0
    
    # Add new columns for tracking all types of changes
    bug_data['has_any_changes'] = False
    bug_data['other_changes'] = 0

    # Create backup of original data
    bug_data.to_csv('data/processed/bug_data_backup.csv', index=False)

    for index, bug in bug_data.iterrows():
        try:
            print(f"\nProcessing {index + 1}/{total_bugs}: {bug['repo_name']} #{bug['issue_number']}")
            
            # If we have a PR number, check PR files
            files = None
            if pd.notna(bug['pr_number']):
                files = get_pr_files(bug['repo_name'], int(bug['pr_number']))
            
            # If no PR or couldn't get PR files, check issue events
            if not files:
                events = get_issue_events(bug['repo_name'], bug['issue_number'])
                if events:
                    # Look for referenced PRs in events
                    for event in events:
                        if event.get('event') == 'referenced' and 'pull_request' in event:
                            pr_number = event['pull_request']['number']
                            files = get_pr_files(bug['repo_name'], pr_number)
                            if files:
                                break

            if files:
                has_config_changes, has_code_changes, has_any_changes, file_types = analyze_changes(files)
                
                # Update all change types
                bug_data.at[index, 'has_config_changes'] = has_config_changes
                bug_data.at[index, 'has_code_changes'] = has_code_changes
                bug_data.at[index, 'has_any_changes'] = has_any_changes
                bug_data.at[index, 'other_changes'] = file_types['other']
                
                if (has_config_changes != bug['has_config_changes'] or 
                    has_code_changes != bug['has_code_changes']):
                    print(f"Updating bug {bug['issue_number']}:")
                    print(f"Old: config={bug['has_config_changes']}, code={bug['has_code_changes']}")
                    print(f"New: config={has_config_changes}, code={has_code_changes}")
                    print(f"File changes: {file_types}")
                    updates += 1
            
            # Sleep to avoid hitting rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing bug {bug['issue_number']}: {str(e)}")
            errors += 1
            continue

    # Calculate statistics
    truly_no_changes = len(bug_data[~bug_data['has_any_changes']])
    only_other_changes = len(bug_data[
        (bug_data['other_changes'] > 0) & 
        ~bug_data['has_code_changes'] & 
        ~bug_data['has_config_changes']
    ])
    
    print(f"\nProcessing complete!")
    print(f"Total bugs processed: {total_bugs}")
    print(f"Updates made: {updates}")
    print(f"Errors encountered: {errors}")
    print(f"\nAnalysis:")
    print(f"Bugs with truly no changes: {truly_no_changes}")
    print(f"Bugs with only other changes: {only_other_changes}")
    print(f"Bugs with code changes: {len(bug_data[bug_data['has_code_changes']])}")
    print(f"Bugs with config changes: {len(bug_data[bug_data['has_config_changes']])}")

    # Save updated data
    bug_data.to_csv('data/processed/bug_data_updated.csv', index=False)
    print("\nUpdated data saved to bug_data_updated.csv")

if __name__ == "__main__":
    update_bug_data() 