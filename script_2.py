#!/usr/bin/env python3
import os
import requests
import datetime
import time
import pandas as pd
import numpy as np
from github import Github
from scipy.stats import spearmanr

# ------------------------------------------------------------------------------
# Setup: GitHub API base URL and authentication headers using a personal token.
# Ensure you have set the environment variable GITHUB_TOKEN before running.
# ------------------------------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    raise Exception("Please set the GITHUB_TOKEN environment variable.")

BASE_URL = "https://api.github.com"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# ------------------------------------------------------------------------------
# Function: get_repo_info
# Description: Fetch repository-level information such as star count,
#              size (in KB, used as a proxy for LOC), and creation time.
# ------------------------------------------------------------------------------
def get_repo_info(owner, repo):
    url = f"{BASE_URL}/repos/{owner}/{repo}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error fetching repo info for {owner}/{repo}: {response.status_code}")
        return None
    data = response.json()
    repo_info = {
        "owner": owner,
        "repo": repo,
        "stars": data.get("stargazers_count", 0),
        "size_kb": data.get("size", 0),  # Repository size in KB as a proxy for LOC.
        "created_at": data.get("created_at")
    }
    return repo_info

# ------------------------------------------------------------------------------
# Function: get_contributor_count
# Description: Retrieve the total number of contributors for the repository.
# ------------------------------------------------------------------------------
def get_contributor_count(owner, repo):
    url = f"{BASE_URL}/repos/{owner}/{repo}/contributors?per_page=100"
    count = 0
    page = 1
    while True:
        r = requests.get(url + f"&page={page}", headers=HEADERS)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        count += len(data)
        page += 1
        time.sleep(1)  # Pause to respect rate limits.
    return count

# ------------------------------------------------------------------------------
# Function: get_issues
# Description: Fetch closed issues from the repository.
#              Note: This filters out pull requests (which are also issues).
# ------------------------------------------------------------------------------
def get_issues(owner, repo, state="closed", max_pages=10):
    issues = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/repos/{owner}/{repo}/issues?state={state}&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        # Filter out pull requests (issues with a 'pull_request' key).
        issues.extend([issue for issue in data if "pull_request" not in issue])
        time.sleep(1)
    return issues

# ------------------------------------------------------------------------------
# Function: compute_issue_resolution_times
# Description: For each issue, compute the resolution time (in hours)
#              as the difference between closed_at and created_at.
# ------------------------------------------------------------------------------
def compute_issue_resolution_times(issues):
    resolution_times = []
    for issue in issues:
        try:
            created_at = datetime.datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            closed_at = datetime.datetime.strptime(issue["closed_at"], "%Y-%m-%dT%H:%M:%SZ")
            resolution_time = (closed_at - created_at).total_seconds() / 3600  # Hours
            resolution_times.append(resolution_time)
        except Exception as e:
            print(f"Error computing resolution time for issue {issue.get('number')}: {e}")
    return resolution_times

# ------------------------------------------------------------------------------
# Function: get_first_response_time
# Description: Fetch the first comment for an issue to compute the response time.
# ------------------------------------------------------------------------------
def get_first_response_time(owner, repo, issue_number):
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=1&page=1"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return None
    comments = response.json()
    if not comments:
        return None
    try:
        first_comment_time = datetime.datetime.strptime(comments[0]["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        # Fetch the issue to get its creation time.
        issue_url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}"
        issue_resp = requests.get(issue_url, headers=HEADERS)
        if issue_resp.status_code != 200:
            return None
        issue_data = issue_resp.json()
        created_at = datetime.datetime.strptime(issue_data["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        response_time = (first_comment_time - created_at).total_seconds() / 3600  # Hours
        return response_time
    except Exception as e:
        print(f"Error computing response time for issue {issue_number}: {e}")
        return None

# ------------------------------------------------------------------------------
# Function: compute_average_response_times
# Description: Compute average response time across issues that have a comment.
# ------------------------------------------------------------------------------
def compute_average_response_times(owner, repo, issues):
    response_times = []
    for issue in issues:
        issue_number = issue["number"]
        resp_time = get_first_response_time(owner, repo, issue_number)
        if resp_time is not None and resp_time >= 0:
            response_times.append(resp_time)
        time.sleep(0.5)  # Brief pause to avoid rate limits.
    return response_times

# ------------------------------------------------------------------------------
# Function: get_pull_requests
# Description: Retrieve pull requests (closed) from the repository.
# ------------------------------------------------------------------------------
def get_pull_requests(owner, repo, state="closed", max_pages=10):
    prs = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/repos/{owner}/{repo}/pulls?state={state}&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        prs.extend(data)
        time.sleep(1)
    return prs

# ------------------------------------------------------------------------------
# Function: get_pr_files
# Description: Retrieve the list of files modified in a given pull request.
# ------------------------------------------------------------------------------
def get_pr_files(owner, repo, pr_number):
    files = []
    page = 1
    while True:
        url = f"{BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        files.extend(data)
        page += 1
        time.sleep(1)
    return files

# ------------------------------------------------------------------------------
# Function: analyze_pr_file_types
# Description: Analyze the modified file types in pull requests and compute
#              the percentage distribution of file extensions.
# ------------------------------------------------------------------------------
def analyze_pr_file_types(owner, repo):
    prs = get_pull_requests(owner, repo)
    ext_counts = {}
    total_files = 0
    for pr in prs:
        pr_number = pr["number"]
        # (Optionally, filter PRs that are directly associated with issue resolutions.)
        files = get_pr_files(owner, repo, pr_number)
        for file in files:
            filename = file.get("filename", "")
            # Extract the file extension.
            if '.' in filename:
                ext = filename.split('.')[-1].lower()
            else:
                ext = "no_extension"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_files += 1
    # Compute the percentage distribution.
    ext_percentages = {ext: (count / total_files) * 100 for ext, count in ext_counts.items()} if total_files > 0 else {}
    return ext_percentages

# ------------------------------------------------------------------------------
# Main analysis function to process multiple repositories.
# This example demonstrates both research questions:
# RQ1: Correlation between repository quality and issue resolution dynamics.
# RQ2: Distribution of file types modified in PRs related to issue resolution.
# ------------------------------------------------------------------------------
def main():
    # Define a list of repositories to analyze.
    # For a real study, this list should be curated based on microservice characteristics.
    repositories = [
        # Add more repositories as needed.
        # demo projects
        {"owner": "GoogleCloudPlatform", "repo": "microservices-demo"},
        {"owner": "microservices-patterns", "repo": "ftgo-application"},
        {"owner": "instana", "repo": "robot-shop"},
        {"owner": "paulc4", "repo": "microservices-demo"},
        {"owner": "microservices-demo", "repo": "carts"},
        {"owner": "begmaroman", "repo": "go-micro-boilerplate"},
        {"owner": "mehdihadeli", "repo": "food-delivery-microservices"},
        {"owner": "mehdihadeli", "repo": "game-leaderboard-microservices"},
        {"owner": "acmeair", "repo": "acmeair"},
        {"owner": "istio", "repo": "istio"},

        # real world projects
        # https://github.com/davidetaibi/Microservices_Project_List
        # https://github.com/topics/spring-boot-microservices
        {"owner": "dotnet-architecture", "repo": "eShopOnContainers"},

    ]
    
    # Initialize a list to collect repository-level metrics for RQ1.
    repo_metrics = []
    
    for repo_item in repositories:
        owner = repo_item["owner"]
        repo = repo_item["repo"]
        print(f"\nProcessing repository: {owner}/{repo}")
        
        # Fetch repository information.
        info = get_repo_info(owner, repo)
        if info is None:
            continue
        
        # Get contributor count.
        contributor_count = get_contributor_count(owner, repo)
        
        # Fetch closed issues.
        issues = get_issues(owner, repo, state="closed", max_pages=10)
        if not issues:
            print(f"No issues found for {owner}/{repo}")
            continue
        
        # Compute average issue resolution time.
        resolution_times = compute_issue_resolution_times(issues)
        avg_resolution_time = np.mean(resolution_times) if resolution_times else None
        
        # Compute average response time.
        response_times = compute_average_response_times(owner, repo, issues)
        avg_response_time = np.mean(response_times) if response_times else None
        
        # Aggregate repository metrics.
        metrics = {
            "owner": owner,
            "repo": repo,
            "stars": info["stars"],
            "size_kb": info["size_kb"],
            "contributors": contributor_count,
            "avg_resolution_time_hours": avg_resolution_time,
            "avg_response_time_hours": avg_response_time
        }
        repo_metrics.append(metrics)
    
    # Convert the collected metrics into a DataFrame for RQ1 analysis.
    df_metrics = pd.DataFrame(repo_metrics)
    print("\nRepository Metrics Data:")
    print(df_metrics)
    
    # Perform Spearman correlation analysis between repository quality metrics and issue times.
    if not df_metrics.empty:
        for metric in ["stars", "size_kb", "contributors"]:
            if df_metrics[metric].isnull().all() or df_metrics["avg_resolution_time_hours"].isnull().all():
                continue
            corr, p_value = spearmanr(df_metrics[metric], df_metrics["avg_resolution_time_hours"], nan_policy='omit')
            print(f"\nSpearman correlation between {metric} and avg resolution time: {corr:.3f} (p-value: {p_value:.3f})")
            
            if df_metrics["avg_response_time_hours"].notnull().any():
                corr_resp, p_value_resp = spearmanr(df_metrics[metric], df_metrics["avg_response_time_hours"], nan_policy='omit')
                print(f"Spearman correlation between {metric} and avg response time: {corr_resp:.3f} (p-value: {p_value_resp:.3f})")
    
    # RQ2: Analyze file type modifications in pull requests.
    for repo_item in repositories:
        owner = repo_item["owner"]
        repo = repo_item["repo"]
        print(f"\nAnalyzing PR file types for repository: {owner}/{repo}")
        file_type_distribution = analyze_pr_file_types(owner, repo)
        print("File Type Distribution (in %):")
        for ext, perc in file_type_distribution.items():
            print(f"  {ext}: {perc:.2f}%")
    
if __name__ == "__main__":
    main()
