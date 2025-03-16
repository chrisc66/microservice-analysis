"""
Module for collecting bug-related data from GitHub repositories.
"""

import os
import json
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import random

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from github import Github, Repository, Issue, PullRequest, Commit
from github.GithubException import RateLimitExceededException
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from config.config import (
    BUG_KEYWORDS, CONFIG_FILES, APP_CODE_FILES,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe lock for saving data and token rotation
save_lock = threading.Lock()
token_lock = threading.Lock()

class TokenManager:
    def __init__(self, tokens: List[str]):
        """Initialize with a list of GitHub tokens."""
        self.tokens = tokens
        self.current_index = 0
        self.clients = [Github(token) for token in tokens]
        self._lock = threading.Lock()
        
    def get_client(self) -> Github:
        """Get the next available GitHub client."""
        with self._lock:
            # Try each client until we find one with available rate limit
            for _ in range(len(self.clients)):
                client = self.clients[self.current_index]
                try:
                    rate_limit = client.get_rate_limit()
                    if rate_limit.core.remaining > 100:  # Ensure sufficient remaining calls
                        return client
                except:
                    pass  # Skip failed clients
                
                # Move to next client
                self.current_index = (self.current_index + 1) % len(self.clients)
                
            # If no client has sufficient rate limit, use current one and let it handle the rate limit
            return self.clients[self.current_index]

class BugDataCollector:
    def __init__(self, tokens: List[str], max_workers: int = 5):
        """Initialize the bug data collector with GitHub tokens."""
        self.token_manager = TokenManager(tokens)
        self.bug_data = []
        self.max_workers = max_workers
        self._data_lock = threading.Lock()

    def get_github_client(self) -> Github:
        """Get an available GitHub client."""
        return self.token_manager.get_client()

    @lru_cache(maxsize=1000)
    def is_bug_related(self, text: str) -> bool:
        """Cache bug-related text checks to avoid redundant processing."""
        text = text.lower()
        return any(keyword in text for keyword in BUG_KEYWORDS)

    def get_file_category(self, filename: str) -> str:
        """Categorize file as config, application code, or other."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in CONFIG_FILES:
            return 'config'
        elif ext in APP_CODE_FILES:
            return 'app_code'
        return 'other'

    def analyze_pr_files(self, pr: PullRequest) -> Dict[str, int]:
        """Analyze files changed in a pull request."""
        try:
            file_counts = {'config': 0, 'app_code': 0, 'other': 0}
            for file in pr.get_files():
                category = self.get_file_category(file.filename)
                file_counts[category] += 1
            return file_counts
        except Exception as e:
            logger.warning(f"Error analyzing PR files: {str(e)}")
            return {'config': 0, 'app_code': 0, 'other': 0}

    def get_bug_resolution_time(self, issue: Issue, pr: Optional[PullRequest] = None) -> Optional[float]:
        """Calculate bug resolution time in hours."""
        try:
            if not issue.closed_at:
                return None
            end_time = pr.merged_at if pr and pr.merged_at else issue.closed_at
            if not end_time:
                return None
            resolution_time = (end_time - issue.created_at).total_seconds() / 3600
            return resolution_time
        except Exception as e:
            logger.warning(f"Error calculating resolution time: {str(e)}")
            return None

    def find_associated_pr(self, repo: Repository, issue_number: int) -> Optional[PullRequest]:
        """Find PR associated with an issue using direct PR fetching."""
        try:
            # Get recent PRs in batches to avoid rate limits
            pulls = repo.get_pulls(state='closed', sort='updated', direction='desc')
            
            # Only check first 200 PRs to avoid excessive API calls
            for pr in pulls[:200]:
                try:
                    # Check PR body for issue reference
                    if pr.body and f"#{issue_number}" in pr.body:
                        return pr
                    # Also check PR title
                    if f"#{issue_number}" in pr.title:
                        return pr
                except Exception:
                    continue
            return None
        except Exception as e:
            logger.warning(f"Error finding PR for issue #{issue_number}: {str(e)}")
            return None

    def collect_bug_data(self, repo: Repository) -> List[Dict]:
        """Collect bug-related data from a repository."""
        bug_entries = []
        max_issues_per_repo = 500
        
        try:
            # Get issues in batches
            issues = repo.get_issues(state='closed', sort='updated', direction='desc')
            total_issues = min(issues.totalCount, max_issues_per_repo)
            
            logger.info(f"Processing up to {total_issues} issues from {repo.full_name}")
            
            with tqdm(total=total_issues, desc=f"Processing issues in {repo.full_name}", leave=False) as pbar:
                for i, issue in enumerate(issues):
                    if i >= max_issues_per_repo:
                        break
                        
                    try:
                        # Skip if not bug-related
                        if not (
                            any('bug' in label.name.lower() for label in issue.labels) or
                            self.is_bug_related(issue.title) or
                            (issue.body and self.is_bug_related(issue.body))
                        ):
                            pbar.update(1)
                            continue
                        
                        # Find associated PR using optimized search
                        associated_pr = self.find_associated_pr(repo, issue.number)
                        
                        if associated_pr:
                            file_changes = self.analyze_pr_files(associated_pr)
                            resolution_time = self.get_bug_resolution_time(issue, associated_pr)
                            
                            bug_entry = {
                                'repo_name': repo.full_name,
                                'issue_number': issue.number,
                                'issue_title': issue.title,
                                'issue_created_at': issue.created_at.isoformat(),
                                'issue_closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                                'pr_number': associated_pr.number,
                                'pr_merged_at': associated_pr.merged_at.isoformat() if associated_pr.merged_at else None,
                                'config_files_changed': file_changes['config'],
                                'app_code_files_changed': file_changes['app_code'],
                                'other_files_changed': file_changes['other'],
                                'resolution_time_hours': resolution_time,
                                'labels': [label.name for label in issue.labels]
                            }
                            
                            bug_entries.append(bug_entry)
                            
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.warning(f"Error processing issue in {repo.full_name}: {str(e)}")
                        pbar.update(1)
                        continue
                    
        except RateLimitExceededException:
            logger.error("GitHub API rate limit exceeded!")
            raise
        except Exception as e:
            logger.error(f"Error collecting bug data from {repo.full_name}: {str(e)}")
            
        return bug_entries

    def process_repository(self, repo_info: Dict) -> List[Dict]:
        """Process a single repository."""
        try:
            logger.info(f"Processing repository: {repo_info['full_name']}")
            
            # Get a GitHub client with available rate limit
            github = self.get_github_client()
            
            repo = github.get_repo(repo_info['full_name'])
            bug_entries = self.collect_bug_data(repo)
            
            # Thread-safe update of bug data
            with self._data_lock:
                self.bug_data.extend(bug_entries)
                self.save_results()
            
            return bug_entries
            
        except Exception as e:
            logger.error(f"Error processing repository {repo_info['full_name']}: {str(e)}")
            return []

    def process_repositories(self):
        """Process all candidate repositories using parallel processing."""
        repo_file = os.path.join(RAW_DATA_DIR, 'candidate_repos.json')
        if not os.path.exists(repo_file):
            logger.error("Candidate repositories file not found!")
            return
            
        with open(repo_file, 'r') as f:
            repos_data = json.load(f)
            
        logger.info(f"Found {len(repos_data)} repositories to process")
        
        # Process repositories in smaller batches to manage rate limits
        batch_size = min(self.max_workers * 2, 10)  # Process 10 repos at a time
        for i in range(0, len(repos_data), batch_size):
            batch = repos_data[i:i + batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_repo = {
                    executor.submit(self.process_repository, repo_info): repo_info
                    for repo_info in batch
                }
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_repo), 
                                 total=len(batch), 
                                 desc=f"Processing repositories (batch {i//batch_size + 1}/{(len(repos_data) + batch_size - 1)//batch_size})"):
                    repo_info = future_to_repo[future]
                    try:
                        bug_entries = future.result()
                        logger.info(f"Completed {repo_info['full_name']} - Found {len(bug_entries)} bug entries")
                    except Exception as e:
                        logger.error(f"Error processing {repo_info['full_name']}: {str(e)}")
            
            # Check rate limits after each batch
            rate_limit = self.token_manager.get_client().get_rate_limit()
            remaining = rate_limit.core.remaining
            if remaining < 100:  # Buffer of 100 requests
                reset_time = rate_limit.core.reset
                wait_time = int((reset_time - datetime.now(timezone.utc)).total_seconds()) + 1
                logger.warning(f"Rate limit low ({remaining} remaining). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
        
        self.save_results()

    def save_results(self):
        """Thread-safe save of collected bug data."""
        with save_lock:
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            output_file = os.path.join(PROCESSED_DATA_DIR, 'bug_data.csv')
            
            df = pd.DataFrame(self.bug_data)
            df.to_csv(output_file, index=False)
            
            logger.info(f"Saved {len(self.bug_data)} bug entries to {output_file}")

def main():
    """Main function to run the bug data collector."""
    load_dotenv()
    
    # Load multiple tokens from environment variables
    # Format: GITHUB_TOKEN_1=token1
    #         GITHUB_TOKEN_2=token2
    #         etc.
    tokens = []
    i = 1
    while True:
        token = os.getenv(f'GITHUB_TOKEN_{i}')
        if not token:
            # Try the default token name for the first token
            if i == 1:
                token = os.getenv('GITHUB_TOKEN')
                if token:
                    tokens.append(token)
            break
        tokens.append(token)
        i += 1
    
    if not tokens:
        logger.error("No GitHub tokens found! Please set GITHUB_TOKEN_1, GITHUB_TOKEN_2, etc. environment variables.")
        return
    
    logger.info(f"Found {len(tokens)} GitHub tokens")
    
    # Set logging level to INFO
    logging.getLogger().setLevel(logging.INFO)
    
    # Initialize collector with appropriate number of workers
    max_workers = min(os.cpu_count() or 4, 5)  # Use at most 5 workers
    collector = BugDataCollector(tokens, max_workers=max_workers)
    collector.process_repositories()

if __name__ == "__main__":
    main() 