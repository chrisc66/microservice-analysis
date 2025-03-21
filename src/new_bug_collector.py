"""
Script to collect bug-related data from microservice repositories.
Only collects bugs that have application code or configuration file changes.
"""

import os
import json
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import csv

import pandas as pd
from github import Github, Repository, Issue, PullRequest
from github.GithubException import RateLimitExceededException
from dotenv import load_dotenv
from tqdm import tqdm

# Constants
BUG_KEYWORDS = {
    # Strong bug indicators
    'bug', 'defect', 'regression', 'broken',
    
    # Error conditions
    'error', 'exception', 'crash', 'failure',
    'failed', 'fails', 'failing',
    
    # Common bug descriptions
    'incorrect', 'unexpected', 'wrong',
    'invalid', 'malfunction', 'corrupt',
    
    # Security issues
    'vulnerability', 'security issue', 'exploit',
    'unsafe', 'cve-'
}

# Labels that indicate bugs
BUG_LABELS = {
    'bug', 'type: bug', 'kind/bug',
    'type/bug', 'bug/defect', 'type: defect',
    'confirmed-bug', 'reproducible-bug',
    'security', 'security-issue'
}

# Labels that indicate non-bugs
NON_BUG_LABELS = {
    'enhancement', 'feature', 'feature-request',
    'documentation', 'question', 'help wanted',
    'wontfix', 'duplicate', 'invalid',
    'type: feature', 'kind/feature'
}

CONFIG_FILES = {'.yml', '.yaml', '.json', '.xml', '.conf', '.config', '.properties', '.ini'}
APP_CODE_FILES = {'.java', '.py', '.js', '.ts', '.go', '.cs', '.cpp', '.rb', '.php'}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenManager:
    """Manages GitHub API tokens with rotation and rate limit tracking."""
    
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.clients = {token: Github(token) for token in tokens}
        self.current_token_idx = 0
        self.token_locks = {token: threading.Lock() for token in tokens}
        self.token_quotas = {
            token: {
                'core': {'remaining': 5000, 'reset_time': None},
                'search': {'remaining': 30, 'reset_time': None}
            } 
            for token in tokens
        }
        self.last_check = {token: 0 for token in tokens}
        self.check_interval = 60  # Check rate limit every 60 seconds
        self.last_used = {token: 0 for token in tokens}  # Track last usage time
        
    def _update_quota(self, token: str) -> None:
        """Update quota information for a token."""
        try:
            client = self.clients[token]
            rate_limit = client.get_rate_limit()
            self.token_quotas[token] = {
                'core': {
                    'remaining': rate_limit.core.remaining,
                    'reset_time': rate_limit.core.reset
                },
                'search': {
                    'remaining': rate_limit.search.remaining,
                    'reset_time': rate_limit.search.reset
                }
            }
            self.last_check[token] = time.time()
            logger.debug(f"Token {token[:8]}... Core: {rate_limit.core.remaining}, Search: {rate_limit.search.remaining}")
        except Exception as e:
            logger.error(f"Error checking rate limit for token {token[:8]}...: {str(e)}")
            self.token_quotas[token]['core']['remaining'] = 0
            self.token_quotas[token]['search']['remaining'] = 0
            
    def _should_update_quota(self, token: str) -> bool:
        """Determine if we should update quota information."""
        return time.time() - self.last_check[token] > self.check_interval
        
    def _get_next_token(self, require_search: bool = False) -> Optional[str]:
        """Get next available token using round-robin with quota check."""
        api_type = 'search' if require_search else 'core'
        min_quota = 5 if require_search else 100
        
        # Try each token in order
        for _ in range(len(self.tokens)):
            token = self.tokens[self.current_token_idx]
            self.current_token_idx = (self.current_token_idx + 1) % len(self.tokens)
            
            # Update quota if needed
            if self._should_update_quota(token):
                self._update_quota(token)
            
            quota = self.token_quotas[token]
            
            # Check if token is reset
            if quota[api_type]['reset_time'] and quota[api_type]['reset_time'] < datetime.now(timezone.utc):
                self._update_quota(token)
                quota = self.token_quotas[token]
            
            # Use token if it has sufficient quota
            if quota[api_type]['remaining'] > min_quota:
                self.last_used[token] = time.time()
                return token
                
        return None
        
    def get_client(self, require_search: bool = False) -> Github:
        """Get a GitHub client with available rate limit."""
        while True:
            # Try to get next available token
            token = self._get_next_token(require_search)
            
            if token:
                return self.clients[token]
            
            # If no token available, find the one that will reset soonest
            earliest_reset = None
            earliest_token = None
            api_type = 'search' if require_search else 'core'
            
            for token in self.tokens:
                quota = self.token_quotas[token][api_type]
                if quota['reset_time']:
                    if not earliest_reset or quota['reset_time'] < earliest_reset:
                        earliest_reset = quota['reset_time']
                        earliest_token = token
            
            if earliest_reset:
                wait_time = int((earliest_reset - datetime.now(timezone.utc)).total_seconds()) + 1
                if wait_time > 0:
                    logger.warning(f"All tokens exhausted for {api_type} API. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    self._update_quota(earliest_token)
            else:
                # If we can't determine reset times, wait a minute
                logger.warning(f"Cannot determine token reset times for {api_type} API. Waiting 60 seconds...")
                time.sleep(60)
            
            # Update all quotas after waiting
            for token in self.tokens:
                self._update_quota(token)

class BugDataCollector:
    def __init__(self, tokens: List[str], max_workers: int = None):
        """Initialize the bug data collector."""
        self.token_manager = TokenManager(tokens)
        self.bug_data = []
        self.max_workers = min(max_workers or len(tokens) * 2, 8)  # 2 workers per token, max 8
        self._data_lock = threading.Lock()
        self._repo_semaphore = threading.Semaphore(len(tokens))  # Limit concurrent repos to number of tokens
        self._save_counter = 0
        self._save_interval = 10
        
    @lru_cache(maxsize=1000)
    def is_bug_related(self, text: str) -> bool:
        """Check if text contains bug-related keywords."""
        if not text:
            return False
        text = text.lower()
        
        # Count how many bug keywords are found
        bug_keyword_count = sum(1 for keyword in BUG_KEYWORDS if keyword in text)
        
        # More strict checking for common false positives
        if bug_keyword_count > 0:
            # Ignore if it's a feature request or question
            if any(phrase in text for phrase in [
                'feature request', 'enhancement request',
                'please add', 'would be nice', 'would be great',
                'how to', 'how do i', 'is it possible'
            ]):
                return False
                
        # Require stronger evidence for certain keywords
        if bug_keyword_count == 1 and any(word in text for word in ['issue', 'problem']):
            return False  # Need more evidence for weak keywords
            
        return bug_keyword_count > 0
        
    def has_bug_labels(self, labels) -> bool:
        """Check if the issue has bug-related labels."""
        label_names = {label.name.lower() for label in labels}
        
        # If it has any non-bug labels, it's probably not a bug
        if any(label in label_names for label in NON_BUG_LABELS):
            return False
            
        # Check for bug labels
        return any(label in label_names for label in BUG_LABELS)
        
    def is_likely_bug(self, issue: Issue) -> bool:
        """Determine if an issue is likely to be a bug report."""
        # First check labels
        if self.has_bug_labels(issue.labels):
            return True
            
        # Check title and body for bug keywords
        title_is_bug = self.is_bug_related(issue.title)
        body_is_bug = issue.body and self.is_bug_related(issue.body)
        
        # Need both title and body to indicate bug if no bug labels
        if title_is_bug and body_is_bug:
            return True
            
        # If title strongly indicates bug, that's enough
        if title_is_bug and any(strong_kw in issue.title.lower() 
                               for strong_kw in ['bug', 'defect', 'regression', 'broken']):
            return True
            
        return False
        
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
        """Find PR associated with an issue."""
        try:
            # First check recent PRs (uses core API quota)
            pulls = repo.get_pulls(state='closed', sort='updated', direction='desc')
            for pr in pulls[:50]:  # Reduced from 100 to 50
                if pr.body and f"#{issue_number}" in pr.body:
                    return pr
                if f"#{issue_number}" in pr.title:
                    return pr
                    
            # Try searching through all PRs (uses search API quota)
            search_query = f"repo:{repo.full_name} type:pr is:closed {issue_number}"
            github = self.token_manager.get_client(require_search=True)  # Specify search API
            results = github.search_issues(query=search_query)
            
            if results.totalCount > 0:
                pr_number = results[0].number
                return repo.get_pull(pr_number)
                
            return None
            
        except Exception as e:
            logger.warning(f"Error finding PR for issue #{issue_number}: {str(e)}")
            return None
            
    def process_single_repository(self, repo_data: dict) -> List[dict]:
        """Process a single repository and return its bug data."""
        repo_name = repo_data['full_name']
        batch_bugs = []
        bugs_found = 0
        issues_processed = 0
        
        # Create output directory if it doesn't exist
        output_dir = Path(__file__).parent.parent / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'bug_data.csv'
        
        try:
            with self._repo_semaphore:  # Ensure we don't exceed token count
                logger.info(f"Processing repository: {repo_name}")
                
                # Get GitHub client
                github = self.token_manager.get_client()
                repo = github.get_repo(repo_name)
                
                # Get closed issues
                issues = repo.get_issues(state='closed', sort='updated', direction='desc')
                total_issues = min(issues.totalCount, 1000)  # Process up to 1000 closed issues
                logger.info(f"Found {total_issues} closed issues in {repo_name}")
                
                with tqdm(total=total_issues, desc=f"Processing {repo_name}", leave=True) as pbar:
                    for i, issue in enumerate(issues):
                        if i >= 1000:  # Stop after 1000 issues
                            break
                            
                        try:
                            issues_processed += 1
                            
                            # Check if bug-related
                            is_bug = self.is_likely_bug(issue)
                            
                            if not is_bug:
                                pbar.update(1)
                                continue
                            
                            # Try to find associated PR and check file changes
                            has_relevant_changes = False
                            associated_pr = None
                            file_changes = {'config': 0, 'app_code': 0, 'other': 0}
                            changed_files = []
                            
                            # Try to find associated PR
                            max_retries = 3
                            for _ in range(max_retries):
                                try:
                                    associated_pr = self.find_associated_pr(repo, issue.number)
                                    if associated_pr:
                                        # Analyze files to check for app or config changes
                                        for file in associated_pr.get_files():
                                            file_type = self.get_file_category(file.filename)
                                            file_changes[file_type] += 1
                                            changed_files.append({
                                                'filename': file.filename,
                                                'lines_added': file.additions,
                                                'lines_deleted': file.deletions,
                                                'file_type': file_type
                                            })
                                            
                                            # Check if we have app or config changes
                                            if file_type in ['config', 'app_code']:
                                                has_relevant_changes = True
                                        
                                        break
                                except RateLimitExceededException:
                                    logger.warning(f"Rate limit hit, rotating token for {repo_name}")
                                    github = self.token_manager.get_client()
                                    repo = github.get_repo(repo_name)
                                    time.sleep(1)
                            
                            # Skip if no app or config changes
                            if not has_relevant_changes:
                                pbar.update(1)
                                continue
                            
                            bugs_found += 1
                            logger.info(f"Found bug with app/config changes in {repo_name}: Issue #{issue.number} - {issue.title}")
                            
                            # Create bug entry
                            bug_entry = {
                                'repo_name': repo_name,
                                'issue_number': issue.number,
                                'issue_title': issue.title or "NA",
                                'issue_body': (issue.body or "NA").replace('\n', ' ').replace('\r', ''),
                                'issue_created_at': issue.created_at.isoformat() if issue.created_at else "NA",
                                'issue_closed_at': issue.closed_at.isoformat() if issue.closed_at else "NA",
                                'issue_comments_count': issue.comments or 0,
                                'issue_url': issue.html_url or "NA",
                                'pr_number': associated_pr.number if associated_pr else "NA",
                                'pr_merged_at': associated_pr.merged_at.isoformat() if associated_pr and associated_pr.merged_at else "NA",
                                'pr_url': associated_pr.html_url if associated_pr else "NA",
                                'config_files_changed': file_changes['config'],
                                'app_code_files_changed': file_changes['app_code'],
                                'other_files_changed': file_changes['other'],
                                'total_files_changed': len(changed_files),
                                'lines_added': sum(f['lines_added'] for f in changed_files),
                                'lines_deleted': sum(f['lines_deleted'] for f in changed_files),
                                'config_files_lines_changed': sum((f['lines_added'] + f['lines_deleted']) 
                                                               for f in changed_files if f['file_type'] == 'config'),
                                'app_code_files_lines_changed': sum((f['lines_added'] + f['lines_deleted']) 
                                                                 for f in changed_files if f['file_type'] == 'app_code'),
                                'resolution_time_hours': self.get_bug_resolution_time(issue, associated_pr) or "NA",
                                'labels': [label.name for label in issue.labels] if issue.labels else ["NA"],
                                'has_config_changes': file_changes['config'] > 0,
                                'has_code_changes': file_changes['app_code'] > 0,
                                'bug_severity': self.get_bug_severity(issue),
                                'bug_type': self.get_bug_type(issue),
                                'changed_files': changed_files,
                                'services_affected': [self.identify_service(f['filename']) for f in changed_files if self.identify_service(f['filename'])],
                                'is_cross_service_bug': len(set(self.identify_service(f['filename']) 
                                                             for f in changed_files 
                                                             if self.identify_service(f['filename']))) > 1
                            }

                            # Save bug immediately
                            with self._data_lock:
                                # Convert complex types to strings for CSV
                                csv_entry = bug_entry.copy()
                                csv_entry['labels'] = ';'.join(str(label) for label in csv_entry['labels'])
                                csv_entry['changed_files'] = json.dumps(csv_entry['changed_files'])
                                csv_entry['services_affected'] = ';'.join(str(svc) for svc in csv_entry['services_affected']) if csv_entry['services_affected'] else "NA"
                                csv_entry['has_config_changes'] = str(csv_entry['has_config_changes'])
                                csv_entry['has_code_changes'] = str(csv_entry['has_code_changes'])
                                csv_entry['is_cross_service_bug'] = str(csv_entry['is_cross_service_bug'])

                                try:
                                    df = pd.DataFrame([csv_entry])
                                    if os.path.exists(output_file):
                                        df.to_csv(output_file, mode='a', header=False, index=False, escapechar='\\', quoting=csv.QUOTE_ALL)
                                    else:
                                        df.to_csv(output_file, index=False, escapechar='\\', quoting=csv.QUOTE_ALL)
                                    logger.info(f"Successfully saved bug #{issue.number} from {repo_name} to CSV")
                                except Exception as e:
                                    logger.error(f"Error saving to CSV: {str(e)}")
                            
                            batch_bugs.append(bug_entry)
                            pbar.set_postfix({'bugs': len(batch_bugs), 'saved': True})
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.warning(f"Error processing issue in {repo_name}: {str(e)}")
                            pbar.update(1)
                            continue
                
                logger.info(f"Repository {repo_name} completed: {bugs_found} bugs with app/config changes found out of {issues_processed} issues processed")
                
        except Exception as e:
            logger.error(f"Error processing repository {repo_name}: {str(e)}")
        
        return batch_bugs

    def process_repositories(self):
        """Process repositories in parallel using thread pool."""
        # Read microservice repositories
        data_dir = Path(__file__).parent.parent / 'data'
        repo_file = data_dir / 'raw' / 'microservice_repos.json'
        
        with open(repo_file, 'r') as f:
            repos_data = json.load(f)
            
        logger.info(f"Found {len(repos_data)} repositories to process")
        total_bugs_found = 0
        
        # Process repositories in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_repo = {
                executor.submit(self.process_single_repository, repo_data): repo_data
                for repo_data in repos_data
            }
            
            # Process completed repositories and save results
            for future in as_completed(future_to_repo):
                repo_data = future_to_repo[future]
                try:
                    batch_bugs = future.result()
                    if batch_bugs:
                        total_bugs_found += len(batch_bugs)
                        logger.info(f"Completed processing {repo_data['full_name']} with {len(batch_bugs)} bugs (Total: {total_bugs_found})")
                except Exception as e:
                    logger.error(f"Error processing {repo_data['full_name']}: {str(e)}")
                    
        logger.info(f"Completed processing all repositories. Total bugs found: {total_bugs_found}")

    def get_bug_severity(self, issue: Issue) -> str:
        """Determine bug severity based on labels and content."""
        try:
            # Check labels first
            label_names = {label.name.lower() for label in issue.labels}
            
            # Common severity labels
            if any(label in label_names for label in {'critical', 'blocker', 'p0'}):
                return 'critical'
            if any(label in label_names for label in {'major', 'high', 'p1'}):
                return 'major'
            if any(label in label_names for label in {'minor', 'low', 'p2'}):
                return 'minor'
            
            # Check content for severity indicators
            text = f"{issue.title} {issue.body}".lower()
            if any(word in text for word in {'crash', 'critical', 'severe', 'urgent', 'blocker', 'broken'}):
                return 'critical'
            if any(word in text for word in {'major', 'significant', 'important'}):
                return 'major'
            
            return 'normal'
        except Exception:
            return "NA"

    def get_bug_type(self, issue: Issue) -> str:
        """Categorize the type of bug based on content."""
        try:
            text = f"{issue.title} {issue.body}".lower()
            
            if any(word in text for word in {'config', 'configuration', 'yaml', 'json', 'settings'}):
                return 'configuration'
            if any(word in text for word in {'network', 'connection', 'timeout', 'dns'}):
                return 'networking'
            if any(word in text for word in {'database', 'sql', 'data', 'query'}):
                return 'database'
            if any(word in text for word in {'security', 'auth', 'permission', 'access'}):
                return 'security'
            if any(word in text for word in {'ui', 'interface', 'display', 'screen'}):
                return 'ui'
            if any(word in text for word in {'performance', 'slow', 'memory', 'cpu'}):
                return 'performance'
            
            return 'functional'
        except Exception:
            return "NA"

    def identify_service(self, filename: str) -> Optional[str]:
        """Try to identify which microservice a file belongs to based on its path."""
        # Common patterns in microservice architectures
        path_parts = filename.lower().split('/')
        
        # Look for service indicators in path
        service_indicators = ['service', 'svc', 'api', 'server', 'app']
        for i, part in enumerate(path_parts):
            # Check if this part contains a service indicator
            if any(indicator in part for indicator in service_indicators):
                # Return the service name (previous part if it exists, or this part)
                return path_parts[i-1] if i > 0 else part
            
            # Check for common microservice names
            if any(name in part for name in ['auth', 'user', 'order', 'payment', 'cart', 'catalog', 'inventory']):
                return part
                
        return None

def main():
    """Main function to run the bug data collector."""
    # Load environment variables
    load_dotenv()
    
    # Load GitHub tokens
    github_tokens = []
    for i in range(1, 5):  # Try GITHUB_TOKEN_1 through GITHUB_TOKEN_4
        token = os.getenv(f'GITHUB_TOKEN_{i}')
        if token:
            github_tokens.append(token)
            logger.info(f"Loaded GITHUB_TOKEN_{i}")
    
    if not github_tokens:
        logger.error("No GitHub tokens found! Please set GITHUB_TOKEN_1 through GITHUB_TOKEN_4")
        return
        
    logger.info(f"Starting bug data collection with {len(github_tokens)} tokens")
    
    try:
        collector = BugDataCollector(github_tokens)
        collector.process_repositories()
        logger.info("Bug data collection completed")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 