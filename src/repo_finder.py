"""
Repository finder module to identify suitable microservice repositories for analysis.
"""

import os
from datetime import datetime, timedelta, timezone
import json
from typing import List, Dict
import logging
import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from github import Github, Repository
from github.GithubException import RateLimitExceededException, GithubException
from dotenv import load_dotenv
from tqdm import tqdm

from config.config import (
    MIN_STARS, MIN_ISSUES, MIN_PRS, ACTIVITY_MONTHS,
    MICROSERVICE_INDICATORS, RAW_DATA_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RepositoryFinder:
    def __init__(self, token: str = None):
        """Initialize the repository finder with GitHub token."""
        self.github = Github(token)
        self.existing_repos = set()  # Set to track existing repository full names
        self.new_repos = []  # List to store newly found repositories
        self.load_existing_repos()
        
        # Check rate limit at start
        rate_limit = self.github.get_rate_limit()
        logger.info(f"GitHub API Rate Limit - Remaining: {rate_limit.core.remaining}/{rate_limit.core.limit}")

    def load_existing_repos(self):
        """Load existing repositories from JSON file if it exists."""
        self.main_file = os.path.join(RAW_DATA_DIR, 'candidate_repos.json')
        if os.path.exists(self.main_file):
            try:
                with open(self.main_file, 'r') as f:
                    existing_data = json.load(f)
                    self.existing_repos = {repo['full_name'] for repo in existing_data}
                logger.info(f"Loaded {len(self.existing_repos)} existing repositories")
            except Exception as e:
                logger.error(f"Error loading existing repositories: {str(e)}")
                self.existing_repos = set()

    def check_rate_limit(self):
        """Check GitHub API rate limit and wait if necessary."""
        rate_limit = self.github.get_rate_limit()
        if rate_limit.core.remaining < 10:  # Buffer of 10 requests
            reset_time = rate_limit.core.reset
            wait_time = int((reset_time - datetime.now(timezone.utc)).total_seconds()) + 1
            if wait_time > 0:
                logger.warning(f"Rate limit low. Waiting {wait_time} seconds for reset...")
                time.sleep(wait_time)
            return False
        return True

    def is_microservice_repo(self, repo: Repository) -> bool:
        """
        Check if a repository is likely a microservice-based project.
        
        Args:
            repo: GitHub repository object
        
        Returns:
            bool: True if the repository shows signs of microservice architecture
        """
        try:
            logger.debug(f"Checking microservice indicators for {repo.full_name}")
            
            # Check repository description and topics
            description = (repo.description or "").lower()
            topics = [topic.lower() for topic in repo.get_topics()]
            
            # Keywords that indicate microservice architecture
            microservice_keywords = {
                'microservice', 'microservices', 'micro-service', 'micro-services',
                'service-mesh', 'kubernetes', 'k8s', 'docker', 'container',
                'istio', 'service-discovery', 'api-gateway'
            }
            
            # Check description and topics for microservice keywords
            if any(keyword in description for keyword in microservice_keywords):
                logger.info(f"Repository {repo.full_name} identified as microservice from description")
                return True
                
            if any(keyword in topic for keyword in microservice_keywords for topic in topics):
                logger.info(f"Repository {repo.full_name} identified as microservice from topics")
                return True
            
            # Check files in root directory
            try:
                root_contents = repo.get_contents("")
                root_files = [content.name.lower() for content in root_contents if content.type == "file"]
                
                # Check for common microservice configuration files
                microservice_files = {
                    'docker-compose.yml', 'docker-compose.yaml', 'dockerfile',
                    'kubernetes', 'k8s.yaml', 'k8s.yml', 'helm.yaml',
                    'service.yaml', 'deployment.yaml', 'ingress.yaml',
                    'istio.yaml', 'consul.yaml', 'eureka.yaml'
                }
                
                if any(mfile in root_files for mfile in microservice_files):
                    logger.info(f"Repository {repo.full_name} identified as microservice from root files")
                    return True
                
                # Check for kubernetes/helm directories
                directories = [content.name.lower() for content in root_contents if content.type == "dir"]
                microservice_dirs = {'k8s', 'kubernetes', 'helm', 'deploy', 'deployment', 'services', 'microservices'}
                
                if any(mdir in directories for mdir in microservice_dirs):
                    logger.info(f"Repository {repo.full_name} identified as microservice from directories")
                    return True
                    
                # Check common deployment directories for kubernetes files
                for deploy_dir in ['deploy', 'deployment', 'k8s', 'kubernetes']:
                    if deploy_dir in directories:
                        try:
                            deploy_contents = repo.get_contents(deploy_dir)
                            deploy_files = [content.name.lower() for content in deploy_contents if content.type == "file"]
                            if any(file.endswith(('.yaml', '.yml')) for file in deploy_files):
                                logger.info(f"Repository {repo.full_name} identified as microservice from deployment files")
                                return True
                        except:
                            continue
                
            except GithubException as e:
                if e.status == 404:
                    logger.warning(f"Cannot access root directory of {repo.full_name}")
                else:
                    logger.warning(f"Error accessing root directory of {repo.full_name}: {str(e)}")
                return False
                
            return False
            
        except GithubException as e:
            if e.status == 404:
                logger.warning(f"Repository {repo.full_name} not found or no access")
            else:
                logger.warning(f"Error checking microservice indicators for {repo.full_name}: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Error checking microservice indicators for {repo.full_name}: {str(e)}")
            return False

    def is_actively_maintained(self, repo: Repository) -> bool:
        """
        Check if a repository is actively maintained.
        
        Args:
            repo: GitHub repository object
        
        Returns:
            bool: True if the repository has had commits in the last ACTIVITY_MONTHS months
        """
        try:
            # Check last commit date
            last_commit = repo.get_commits()[0]
            last_commit_date = last_commit.commit.author.date
            
            # Create timezone-aware datetime for comparison
            now = datetime.now(timezone.utc)
            cutoff_date = now - timedelta(days=30 * ACTIVITY_MONTHS)
            
            return last_commit_date >= cutoff_date
        except Exception as e:
            logger.warning(f"Error checking maintenance status for {repo.full_name}: {str(e)}")
            return False

    def meets_criteria(self, repo: Repository) -> bool:
        """
        Check if a repository meets all selection criteria.
        
        Args:
            repo: GitHub repository object
        
        Returns:
            bool: True if the repository meets all criteria
        """
        try:
            if (
                repo.stargazers_count >= MIN_STARS and
                repo.get_issues().totalCount >= MIN_ISSUES and
                repo.get_pulls().totalCount >= MIN_PRS and
                self.is_actively_maintained(repo) and
                self.is_microservice_repo(repo)
            ):
                return True
        except Exception as e:
            logger.warning(f"Error checking criteria for {repo.full_name}: {str(e)}")
        return False

    def save_new_repos(self):
        """Save newly found repositories to a temporary file."""
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        temp_file = os.path.join(RAW_DATA_DIR, 'new_repos_temp.json')
        
        with open(temp_file, 'w') as f:
            json.dump(self.new_repos, f, indent=2)
        
        logger.info(f"Saved {len(self.new_repos)} new repositories to temporary file")

    def find_repositories(self, max_repos: int = 50) -> List[Dict]:
        """
        Find repositories that match our criteria.
        
        Args:
            max_repos: Maximum number of NEW repositories to find
        
        Returns:
            List[Dict]: List of repository information dictionaries
        """
        search_query = f"stars:>={MIN_STARS} language:java,python,javascript,typescript,go"
        new_repos_found = 0
        repos_checked = 0
        
        try:
            logger.info(f"Starting repository search with query: {search_query}")
            repos = self.github.search_repositories(query=search_query)
            total_count = repos.totalCount
            logger.info(f"Found {total_count} total repositories matching initial search criteria")
            
            with tqdm(total=max_repos, desc="Finding new repositories") as pbar:
                for repo in repos:
                    repos_checked += 1
                    if repos_checked % 10 == 0:
                        logger.info(f"Checked {repos_checked} repositories, found {new_repos_found} matching all criteria")
                        self.check_rate_limit()
                    
                    # Skip if we already have this repository
                    if repo.full_name in self.existing_repos:
                        continue
                        
                    if new_repos_found >= max_repos:
                        break
                    
                    logger.debug(f"Checking repository: {repo.full_name}")
                    try:
                        if self.meets_criteria(repo):
                            repo_info = {
                                'full_name': repo.full_name,
                                'url': repo.html_url,
                                'stars': repo.stargazers_count,
                                'issues_count': repo.get_issues().totalCount,
                                'prs_count': repo.get_pulls().totalCount,
                                'description': repo.description,
                                'created_at': repo.created_at.isoformat(),
                                'last_updated': repo.updated_at.isoformat(),
                            }
                            self.new_repos.append(repo_info)
                            self.existing_repos.add(repo.full_name)
                            new_repos_found += 1
                            pbar.update(1)
                            
                            # Save progress periodically
                            if new_repos_found % 10 == 0:
                                self.save_new_repos()
                    except GithubException as e:
                        if e.status == 403:  # Rate limit exceeded
                            logger.error("Rate limit exceeded during repository check")
                            self.check_rate_limit()
                        else:
                            logger.warning(f"Error checking repository {repo.full_name}: {str(e)}")
                        continue
                            
        except RateLimitExceededException:
            logger.error("GitHub API rate limit exceeded!")
            self.save_new_repos()
        except Exception as e:
            logger.error(f"Error during repository search: {str(e)}")
            self.save_new_repos()
            
        logger.info(f"Search completed. Checked {repos_checked} repositories, found {new_repos_found} matching all criteria")
        return self.new_repos

    def merge_and_save_final(self):
        """Merge existing and new repositories and save to final file."""
        temp_file = os.path.join(RAW_DATA_DIR, 'new_repos_temp.json')
        
        # Load existing repositories
        existing_data = []
        if os.path.exists(self.main_file):
            with open(self.main_file, 'r') as f:
                existing_data = json.load(f)
        
        # Load new repositories
        new_data = []
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                new_data = json.load(f)
        
        # Merge repositories
        all_repos = existing_data + new_data
        
        # Save merged data
        with open(self.main_file, 'w') as f:
            json.dump(all_repos, f, indent=2)
        
        # Remove temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        logger.info(f"Merged and saved total of {len(all_repos)} repositories")
        return all_repos

def main():
    """Main function to run the repository finder."""
    load_dotenv()
    github_token = os.getenv('GITHUB_TOKEN')
    
    if not github_token:
        logger.error("GitHub token not found! Please set GITHUB_TOKEN environment variable.")
        return
    
    finder = RepositoryFinder(github_token)
    new_repos = finder.find_repositories()
    logger.info(f"Found {len(new_repos)} new repositories")
    
    all_repos = finder.merge_and_save_final()
    logger.info(f"Total repositories after merge: {len(all_repos)}")

if __name__ == "__main__":
    main() 