"""
Repository finder module to identify suitable microservice repositories for analysis.
"""

import os
from datetime import datetime, timedelta, timezone
import json
from typing import List, Dict, Set, Optional, Generator
import logging
import sys
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
from urllib3.util import Retry

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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Search configuration
LANGUAGES = ["java", "python", "go", "javascript", "typescript"]
SEARCH_TERMS = {
    "microservice": [
        "microservices in:name,description,topics",
        "microservice architecture in:description",
        "distributed system in:description",
        "service-oriented in:description"
    ],
    "cloud_native": [
        "cloud native application in:description",
        "distributed system in:description",
        "service oriented architecture in:description",
        "containerized services in:description",
        "kubernetes application in:description"
    ]
}

# Known good microservice repositories
WHITELIST_REPOS = {
    'googlecloudplatform/microservices-demo',
    'microservices-demo/microservices-demo',
    'weaveworks/microservices-demo',
    'aws-samples/aws-microservices-deploy-options',
    'spring-petclinic/spring-petclinic-microservices',
    'dotnet-architecture/eShopOnContainers'
}

# Infrastructure/framework keywords to exclude
EXCLUDE_KEYWORDS = {
    'framework', 'library', 'sdk', 'toolkit', 'boilerplate',
    'starter', 'learning', 'infrastructure', 'platform', 
    'runtime', 'engine', 'orchestrator'
}

# Major projects to exclude
EXCLUDE_PROJECTS = {
    'pytorch', 'tensorflow', 'consul', 'etcd', 'prometheus',
    'grafana', 'jenkins', 'gitlab', 'envoy', 'linkerd',
    'jaeger', 'kong', 'traefik', 'nginx', 'rancher',
    'openshift', 'mesos', 'nomad'
}

class TokenManager:
    """Simple manager for multiple GitHub tokens with rate limit handling."""
    
    def __init__(self, tokens: List[str]):
        """Initialize with a list of GitHub tokens."""
        if not tokens:
            raise ValueError("At least one GitHub token is required")
        
        self.tokens = tokens
        
        # Create GitHub clients with retry configuration
        self.clients = {
            token: Github(
                login_or_token=token,
                per_page=100,
                retry=3  # Enable retries with default settings
            ) for token in tokens
        }
        
        self.current_token_index = 0
        self.token_lock = threading.Lock()
        self.last_rotation_time = time.time()
        self.requests_since_rotation = 0  # Track requests since last rotation
        
        # Log initial token count
        logger.info(f"TokenManager initialized with {len(tokens)} tokens")
        self.print_all_tokens_quota()
    
    def get_client(self) -> Github:
        """Get the current GitHub client."""
        with self.token_lock:
            # Always check current token's quota before using it
            try:
                current_client = self.clients[self.tokens[self.current_token_index]]
                rate_limit = current_client.get_rate_limit()
                
                # If current token is low on either core or search quota, try to rotate
                if rate_limit.core.remaining < 100 or rate_limit.search.remaining < 5:
                    self.rotate_token()
                
            except Exception:
                # If we can't get rate limit, rotate token
                self.rotate_token()
            
            return self.clients[self.tokens[self.current_token_index]]
    
    def _find_best_token(self) -> Optional[int]:
        """Find the token with the best available quota."""
        best_score = -1
        best_index = None
        
        # First try to find a token with good quota
        for i, token in enumerate(self.tokens):
            if i == self.current_token_index:
                continue  # Skip current token
                
            try:
                client = self.clients[token]
                rate_limit = client.get_rate_limit()
                
                # If this token has good quota, use it immediately
                if rate_limit.core.remaining > 100 and rate_limit.search.remaining > 5:
                    return i
                
                # Calculate score based on remaining quota and time to reset
                core_score = rate_limit.core.remaining
                search_score = rate_limit.search.remaining * 10  # Weight search quota more heavily
                
                # Add bonus if reset time is soon
                now = time.time()
                if rate_limit.core.reset.timestamp() - now < 300:  # Reset within 5 minutes
                    core_score += 1000
                if rate_limit.search.reset.timestamp() - now < 300:
                    search_score += 1000
                
                total_score = core_score + search_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_index = i
                    
            except Exception:
                continue
        
        return best_index
    
    def rotate_token(self) -> str:
        """Rotate to the next token with available quota."""
        with self.token_lock:
            # Try to find the best available token
            best_index = self._find_best_token()
            
            if best_index is not None:
                self.current_token_index = best_index
                token = self.tokens[best_index]
                client = self.clients[token]
                try:
                    rate_limit = client.get_rate_limit()
                    logger.info(f"Switched to token {token[:8]}... (Core: {rate_limit.core.remaining}, Search: {rate_limit.search.remaining})")
                except Exception:
                    logger.info(f"Switched to token {token[:8]}... (quota unknown)")
                self.last_rotation_time = time.time()
                self.requests_since_rotation = 0
                return token
            
            # If we get here, all tokens are low on quota
            self.print_all_tokens_quota()
            
            # Find the token that will reset soonest
            earliest_reset = float('inf')
            best_token_index = 0
            
            for i, token in enumerate(self.tokens):
                try:
                    client = self.clients[token]
                    rate_limit = client.get_rate_limit()
                    
                    core_reset = rate_limit.core.reset.timestamp()
                    search_reset = rate_limit.search.reset.timestamp()
                    next_reset = min(core_reset, search_reset)
                    
                    if next_reset < earliest_reset:
                        earliest_reset = next_reset
                        best_token_index = i
                except Exception:
                    continue
            
            # All tokens are exhausted, wait for reset
            now = time.time()
            wait_time = max(0, earliest_reset - now)
            
            if wait_time > 0:
                reset_time = datetime.fromtimestamp(earliest_reset).strftime('%Y-%m-%d %H:%M:%S')
                logger.warning(f"All tokens exhausted. Waiting {wait_time:.0f} seconds for next reset at {reset_time}")
                time.sleep(min(60, wait_time))  # Wait at most 1 minute
            
            # Return the token that should have reset
            self.current_token_index = best_token_index
            self.last_rotation_time = time.time()
            self.requests_since_rotation = 0
            return self.tokens[best_token_index]
    
    def handle_rate_limit(self) -> str:
        """Handle rate limit exceeded and rotate to next token."""
        logger.warning(f"Rate limit exceeded for token {self.tokens[self.current_token_index][:8]}...")
        # Force rotation to a different token
        current = self.current_token_index
        token = self.rotate_token()
        
        # If we got the same token back, try one more time
        if self.current_token_index == current:
            self.current_token_index = (current + 1) % len(self.tokens)
            token = self.tokens[self.current_token_index]
            
        return token
    
    def print_all_tokens_quota(self):
        """Print quota information for all tokens."""
        logger.info("=== QUOTA STATUS FOR ALL TOKENS ===")
        for idx, token in enumerate(self.tokens, 1):
            try:
                client = self.clients[token]
                rate_limit = client.get_rate_limit()
                
                core_reset_time = rate_limit.core.reset.timestamp()
                search_reset_time = rate_limit.search.reset.timestamp()
                
                # Format reset times
                core_reset_str = datetime.fromtimestamp(core_reset_time).strftime('%Y-%m-%d %H:%M:%S')
                search_reset_str = datetime.fromtimestamp(search_reset_time).strftime('%Y-%m-%d %H:%M:%S')
                
                logger.info(f"Token {idx} ({token[:8]}...):")
                logger.info(f"  Core:   {rate_limit.core.remaining}/{rate_limit.core.limit} remaining (resets at {core_reset_str})")
                logger.info(f"  Search: {rate_limit.search.remaining}/{rate_limit.search.limit} remaining (resets at {search_reset_str})")
            except Exception as e:
                logger.warning(f"  Error getting quota for token {token[:8]}...: {str(e)}")
        logger.info("=====================================")

    def cleanup(self):
        """Clean up resources used by the token manager."""
        # Close GitHub clients
        for client in self.clients.values():
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error closing GitHub client: {str(e)}")

def generate_search_queries(base_query: str, search_strategy: str) -> Generator[str, None, None]:
    """Generate search queries based on languages and search terms."""
    # First, add language-agnostic queries (no language filter)
    for term in SEARCH_TERMS["microservice"]:
        yield f"{base_query} {term} sort:{search_strategy}"
    
    for term in SEARCH_TERMS["cloud_native"]:
        yield f"{base_query} {term} sort:{search_strategy}"
    
    # Then, add language-specific queries for better targeting
    for lang in LANGUAGES:
        # Language-specific base query
        lang_query = f"{base_query} language:{lang}"
        
        # Add microservice-specific terms
        for term in SEARCH_TERMS["microservice"]:
            yield f"{lang_query} {term} sort:{search_strategy}"
        
        # Add cloud-native terms
        for term in SEARCH_TERMS["cloud_native"]:
            yield f"{lang_query} {term} sort:{search_strategy}"

def save_repositories(repos: List[Dict], filename: str):
    """Save repositories to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w') as f:
            json.dump(repos, f, indent=2)
        logger.info(f"Successfully saved {len(repos)} repositories to {filename}")
    except Exception as e:
        logger.error(f"Error saving repositories to {filename}: {str(e)}")

def load_repositories(filename: str) -> List[Dict]:
    """Load repositories from a JSON file."""
    if not os.path.exists(filename):
        return []
        
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error loading repositories from {filename}: {str(e)}")
        return []

class MicroserviceValidator:
    """Validates if a repository is a real microservice application."""
    
    def __init__(self):
        self._cache = {}  # Cache validation results
        self._path_cache = {}  # Cache for directory structures
        
        # Indicators for tools/libraries rather than applications
        self.TOOL_INDICATORS = {
            'framework', 'library', 'sdk', 'toolkit', 'boilerplate',
            'starter', 'template', 'engine', 'runtime', 'platform',
            'infrastructure', 'orchestrator', 'gateway'
        }
        
        # Required infrastructure components
        self.INFRA_COMPONENTS = {
            'docker-compose.yml', 'docker-compose.yaml',
            'kubernetes', 'k8s', 'helm', 'istio',
            'prometheus', 'grafana', 'jaeger',
            'service-mesh', 'api-gateway'
        }
    
    def _get_directory_structure(self, repo: Repository) -> Set[str]:
        """Get the directory structure of a repository."""
        if repo.full_name in self._path_cache:
            return self._path_cache[repo.full_name]
            
        paths = set()
        try:
            # Check root directory and first level of subdirectories
            root_contents = repo.get_contents("")
            for content in root_contents:
                name = content.name.lower()
                paths.add(name)
                
                # For directories, check their contents but only first level
                if content.type == "dir":
                    try:
                        subdir_contents = repo.get_contents(content.path)
                        for subitem in subdir_contents:
                            paths.add(f"{name}/{subitem.name.lower()}")
                    except Exception:
                        continue
                
            self._path_cache[repo.full_name] = paths
            logger.debug(f"Found paths for {repo.full_name}: {paths}")
            
        except Exception as e:
            logger.warning(f"Error getting directory structure for {repo.full_name}: {str(e)}")
        
        return paths
    
    def _has_multiple_services(self, paths: Set[str]) -> bool:
        """Check if the repository has multiple service components."""
        # Service-related directory names
        service_dirs = {'services', 'microservices', 'apps', 'components', 'api', 'backend', 'srv'}
        service_indicators = {'-service', '-ms', '-api', '-app', '-backend', 'service-', 'svc-'}
        
        # Count service indicators
        service_count = 0
        
        # First check for service directories
        for path in paths:
            path_parts = path.split('/')
            
            # Check root level directory names
            if path_parts[0] in service_dirs:
                service_count += 1
            
            # Check for service indicators in names
            for indicator in service_indicators:
                if indicator in path:
                    service_count += 1
            
            # If we find multiple services, return True
            if service_count >= 2:
                return True
        
        return False
    
    def _has_required_infrastructure(self, paths: Set[str]) -> bool:
        """Check if the repository has required infrastructure components."""
        # Infrastructure indicators
        infra_indicators = {
            'docker', 'kubernetes', 'k8s', 'helm',
            'docker-compose', 'compose', 'deployment',
            'manifests', 'config', 'prometheus',
            'kube', 'deploy', 'ci', 'cd'
        }
        
        # Count infrastructure indicators
        found_indicators = 0
        for path in paths:
            for indicator in infra_indicators:
                if indicator in path:
                    found_indicators += 1
                    if found_indicators >= 1:  # Only require one infrastructure indicator
                        return True
        return False
    
    def _is_not_tool_or_library(self, repo: Repository) -> bool:
        """Check if the repository is not a tool or library."""
        # Quick check on name first
        name_lower = repo.name.lower()
        for indicator in {'framework', 'library', 'sdk', 'toolkit'}:
            if indicator in name_lower:
                return False
        
        # Check description
        desc_lower = (repo.description or "").lower()
        
        # If it explicitly mentions being a microservice application, accept it
        if any(term in desc_lower for term in ['microservice application', 'microservices application', 'microservice demo']):
            return True
        
        # Count negative indicators
        desc_indicators = 0
        for indicator in self.TOOL_INDICATORS:
            if indicator in desc_lower:
                desc_indicators += 1
                if desc_indicators >= 2:  # Require at least two indicators to reject
                    return False
        
        return True
    
    def _has_service_communication(self, paths: Set[str]) -> bool:
        """Check if there's evidence of service-to-service communication."""
        communication_indicators = {
            'api', 'grpc', 'proto', 'swagger', 'openapi',
            'kafka', 'rabbitmq', 'redis', 'queue', 'gateway',
            'eureka', 'consul', 'registry', 'discovery',
            'proxy', 'load-balancer', 'loadbalancer'
        }
        
        for path in paths:
            for indicator in communication_indicators:
                if indicator in path:
                    return True
        return False
    
    def is_microservice_application(self, repo: Repository) -> bool:
        """
        Determine if a repository is a real microservice application.
        
        Returns:
            bool: True if the repository is a real microservice application
        """
        # Check cache first
        cache_key = repo.full_name
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Quick check for tool/library first before making API calls
            if not self._is_not_tool_or_library(repo):
                logger.debug(f"Repository {repo.full_name} appears to be a tool/library")
                self._cache[cache_key] = False
                return False
            
            # Get repository structure (with reduced API calls)
            paths = self._get_directory_structure(repo)
            
            # Apply validation criteria with detailed logging
            has_services = self._has_multiple_services(paths)
            has_infra = self._has_required_infrastructure(paths)
            has_communication = self._has_service_communication(paths)
            
            # Count how many criteria pass (require at least 2 out of 3)
            criteria_count = sum([has_services, has_infra, has_communication])
            is_valid = criteria_count >= 2
            
            # Cache the result
            self._cache[cache_key] = is_valid
            
            if is_valid:
                logger.info(f"Repository {repo.full_name} validated as a microservice application "
                          f"(services={has_services}, infra={has_infra}, communication={has_communication})")
            else:
                logger.debug(
                    f"Repository {repo.full_name} failed validation: "
                    f"services={has_services}, infra={has_infra}, communication={has_communication}"
                )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating repository {repo.full_name}: {str(e)}")
            return False

class RepositoryFinder:
    def __init__(self, tokens: List[str]):
        """Initialize the repository finder with multiple GitHub tokens."""
        self.token_manager = TokenManager(tokens)
        self.microservice_validator = MicroserviceValidator()
        
        # Load existing repositories
        self.main_file = os.path.join(RAW_DATA_DIR, 'candidate_repos.json')
        self.temp_file = os.path.join(RAW_DATA_DIR, 'new_repos_temp.json')
        
        existing_repos = load_repositories(self.main_file)
        self.existing_repos = {repo['full_name'] for repo in existing_repos}
        self.new_repos = []
        
        logger.info(f"Loaded {len(self.existing_repos)} existing repositories")
    
    def check_repository(self, repo: Repository) -> Optional[Dict]:
        """Check if a repository meets the criteria for a microservice application."""
        try:
            if repo.full_name in self.existing_repos:
                logger.debug(f"Skipping {repo.full_name} - already exists in database")
                return None
            
            # Check whitelist first
            if repo.full_name.lower() in WHITELIST_REPOS:
                logger.info(f"Accepting whitelisted repository: {repo.full_name}")
                return self._create_repo_dict(repo)
            
            # Quick checks that don't require API calls
            if repo.stargazers_count < MIN_STARS:
                logger.debug(f"Rejecting {repo.full_name} - insufficient stars ({repo.stargazers_count} < {MIN_STARS})")
                return None
            
            # Check if it's a real microservice application
            if not self.microservice_validator.is_microservice_application(repo):
                logger.debug(f"Rejecting {repo.full_name} - failed microservice validation")
                return None
            
            # Get issues and PRs count
            issues_count = repo.get_issues().totalCount
            if issues_count < MIN_ISSUES:
                logger.debug(f"Rejecting {repo.full_name} - insufficient issues ({issues_count} < {MIN_ISSUES})")
                return None
                
            prs_count = repo.get_pulls().totalCount
            if prs_count < MIN_PRS:
                logger.debug(f"Rejecting {repo.full_name} - insufficient PRs ({prs_count} < {MIN_PRS})")
                return None
            
            # Check activity
            last_commit = repo.get_commits()[0]
            last_commit_date = last_commit.commit.author.date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30 * ACTIVITY_MONTHS)
            
            if last_commit_date < cutoff_date:
                logger.debug(f"Rejecting {repo.full_name} - insufficient activity (last commit: {last_commit_date})")
                return None
            
            # If we get here, the repository meets all criteria
            logger.info(f"Repository {repo.full_name} meets all criteria - adding to candidates")
            result = self._create_repo_dict(repo)
            
            # Save progress immediately
            with threading.Lock():
                self.new_repos.append(result)
                self.existing_repos.add(result['full_name'])
                save_repositories(self.new_repos, self.temp_file)
                self.merge_and_save_final()
            
            return result
            
        except RateLimitExceededException:
            # Handle rate limit by rotating token
            self.token_manager.handle_rate_limit()
            return None
        except Exception as e:
            logger.warning(f"Error checking repository {repo.full_name}: {str(e)}")
            return None
    
    def _create_repo_dict(self, repo: Repository) -> Dict:
        """Create a dictionary with repository information."""
        return {
            'full_name': repo.full_name,
            'url': repo.html_url,
            'stars': repo.stargazers_count,
            'issues_count': repo.get_issues().totalCount,
            'prs_count': repo.get_pulls().totalCount,
            'description': repo.description,
            'created_at': repo.created_at.isoformat(),
            'last_updated': repo.updated_at.isoformat(),
            'topics': repo.get_topics()
        }

    def process_whitelisted_repos(self):
        """Process whitelisted repositories."""
        logger.info("Processing whitelisted repositories...")
        
        for repo_name in WHITELIST_REPOS:
            if repo_name not in self.existing_repos:
                try:
                    client = self.token_manager.get_client()
                    repo = client.get_repo(repo_name)
                    result = self.check_repository(repo)
                    if result:
                        self.new_repos.append(result)
                        self.existing_repos.add(result['full_name'])
                        logger.info(f"Added whitelisted repository: {result['full_name']}")
                        save_repositories(self.new_repos, self.temp_file)
                except RateLimitExceededException:
                    self.token_manager.handle_rate_limit()
                except Exception as e:
                    logger.warning(f"Error processing whitelisted repository {repo_name}: {str(e)}")

    def find_repositories(self, max_repos: int = 50, search_strategy: str = "stars") -> List[Dict]:
        """Find repositories that match our criteria."""
        try:
            # Process whitelisted repositories first
            self.process_whitelisted_repos()
            
            # Generate search queries
            base_query = f"stars:>{MIN_STARS}"
            search_queries = list(generate_search_queries(base_query, search_strategy))
            
            logger.info(f"\nBeginning search with strategy: {search_strategy}")
            
            # Show current quota status before starting
            self.token_manager.print_all_tokens_quota()
            
            # Process queries sequentially to better manage rate limits
            for query in search_queries:
                # Stop if we've found enough repositories
                if len(self.new_repos) >= max_repos:
                    break
                    
                try:
                    self._process_search_query(query, max_repos)
                except Exception as e:
                    logger.error(f"Error processing search query {query}: {str(e)}")
                    continue
                
                # Add a small delay between queries
                time.sleep(2)
            
            # Final merge
            self.merge_and_save_final()
            
            return self.new_repos
            
        except Exception as e:
            logger.error(f"Error in find_repositories: {str(e)}")
            # Emergency save of any progress
            if self.new_repos:
                save_repositories(self.new_repos, self.temp_file)
            raise

    def _process_search_query(self, search_query: str, max_repos: int) -> None:
        """Process a single search query."""
        try:
            logger.info(f"\nProcessing search query: {search_query}")
            
            # Execute search with retries
            repos = None
            max_search_attempts = 3
            
            for attempt in range(max_search_attempts):
                try:
                    client = self.token_manager.get_client()
                    repos = client.search_repositories(query=search_query)
                    total_count = repos.totalCount
                    
                    if total_count > 0:
                        logger.info(f"Found {total_count} repositories matching query")
                        break
                    else:
                        logger.info(f"No repositories found matching query")
                        return
                        
                except RateLimitExceededException:
                    if attempt < max_search_attempts - 1:
                        self.token_manager.handle_rate_limit()
                        time.sleep(2)
                    else:
                        logger.warning(f"Failed to execute search after {max_search_attempts} attempts due to rate limits")
                        return
                        
                except Exception as e:
                    logger.warning(f"Error during search attempt {attempt + 1}: {str(e)}")
                    if attempt < max_search_attempts - 1:
                        time.sleep(2)
                    else:
                        return
            
            if not repos:
                return
            
            # Process repositories one at a time
            processed_count = 0
            for repo in repos[:100]:  # Limit to first 100 repositories
                if len(self.new_repos) >= max_repos:
                    break
                    
                if repo.full_name not in self.existing_repos:
                    try:
                        # Get a fresh client for each repository
                        client = self.token_manager.get_client()
                        repo._requester = client._Github__requester
                        
                        result = self.check_repository(repo)
                        if result:
                            with threading.Lock():
                                if len(self.new_repos) < max_repos:
                                    self.new_repos.append(result)
                                    self.existing_repos.add(result['full_name'])
                                    logger.info(f"Added repository ({len(self.new_repos)}/{max_repos}): {result['full_name']}")
                                    
                                    # Save progress after each successful repository
                                    save_repositories(self.new_repos, self.temp_file)
                                    self.merge_and_save_final()
                    
                    except RateLimitExceededException:
                        self.token_manager.handle_rate_limit()
                        time.sleep(2)
                        continue
                        
                    except Exception as e:
                        logger.warning(f"Error processing repository {repo.full_name}: {str(e)}")
                
                processed_count += 1
                if processed_count % 5 == 0:  # Add delay every 5 repositories
                    time.sleep(2)
                    
        except Exception as e:
            logger.error(f"Error processing search query {search_query}: {str(e)}")
            # Save any progress we've made
            if self.new_repos:
                save_repositories(self.new_repos, self.temp_file)

    def merge_and_save_final(self):
        """Merge existing and new repositories and save to final file."""
        try:
            # Load existing repositories
            existing_data = load_repositories(self.main_file)
            new_data = load_repositories(self.temp_file)
            
            # Create set of existing repository names for deduplication
            existing_names = {repo['full_name'] for repo in existing_data}
            
            # Only add new repositories that don't already exist
            merged_repos = existing_data.copy()
            new_count = 0
            
            for repo in new_data:
                if repo['full_name'] not in existing_names:
                    merged_repos.append(repo)
                    existing_names.add(repo['full_name'])
                    new_count += 1
            
            # Save merged data
            save_repositories(merged_repos, self.main_file)
            logger.info(f"Merged {new_count} new repositories into main file (total: {len(merged_repos)})")
            
            # Remove temporary file
            try:
                if os.path.exists(self.temp_file):
                    os.remove(self.temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error during merge: {str(e)}")
            # Keep the temp file in case of merge error
            logger.warning("Keeping temporary file due to merge error")

def main():
    """Main function to run the repository finder."""
    load_dotenv()
    
    # Load GitHub tokens
    github_tokens = []
    for i in range(1, 5):
        token = os.getenv(f'GITHUB_TOKEN_{i}')
        if token:
            github_tokens.append(token)
            logger.info(f"Loaded token {i}: {token[:8]}...")
    
    if not github_tokens:
        logger.error("No GitHub tokens found! Please set GITHUB_TOKEN_1 through GITHUB_TOKEN_4 environment variables.")
        return
        
    logger.info(f"Starting repository finder with {len(github_tokens)} tokens")
    
    finder = None
    try:
        finder = RepositoryFinder(github_tokens)
        
        # Try different search strategies
        for strategy in ["stars", "updated", "forks"]:
            logger.info(f"\nTrying search strategy: {strategy}")
            new_repos = finder.find_repositories(max_repos=50, search_strategy=strategy)
            logger.info(f"Found {len(new_repos)} new repositories using {strategy} strategy")
            time.sleep(5)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Clean up resources
        if finder and hasattr(finder, 'token_manager'):
            try:
                finder.token_manager.cleanup()
                logger.info("Cleaned up token manager resources")
            except Exception as e:
                logger.warning(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main() 