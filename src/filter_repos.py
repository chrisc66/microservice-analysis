"""
Script to filter actual microservice applications from candidate repositories.
"""

import json
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Framework/tool patterns to identify non-applications
FRAMEWORK_PATTERNS = {
    # Development tools and frameworks
    'build tool': 'tool for building',
    'development framework': 'framework for developing',
    'development kit': 'development toolkit',
    'starter kit': 'starter template',
    'boilerplate code': 'template for',
    'code generator': 'generates code',
    
    # Libraries and SDKs
    'client library': 'library for',
    'sdk': 'software development kit',
    'api wrapper': 'wrapper around',
    'toolkit': 'set of tools',
    
    # Templates and bootstraps
    'project template': 'template for creating',
    'application template': 'template to create',
    'bootstrap template': 'bootstrap for',
    'scaffold': 'scaffolding for',
    
    # Learning resources
    'learning resource': 'learn how to',
    'tutorial': 'step by step',
    'example implementation': 'example of how',
    'sample project': 'sample application',
}

# Application indicators that suggest real microservice applications
APPLICATION_INDICATORS = {
    # Architecture patterns
    'microservices architecture': 5,
    'distributed system': 4,
    'service mesh': 4,
    'event-driven architecture': 4,
    
    # Concrete implementations
    'production ready': 5,
    'production deployment': 5,
    'in production': 4,
    'deployed to': 3,
    
    # Multiple services
    'multiple services': 4,
    'several microservices': 4,
    'collection of services': 4,
    'composed of services': 4,
    
    # Infrastructure and deployment
    'kubernetes cluster': 3,
    'docker compose': 3,
    'service discovery': 3,
    'load balancing': 3,
    
    # Communication patterns
    'inter-service communication': 4,
    'message queue': 3,
    'api gateway': 3,
    'service registry': 3,
    
    # Real application indicators
    'business logic': 4,
    'user interface': 3,
    'database': 2,
    'authentication': 2,
}

def is_framework_or_tool(repo: dict) -> bool:
    """
    Determine if a repository is a framework/tool rather than an application.
    
    Args:
        repo (dict): Repository information
        
    Returns:
        bool: True if the repository appears to be a framework/tool
    """
    name = repo['full_name'].lower()
    description = (repo['description'] or '').lower()
    topics = [topic.lower() for topic in repo.get('topics', [])]
    
    # Join all text for analysis
    all_text = f"{name} {description} {' '.join(topics)}"
    
    # Check for framework/tool patterns
    framework_score = 0
    for pattern, context in FRAMEWORK_PATTERNS.items():
        if pattern in all_text:
            # If we find both the pattern and its context, it's likely a framework
            if context in all_text:
                framework_score += 2
            else:
                framework_score += 1
    
    # Strong framework indicators in name
    name_parts = name.split('/')[-1].split('-')
    framework_name_indicators = {
        'framework', 'lib', 'sdk', 'toolkit', 'generator',
        'bootstrap', 'template', 'scaffold', 'starter'
    }
    if any(part in framework_name_indicators for part in name_parts):
        framework_score += 2
    
    # Check topics for framework indicators
    framework_topics = {
        'framework', 'library', 'sdk', 'toolkit', 'boilerplate',
        'starter', 'template', 'generator', 'bootstrap'
    }
    if any(topic in framework_topics for topic in topics):
        framework_score += 2
    
    return framework_score >= 3  # Require multiple strong indicators

def is_microservice_application(repo: dict) -> bool:
    """
    Determine if a repository is a real microservice application.
    
    Args:
        repo (dict): Repository information
        
    Returns:
        bool: True if the repository is a microservice application
    """
    # First check if it's a framework/tool
    if is_framework_or_tool(repo):
        logger.debug(f"Rejecting {repo['full_name']} - appears to be a framework/tool")
        return False
    
    name = repo['full_name'].lower()
    description = (repo['description'] or '').lower()
    topics = [topic.lower() for topic in repo.get('topics', [])]
    
    # Join all text for analysis
    all_text = f"{name} {description} {' '.join(topics)}"
    
    # Calculate application score based on indicators
    app_score = 0
    matched_indicators = []
    
    for indicator, weight in APPLICATION_INDICATORS.items():
        if indicator in all_text:
            app_score += weight
            matched_indicators.append(indicator)
    
    # Additional score for microservice-specific topics
    microservice_topics = {
        'microservices', 'microservice', 'distributed-systems',
        'kubernetes', 'docker', 'cloud-native'
    }
    for topic in topics:
        if topic in microservice_topics:
            app_score += 2
            matched_indicators.append(f"topic:{topic}")
    
    # Log the decision with details
    if app_score >= 6:
        logger.info(f"Accepting {repo['full_name']} (score: {app_score}) - matched indicators: {', '.join(matched_indicators)}")
        return True
    else:
        logger.debug(f"Rejecting {repo['full_name']} (score: {app_score}) - insufficient application indicators")
        return False

def filter_repositories(input_file: str, output_file: str) -> None:
    """
    Filter microservice applications from candidate repositories.
    
    Args:
        input_file (str): Path to input JSON file with candidate repositories
        output_file (str): Path to output JSON file for filtered repositories
    """
    try:
        # Load candidate repositories
        with open(input_file, 'r') as f:
            repos = json.load(f)
        
        logger.info(f"Loaded {len(repos)} repositories from {input_file}")
        
        # Filter microservice applications
        microservice_apps = [
            repo for repo in repos
            if is_microservice_application(repo)
        ]
        
        # Save filtered repositories
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(microservice_apps, f, indent=2)
        
        logger.info(f"Found {len(microservice_apps)} microservice applications")
        logger.info(f"Saved filtered repositories to {output_file}")
        
        # Print accepted repositories with their scores
        logger.info("\nAccepted repositories:")
        for repo in microservice_apps:
            logger.info(f"- {repo['full_name']}")
            
    except Exception as e:
        logger.error(f"Error filtering repositories: {str(e)}")
        raise

def main():
    """Main function to run the repository filter."""
    # Set up file paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw'
    
    input_file = data_dir / 'candidate_repos.json'
    output_file = data_dir / 'microservice_repos.json'
    
    logger.info("Starting repository filter")
    filter_repositories(str(input_file), str(output_file))
    logger.info("Finished filtering repositories")

if __name__ == "__main__":
    main() 