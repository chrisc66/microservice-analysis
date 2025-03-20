"""
Configuration settings for the GitHub data collection and analysis pipeline.
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Repository selection criteria
MIN_STARS = 50  # Reduced minimum stars requirement
MIN_ISSUES = 10  # Reduced minimum issues requirement
MIN_PRS = 5     # Reduced minimum PRs requirement
ACTIVITY_MONTHS = 6  # Number of months since last commit to be considered active

# File type categories (for analyzing changes, not for filtering repositories)
CONFIG_FILES = ['.yaml', '.yml', '.json', '.toml', '.ini', '.conf']
APP_CODE_FILES = ['.java', '.py', '.go', '.js', '.ts', '.cs', '.cpp', '.rb', '.php', '.scala', '.rs', '.c', '.cpp', '.h', '.hpp']

# Keywords for identifying bug-related content
BUG_KEYWORDS = [
    'fix', 'bug', 'error', 'failure', 'crash', 'issue', 'defect',
    'exception', 'incorrect', 'wrong', 'broken'
]

# Microservice indicators
MICROSERVICE_INDICATORS = [
    'docker-compose.yml',
    'docker-compose.yaml',
    'Dockerfile',
    'kubernetes',
    'k8s.yaml',
    'k8s.yml',
    'helm',
    'service-mesh'
]

# API configuration
GITHUB_API_BASE = 'https://api.github.com'
PER_PAGE = 100  # Number of items per page for pagination

# Data storage paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
ANALYSIS_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'results')

# Time-related constants
MAX_BUG_RESOLUTION_DAYS = 365  # Maximum reasonable bug resolution time in days

# Microservice detection keywords
microservice_keywords = {
    'microservice',
    'microservices',
    'micro-service',
    'micro-services',
    'service-mesh',
    'kubernetes',
    'k8s',
    'docker',
    'container',
    'istio',
    'service-discovery',
    'api-gateway'
}

# Configuration files that indicate microservice architecture
microservice_files = {
    'docker-compose.yml',
    'docker-compose.yaml',
    'dockerfile',
    'kubernetes',
    'k8s.yaml',
    'k8s.yml',
    'helm.yaml',
    'service.yaml',
    'deployment.yaml',
    'ingress.yaml',
    'istio.yaml',
    'consul.yaml',
    'eureka.yaml'
}

# Directories that indicate microservice architecture
microservice_dirs = {
    'k8s',
    'kubernetes',
    'helm',
    'deploy',
    'deployment',
    'services',
    'microservices'
} 