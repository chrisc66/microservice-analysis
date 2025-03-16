"""
Configuration settings for the GitHub data collection and analysis pipeline.
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Repository selection criteria
MIN_STARS = 100
MIN_ISSUES = 50
MIN_PRS = 30
ACTIVITY_MONTHS = 12  # Look for activity in the last 12 months

# File type categories
CONFIG_FILES = ['.yaml', '.yml', '.json', '.toml', '.ini', '.conf']
APP_CODE_FILES = ['.java', '.py', '.go', '.js', '.ts', '.cs', '.cpp', '.rb']

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