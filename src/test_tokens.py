"""
Script to test GitHub API tokens and their rate limits.
"""

import os
from datetime import datetime, timezone
from github import Github
from github.RateLimit import RateLimit
from dotenv import load_dotenv
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_time_until_reset(reset_time: datetime) -> str:
    """Format the time until rate limit reset."""
    # Ensure we're using timezone-aware datetime
    now = datetime.now(timezone.utc)
    
    # Make sure reset_time is timezone-aware
    if reset_time.tzinfo is None:
        reset_time = reset_time.replace(tzinfo=timezone.utc)
    
    time_until_reset = reset_time - now
    minutes = int(time_until_reset.total_seconds() / 60)
    return f"{minutes} minutes" if minutes > 0 else "less than a minute"

def format_rate_limit(rate_limit: RateLimit) -> str:
    """Format rate limit information."""
    try:
        return (
            f"Remaining: {rate_limit.remaining}/{rate_limit.limit} "
            f"(Resets in {format_time_until_reset(rate_limit.reset)})"
        )
    except Exception as e:
        return f"Remaining: {rate_limit.remaining}/{rate_limit.limit} (Reset time unavailable)"

def test_token(token: str, token_num: int) -> tuple[bool, Github]:
    """
    Test a GitHub token and print its rate limit information.
    
    Returns:
        tuple[bool, Github]: (success status, Github client instance if successful)
    """
    try:
        g = Github(token, per_page=100, retry=0)  # Disable retry to match repo_finder settings
        rate_limit = g.get_rate_limit()
        
        logger.info(f"\nToken {token_num} Status:")
        logger.info(f"Token prefix: {token[:8]}...")
        logger.info(f"Core API: {format_rate_limit(rate_limit.core)}")
        logger.info(f"Search API: {format_rate_limit(rate_limit.search)}")
        logger.info(f"GraphQL API: {format_rate_limit(rate_limit.graphql)}")
        
        # Test a simple API call
        user = g.get_user()
        logger.info(f"Successfully authenticated as: {user.login}")
        
        # Test search API
        try:
            search_result = g.search_repositories("language:python stars:>100", sort="stars")
            logger.info(f"Search API test successful (found {search_result.totalCount} repositories)")
        except Exception as e:
            logger.warning(f"Search API test failed: {str(e)}")
        
        return True, g
        
    except Exception as e:
        logger.error(f"Token {token_num} Error: {str(e)}")
        return False, None

def print_token_summary(tokens_info: list):
    """Print a summary of all tokens."""
    logger.info("\n=== Token Summary ===")
    logger.info("Token  | Core API | Search API | GraphQL API | Reset Time")
    logger.info("-------|----------|------------|-------------|------------")
    
    total_core = 0
    total_search = 0
    total_graphql = 0
    
    for token_num, g, valid in tokens_info:
        if valid and g is not None:
            try:
                rate_limit = g.get_rate_limit()
                reset_time = format_time_until_reset(rate_limit.core.reset)
                logger.info(
                    f"Token {token_num} | "
                    f"{rate_limit.core.remaining:4d}/{rate_limit.core.limit:4d} | "
                    f"{rate_limit.search.remaining:5d}/{rate_limit.search.limit:4d} | "
                    f"{rate_limit.graphql.remaining:5d}/{rate_limit.graphql.limit:4d} | "
                    f"{reset_time:10}"
                )
                total_core += rate_limit.core.remaining
                total_search += rate_limit.search.remaining
                total_graphql += rate_limit.graphql.remaining
            except Exception as e:
                logger.error(f"Error getting rate limit for token {token_num}: {str(e)}")
                logger.info(f"Token {token_num} | Error getting rate limits")
        else:
            logger.info(f"Token {token_num} | Invalid or not configured")
    
    # Print totals
    logger.info("-------|----------|------------|-------------|------------")
    logger.info(f"Total  | {total_core:4d}      | {total_search:5d}      | {total_graphql:5d}")
    logger.info("")

def main():
    """Test all configured GitHub tokens."""
    load_dotenv()
    
    logger.info("Testing GitHub API tokens...")
    
    tokens_info = []
    valid_tokens = 0
    
    # Test each token
    for i in range(1, 5):
        token = os.getenv(f'GITHUB_TOKEN_{i}')  # Note: Using GITHUB_TOKEN_1 format
        if token:
            logger.info(f"\nTesting GITHUB_TOKEN_{i}...")
            valid, g = test_token(token, i)
            if valid:
                valid_tokens += 1
                tokens_info.append((i, g, True))
            else:
                tokens_info.append((i, None, False))
        else:
            logger.warning(f"GITHUB_TOKEN_{i} not found in environment variables")
            tokens_info.append((i, None, False))
    
    # Print summary
    if valid_tokens > 0:
        print_token_summary(tokens_info)
        logger.info(f"Found {valid_tokens} valid token(s) out of 4 possible tokens")
        logger.info("Token testing completed successfully")
    else:
        logger.error("No valid tokens found. Please check your .env file and token configurations")
        logger.error("Tokens should be named GITHUB_TOKEN_1, GITHUB_TOKEN_2, etc.")

if __name__ == "__main__":
    main() 