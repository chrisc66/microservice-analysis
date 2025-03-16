import os
import time
import atexit
from functools import wraps
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from github import Github, RateLimitExceededException, GithubException

# Configuration
MAX_WORKERS = 12
MAX_RETRIES = 3
REQUEST_DELAY = 2
MAX_ISSUES = 100

# Setup caching (disable in production)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    raise EnvironmentError("Please set the GITHUB_TOKEN environment variable.")

class GitHubScraper:
    def __init__(self, token):
        self.g = Github(token)
        self.rate_lock = Lock()
        atexit.register(self.cleanup)

    def cleanup(self):
        self.g.close()

    def handle_rate_limit(self):
        with self.rate_lock:
            rate_limit = self.g.get_rate_limit().core
            if rate_limit.remaining < 5:
                reset_time = rate_limit.reset.timestamp() - time.time()
                sleep_time = max(reset_time, 0) + 5
                print(f"Rate limit approaching. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

    @staticmethod
    def retry(max_retries=MAX_RETRIES, delay=REQUEST_DELAY):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(self, *args, **kwargs)
                    except RateLimitExceededException:
                        self.handle_rate_limit()
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"Retry {attempt+1}/{max_retries} for {args}: {e}")
                        time.sleep(delay * (attempt + 1))
                return None
            return wrapper
        return decorator

    @retry()
    def get_repo_metrics(self, owner, repo):
        try:
            self.handle_rate_limit()
            repo_obj = self.g.get_repo(f"{owner}/{repo}")
            
            # Basic repository info
            metrics = {
                "owner": owner,
                "repo": repo,
                "stars": repo_obj.stargazers_count,
                "size_kb": repo_obj.size,
                "created_at": repo_obj.created_at,
                "updated_at": repo_obj.updated_at,
                "contributors": repo_obj.get_contributors().totalCount
            }

            # Issue metrics with pagination control
            issues = repo_obj.get_issues(state="closed", sort="created", direction="asc")
            issue_sample = list(itertools.islice(issues, MAX_ISSUES))
            
            resolution_times = []
            response_times = []
            
            for issue in issue_sample:
                if issue.pull_request:
                    continue
                
                if issue.closed_at:
                    resolution_times.append(
                        (issue.closed_at - issue.created_at).total_seconds() / 3600
                    )
                    
                comments = issue.get_comments()
                if comments.totalCount > 0:
                    first_comment = comments[0]
                    response_times.append(
                        (first_comment.created_at - issue.created_at).total_seconds() / 3600
                    )

            metrics.update({
                "avg_resolution_time": np.mean(resolution_times) if resolution_times else None,
                "avg_response_time": np.mean(response_times) if response_times else None,
                "issues_analyzed": len(issue_sample)
            })

            return metrics

        except GithubException as e:
            if e.status == 404:
                print(f"Repository {owner}/{repo} not found")
            else:
                print(f"GitHub API error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error processing {owner}/{repo}: {e}")
            return None

def main():
    scraper = GitHubScraper(GITHUB_TOKEN)
    
    repositories = [
        {"owner": "dapr", "repo": "dapr"},
        {"owner": "conductor-oss", "repo": "conductor"},
        {"owner": "apache", "repo": "skywalking"},
        {"owner": "jhipster", "repo": "generator-jhipster"},
        {"owner": "zeromicro", "repo": "go-zero"},
        {"owner": "Netflix", "repo": "genie"},
        {"owner": "dotnet", "repo": "eShop"},
        {"owner": "dotnet", "repo": "aspnetcore"},
        {"owner": "istio", "repo": "istio"},
        {"owner": "jaegertracing", "repo": "jaeger"},
        {"owner": "openebs", "repo": "openebs"},
        {"owner": "spinnaker", "repo": "spinnaker"},
        {"owner": "getsentry", "repo": "sentry"},
        {"owner": "nytimes", "repo": "gizmo"},
        {"owner": "sitewhere", "repo": "sitewhere"},
        {"owner": "magda-io", "repo": "magda"},
        {"owner": "apolloconfig", "repo": "apollo"},
        {"owner": "GoogleCloudPlatform", "repo": "microservices-demo"},
        {"owner": "instana", "repo": "robot-shop"},
        {"owner": "paulc4", "repo": "microservices-demo"},
        {"owner": "microservices-demo", "repo": "carts"},
        {"owner": "begmaroman", "repo": "go-micro-boilerplate"},
        {"owner": "mehdihadeli", "repo": "food-delivery-microservices"},
        {"owner": "mehdihadeli", "repo": "game-leaderboard-microservices"},
        {"owner": "acmeair", "repo": "acmeair"},
        {"owner": "venkataravuri", "repo": "e-commerce-microservices-sample"},
        {"owner": "digota", "repo": "digota"},
        {"owner": "dotnet-architecture", "repo": "eShopOnContainers"},
        {"owner": "microservices-patterns", "repo": "ftgo-application"},
        {"owner": "TheDigitalNinja", "repo": "million-song-library"},
        {"owner": "idugalic", "repo": "micro-company"},
        {"owner": "aspnet", "repo": "MusicStore"},
        {"owner": "sqshq", "repo": "PiggyMetrics"},
        {"owner": "EdwinVW", "repo": "pitstop"},
        {"owner": "instana", "repo": "robot-shop"},
        {"owner": "JoeCao", "repo": "qbike"},
        {"owner": "spring-petclinic", "repo": "spring-petclinic-microservices"},
        {"owner": "Staffjoy", "repo": "V2"},
        {"owner": "LandRover", "repo": "StaffjoyV2"},
        {"owner": "DescartesResearch", "repo": "TeaStore"},
        {"owner": "FudanSELab", "repo": "train-ticket"},
    ]

    results = []
    futures = defaultdict(dict)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        for repo in repositories:
            owner = repo["owner"]
            name = repo["repo"]
            futures[(owner, name)]["future"] = executor.submit(
                scraper.get_repo_metrics, owner, name
            )
            futures[(owner, name)]["data"] = repo

        # Process results with progress bar
        with tqdm(total=len(repositories), desc="Processing repositories") as pbar:
            for (owner, name), data in list(futures.items()):
                future = data["future"]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Failed to process {owner}/{name}: {e}")
                finally:
                    pbar.update(1)

    # Save and analyze results
    if results:
        df = pd.DataFrame(results)
        print("\nCollected metrics:")
        print(df[["owner", "repo", "stars", "contributors", "avg_resolution_time"]])
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"github_metrics_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nSaved metrics to {filename}")

        # Generate visualization
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x="repo", y="stars", hue="owner", data=df)
        ax.set_title("Repository Stars Comparison")
        ax.set_ylabel("Stars")
        ax.set_xlabel("Repository")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("github_stars_comparison.png")
        plt.close()
        print("Generated visualization: github_stars_comparison.png")
    else:
        print("No data collected")

if __name__ == "__main__":
    main()
