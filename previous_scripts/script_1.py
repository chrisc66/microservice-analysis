from github import Github
from collections import Counter

TOKEN = ""  # Replace with your GitHub token
g = Github(TOKEN)

def search_repositories(query="microservices", per_page=10):
    repositories = g.search_repositories(query=query, sort="stars", order="desc")[:per_page]
    return list(repositories)

def get_issues(repo, state="open", per_page=100):
    return list(repo.get_issues(state=state)[:per_page])

def analyze_repositories():
    repos = search_repositories()
    print(type(repos))
    print(f"Number of repos: {len(repos)}")
    bug_reports = []
    
    for repo in repos:
        print(f"Analyzing repo {repo.full_name}...")
        issues = get_issues(repo)
        print(f"Number of issues: {len(issues)}")
        
        for issue in issues:
            bug_reports.append(issue.title)
            # if "bug" in issue.title.lower() or "bug" in (issue.body or "").lower():
            #     bug_reports.append(issue.title)
    
    print(f"Number of bug reports: {len(bug_reports)}")
    print("Top Bug Reports:")
    for bug, count in Counter(bug_reports).most_common(5):
        print(f"- {bug} ({count} occurrences)")

if __name__ == "__main__":
    analyze_repositories()
