import json
from pathlib import Path

def count_repositories():
    """Count and compare repositories in both JSON files"""
    # Set up file paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw'
    
    candidate_file = data_dir / 'candidate_repos.json'
    microservice_file = data_dir / 'microservice_repos.json'
    
    # Count candidates
    with open(candidate_file, 'r') as f:
        candidate_repos = json.load(f)
    
    # Count microservices
    with open(microservice_file, 'r') as f:
        microservice_repos = json.load(f)
    
    print("\nRepository Count Summary")
    print("=" * 50)
    print(f"Candidate repositories: {len(candidate_repos)}")
    print(f"Microservice repositories: {len(microservice_repos)}")
    print(f"Filtered out: {len(candidate_repos) - len(microservice_repos)}")
    print(f"Percentage kept: {(len(microservice_repos) / len(candidate_repos) * 100):.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    count_repositories() 