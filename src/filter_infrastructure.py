import json
import os
from pathlib import Path

# List of repositories to remove
REPOS_TO_REMOVE = {
    # Infrastructure/Platform Tools
    "grpc/grpc-java",
    "apache/camel",
    "TykTechnologies/tyk",
    "Kong/kong",
    "apache/apisix",
    "istio/istio",
    "polarismesh/polaris",
    "backstage/backstage",
    "kubevela/kubevela",
    "emissary-ingress/emissary",
    "kgateway-dev/kgateway",
    
    # Development Frameworks/Libraries
    "hyperf/hyperf",
    "moleculerjs/moleculer",
    "falconry/falcon",
    "apache/dubbo",
    "grpc/grpc-go",
    "stenciljs/core",
    "thisisagile/easy",
    
    # Infrastructure Management Tools
    "hashicorp/nomad",
    "microsoft/service-fabric",
    "distribworks/dkron",
    "vmware-tanzu/velero",
    "stackrox/kube-linter",
    "GoogleContainerTools/jib",
    
    # Monitoring/Observability Tools
    "pinpoint-apm/pinpoint",
    "kubeshark/kubeshark",
    "deepfence/ThreatMapper",
    "pixie-io/pixie",
    
    # Example/Template Repositories
    "rodrigorodrigues/microservices-design-patterns",
    "SAP-samples/kyma-runtime-extension-samples",
    "eventuate-tram/eventuate-tram-examples-customers-and-orders",
    "mehdihadeli/awesome-software-architecture",
    
    # Message Brokers/Streaming Platforms
    "redpanda-data/redpanda",
    "infinyon/fluvio",
    "apache/rocketmq-site",
    
    # Storage/Database Systems
    "seaweedfs/seaweedfs",
    "juicedata/juicefs",
    "hstreamdb/hstream",
    "apache/ozone"
}

def filter_repositories():
    """Remove infrastructure and tool repositories from microservice_repos.json"""
    # Set up file paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw'
    input_file = data_dir / 'microservice_repos.json'
    
    # Read current repositories
    with open(input_file, 'r') as f:
        repos = json.load(f)
    
    print(f"Original repository count: {len(repos)}")
    
    # Filter out infrastructure and tool repositories
    filtered_repos = [
        repo for repo in repos
        if repo['full_name'] not in REPOS_TO_REMOVE
    ]
    
    print(f"Filtered repository count: {len(filtered_repos)}")
    print(f"Removed {len(repos) - len(filtered_repos)} repositories")
    
    # Save filtered repositories back to the file
    with open(input_file, 'w') as f:
        json.dump(filtered_repos, f, indent=2)
    
    print(f"Updated {input_file}")

if __name__ == "__main__":
    filter_repositories() 