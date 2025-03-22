# Microservice Bug Resolution Analysis

This project analyzes the relationship between file types (configuration vs. application code) and bug resolution times in microservice-based projects. It uses the GitHub API to collect and analyze bug-related data from popular microservice repositories.

## Research Question

Does modifying configuration files (e.g., .yaml, .json) vs. application code (e.g., .java, .py) impact bug resolution time in microservice-based projects?

## Project Structure

```
.
├── src/
│   ├── repo_finder.py      # Script to identify suitable microservice repositories
│   ├── bug_data_collector.py   # Script to collect bug-related data
│   └── data_analyzer.py    # Script to analyze data and generate insights
├── config/
│   └── config.py          # Configuration settings
├── data/
│   ├── raw/              # Raw data collected from GitHub
│   ├── processed/        # Cleaned and processed data
│   └── results/          # Analysis results and visualizations
├── tests/                # Test files
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd microservice-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a GitHub Personal Access Token:
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Generate a new token with `repo` scope
   - Copy the token

5. Create a `.env` file in the project root:
```bash
echo "GITHUB_TOKEN=your_token_here" > .env
```

## Usage

1. Find suitable repositories:
```bash
python src/repo_finder.py
```
This script will:
- Search for repositories meeting the defined criteria
- Perform comprehensive microservice detection
- Track and avoid duplicate repositories
- Save progress periodically
- Handle GitHub API rate limits automatically
- Generate detailed logs for debugging
- Store results in `data/raw/candidate_repos.json`

2. Collect bug data:
```bash
python src/new_bug_collector.py
```
This will collect bug-related data and save it to `data/processed/bug_data.csv`.

3. Analyze data:
```bash
python src/kruskal_wallis_test.py
```
This will generate:
- Statistical analysis
- Visualizations
- A comprehensive report in `analysis_results`
