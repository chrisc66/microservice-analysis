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
python src/data_analyzer.py
```
This will generate:
- Statistical analysis
- Visualizations
- A comprehensive report in `data/results/analysis_report.md`

## Data Collection Process

1. **Repository Selection Criteria**:
   - Minimum stars: 100+
   - Active microservice architecture indicators:
     - Repository metadata (description and topics)
     - Configuration files (Docker, Kubernetes, etc.)
     - Directory structure (k8s, helm, deploy, etc.)
     - Deployment configurations
   - Recent development activity (commits within last few months)
   - Minimum number of issues and PRs
   - Comprehensive microservice detection:
     - Keywords in description and topics (microservice, kubernetes, docker, etc.)
     - Common configuration files (docker-compose, k8s manifests, etc.)
     - Service-oriented directory structure
     - Deployment and orchestration files

2. **Data Points Collected**:
   - Issue metadata (creation, closure, labels)
   - PR information (changes, merge time)
   - File changes (config vs. application code)
   - Resolution times

3. **Data Cleaning**:
   - Removal of outliers
   - Categorization of changes
   - Deduplication of references

## Analysis Methodology

1. **Data Preparation and Cleaning**:
   - Datetime standardization
   - Outlier removal using IQR method (1.5 * IQR rule)
   - Categorization of changes (config vs. application code)

2. **Basic Statistical Analysis**:
   - Descriptive Statistics:
     - Mean and median resolution times
     - Bug counts and proportions by change type
   - Simple Inferential Statistics:
     - Independent t-test comparing config vs. app code resolution times
     - Significance level: α = 0.05

3. **Visualizations**:
   - Resolution time distributions:
     - Box plots showing quartiles and outliers
     - Violin plots showing probability density
   - Temporal analysis:
     - Monthly average resolution times by change type
     - Trend line plots

4. **Report Generation**:
   - Summary statistics
   - Test results and significance
   - Key findings and recommendations
   - Visual analysis with generated plots

## Results

The analysis results can be found in:
- `data/results/analysis_report.md`: Comprehensive findings
- `data/results/resolution_time_distribution.png`: Distribution visualization
- `data/results/monthly_trends.png`: Temporal analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
