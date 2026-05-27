````markdown
# Insurance Risk Analytics

End-to-end insurance risk analytics and predictive modeling pipeline built using Python, statistical analysis, machine learning, DVC, and CI/CD best practices.

---

# Project Objectives

This project analyzes insurance policy and claims data to:

- Perform exploratory data analysis (EDA)
- Identify claim and premium risk patterns
- Conduct statistical hypothesis testing
- Engineer predictive risk features
- Build machine learning models for claim prediction
- Implement reproducible data versioning with DVC
- Maintain clean CI/CD workflows with GitHub Actions

---

# Tech Stack

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Pytest
- Ruff
- DVC
- GitHub Actions

---

# Project Structure

```text
insurance-risk-analytics/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── .gitkeep
│   ├── .gitignore
│   └── *.dvc
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
│
├── tests/
│   ├── data/
│   │   └── sample_insurance_data.txt
│   └── test_data_loader.py
│
├── requirements.txt
├── .gitignore
├── dvc.yaml
└── README.md
````

---

# Dataset

The original dataset is excluded from GitHub tracking and managed using DVC.

Dataset characteristics:

* Pipe-delimited `.txt`
* Insurance policy records
* Vehicle attributes
* Premium data
* Claims data
* Geographic risk variables

---

# Key Features Engineered

* LossRatio
* Margin
* VehicleAge
* HasClaim

---

# Exploratory Data Analysis

EDA includes:

* Claim distribution analysis
* Outlier analysis
* Geographic risk segmentation
* Temporal trend analysis
* Vehicle model risk analysis
* Missing value analysis

---

# Statistical Hypothesis Testing

Implemented tests include:

* Welch’s T-Test
* Chi-Squared Test

Used to evaluate:

* Provincial claim severity differences
* Demographic claim associations

---

# Machine Learning Models

Implemented models:

* Random Forest Regressor
* XGBoost Regressor

Evaluation metrics:

* RMSE
* R² Score

---

# Data Versioning with DVC

DVC is used to:

* Track processed datasets
* Keep GitHub repository lightweight
* Ensure reproducibility

Raw datasets are not committed to GitHub.

---

# CI/CD Pipeline

GitHub Actions workflow automatically:

* Installs dependencies
* Runs Ruff linting
* Executes Pytest test suite

---

# Installation

## Clone Repository

```bash
git clone https://github.com/arsema-s/insurance-risk-analytics.git
cd insurance-risk-analytics
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running Tests

```bash
python -m pytest
```

---

# Running Lint Checks

```bash
ruff check .
```

---

# Running Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Run notebooks in order:

1. `01_eda.ipynb`
2. `02_hypothesis_testing.ipynb`
3. `03_modeling.ipynb`

---

# Reproducibility

To pull DVC-managed datasets:

```bash
dvc pull
```

---

# Author

Arsema Esayas

Insurance Risk Analytics Project

```
```


