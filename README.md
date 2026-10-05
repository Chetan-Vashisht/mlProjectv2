# 💰 Loan Interest Rate Recommender

A machine learning regression system that predicts fair loan interest rates from applicant financial data, built on the LendingClub dataset. Features a live interactive Streamlit web app with explainable predictions.

**Course:** Machine Learning | **Semester:** 5th Sem B.Tech CSE-AI | **Team Size:** 5

---

## 🎯 Project Overview

Traditional loan pricing is opaque to applicants and often inconsistent. This project builds a **regression-based interest rate recommendation engine** trained on real LendingClub data, wrapped in an interactive web interface where users can adjust applicant details and see the recommended rate update live.

### Models Trained
| Model | RMSE | MAE | R² | 5-Fold CV R² |
|-------|------|-----|-----|--------------|
| Linear Regression | 1.334 | 1.079 | 0.931 | 0.930 |
| Random Forest | 1.255 | 0.997 | 0.939 | 0.936 |
| XGBoost | 1.237 | 0.984 | 0.941 | 0.938 |

*(See `reports/model_comparison.csv` for final results)*

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# macOS only: XGBoost requires OpenMP
brew install libomp
```

### 2. Prepare Data
```bash
# Place the LendingClub dataset in data/raw/
# Download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club

# Run the pipeline
python3 src/filter_raw_data.py       # Filter to 2016-2018 data
python3 src/data_cleaning.py         # Clean & handle outliers
python3 src/feature_engineering.py   # Encode, scale, derive features
python3 src/train_model.py           # Train all 3 models
```

### 3. Launch Web App
```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure
```
├── AGENTS.md                  # Agent build instructions
├── CLAUDE.md                  # Full project specification
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/                   # Original CSV (not tracked in git)
│   └── processed/             # Cleaned/engineered CSVs
├── src/
│   ├── filter_raw_data.py     # Size filtering (2016-2018)
│   ├── data_cleaning.py       # Cleaning pipeline
│   ├── feature_engineering.py # Encoding, scaling, derived features
│   ├── train_model.py         # Model training & evaluation
│   └── predict.py             # Reusable prediction helper
├── models/                    # Saved .pkl model files
├── app/
│   └── streamlit_app.py       # Streamlit web interface
├── reports/
│   ├── figures/               # EDA visualizations (7 charts)
│   └── model_comparison.csv   # Model metrics comparison
└── notebooks/                 # Jupyter notebooks (optional)
```

---

## 📊 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `loan_amnt` | Numeric | Loan amount ($1,000–$40,000) |
| `annual_inc` | Numeric | Borrower's annual income |
| `emp_length` | Ordinal | Employment length (0–10 years) |
| `dti` | Numeric | Debt-to-income ratio |
| `fico_avg` | Numeric | Average FICO credit score |
| `term` | Binary | Loan term (36 or 60 months) |
| `home_ownership` | Categorical | RENT / OWN / MORTGAGE |
| `purpose` | Categorical | Loan purpose (13 categories) |
| `grade` | Ordinal | LendingClub risk grade (A–G) |
| `revol_util` | Numeric | Revolving credit utilization |
| + 4 more | Various | open_acc, pub_rec, total_acc, mort_acc |

---

## 🛠 Tech Stack

| Layer | Tool |
|-------|------|
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Modeling | scikit-learn, xgboost |
| Explainability | shap |
| Web Interface | Streamlit |
| Environment | Python 3.10+ |

---

## 📝 Dataset

**Source:** [LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Original: ~2M+ loans (2007–2018), 150+ columns
- Filtered: ~1.37M loans (2016–2018), 17 columns
- Target variable: `int_rate` (interest rate %)

---

## ⚠️ Disclaimer

This is an academic project for demonstration purposes only. It is not a real lending decision tool and should not be used for actual financial decisions.
