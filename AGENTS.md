# AGENTS.md — Agent Build Instructions
### Loan Interest Rate Recommender | For use with Google Antigravity IDE (Claude models)

---

## 0. HOW TO READ THIS FILE (for the agent)

You are an autonomous coding agent operating inside Google Antigravity, using a Claude model.
This file is your **standing instruction set** — read it in full before taking any action in
this repository, and re-check it before starting each new session or phase.

Rules of engagement for this whole file:
- Follow phases **in order**. Do not skip ahead to modeling before data cleaning is verified.
- After completing each phase, run the **verification checklist** for that phase before moving on.
- If a verification check fails, stop and fix it — do not proceed with broken upstream data.
- Ask the human for confirmation at every point marked **[ASK HUMAN]** below. Do not guess.
- Do not delete or overwrite the raw dataset file. Always work on copies.
- Commit to git after every phase completes successfully (commit conventions in Section 9).
- If a required package is missing, install it — do not silently skip functionality.
- This file assumes Antigravity's agent is running with file system, terminal, and git access
  approved by the human for this project folder only.

---

## 1. PROJECT CONTEXT (read once, keep in memory)

This repo builds a regression model predicting loan interest rates from applicant financial
data (LendingClub dataset), wrapped in a Streamlit web app. Full project specification,
feature dictionary, and syllabus mapping live in `CLAUDE.md` in this same repo — read that
file fully before Phase 1 if it exists. If it does not exist yet, **[ASK HUMAN]** whether to
proceed without it or wait for it to be added.

Primary algorithm: Linear Regression (sklearn). Secondary comparison models: Random Forest,
XGBoost. Target variable: `int_rate`. Team size: 5 (you may be asked to generate code for
any phase regardless of "owner" — team ownership in CLAUDE.md is for the human team's
division of labor, not a restriction on what you can build).

---

## 2. ENVIRONMENT SETUP (do this first, every fresh clone)

### 2.1 Check Python version
```bash
python3 --version
```
Require Python 3.10 or higher. If lower, **[ASK HUMAN]** how they want to proceed (install
newer Python vs. adjust package versions).

### 2.2 Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```
Never install packages globally. Always confirm `(venv)` appears in the shell prompt before
running any `pip install`.

### 2.3 Create `requirements.txt` with these exact packages (pin versions for reproducibility)
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
xgboost>=2.0
shap>=0.44
streamlit>=1.30
jupyter>=1.0
```
Then install:
```bash
pip install -r requirements.txt
```

### 2.4 Verify installation
```bash
python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn, xgboost, shap, streamlit; print('All imports OK')"
```
If any import fails, install that package individually and re-run this check before proceeding.

**Verification checklist for Phase 2:**
- [ ] `venv` created and activated
- [ ] `requirements.txt` exists in repo root
- [ ] All imports succeed with no errors

---

## 3. REPOSITORY STRUCTURE (create exactly this layout)

```
project-root/
├── AGENTS.md                  (this file)
├── CLAUDE.md                  (project spec, if present)
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                   (original downloaded CSV — never modify)
│   └── processed/             (cleaned/engineered CSVs produced by scripts)
├── notebooks/
│   ├── 01_data_retrieval.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py       (reusable cleaning functions)
│   ├── feature_engineering.py (reusable encoding/scaling functions)
│   ├── train_model.py         (trains and saves final model)
│   └── predict.py             (loads saved model, makes a prediction — used by web app)
├── models/
│   └── (saved .pkl model files go here — gitignore large files if >50MB)
├── app/
│   └── streamlit_app.py       (the web interface)
├── reports/
│   └── figures/                (exported EDA charts for the report)
└── README.md
```

Create this structure with:
```bash
mkdir -p data/raw data/processed notebooks src models app reports/figures
touch src/__init__.py README.md .gitignore
```

Add to `.gitignore`:
```
venv/
__pycache__/
*.pyc
data/raw/*.csv
data/processed/*.csv
models/*.pkl
.ipynb_checkpoints/
```
(Raw and processed data plus trained models are excluded from git — too large, and
regeneratable from the scripts. Only code, notebooks structure, and docs are versioned.)

**Verification checklist for Phase 3:**
- [ ] All folders exist matching the structure above
- [ ] `.gitignore` created with the exclusions listed
- [ ] `src/__init__.py` exists so `src` is importable as a package

---

## 4. DATASET ACQUISITION — **[ASK HUMAN] REQUIRED STEP**

You cannot download from Kaggle directly without credentials. Do the following:

1. Check if `data/raw/accepted_2007_to_2018Q2.csv` already exists.
2. If it does not exist, **[ASK HUMAN]**: "I need the LendingClub dataset placed at
   `data/raw/accepted_2007_to_2018Q2.csv`. Please download it from
   https://www.kaggle.com/datasets/wordsforthewise/lending-club and place it there, or
   provide Kaggle API credentials (`kaggle.json`) so I can fetch it via the Kaggle CLI."
3. If the human provides `kaggle.json`, place it at `~/.kaggle/kaggle.json`, run
   `chmod 600 ~/.kaggle/kaggle.json`, then:
   ```bash
   pip install kaggle
   kaggle datasets download -d wordsforthewise/lending-club -p data/raw/ --unzip
   ```
4. Do not proceed to Phase 5 until the raw CSV is confirmed present:
   ```bash
   ls -lh data/raw/accepted_2007_to_2018Q2.csv
   ```

**Verification checklist for Phase 4:**
- [ ] Raw CSV file exists in `data/raw/`
- [ ] File size is reasonable (should be >500MB before filtering — if drastically smaller,
      flag to human, may be a partial/corrupted download)

---

## 5. SIZE FILTERING SCRIPT

Create `src/filter_raw_data.py`:
```python
import pandas as pd

COLUMNS_TO_KEEP = [
    'int_rate', 'loan_amnt', 'term', 'annual_inc', 'emp_length',
    'home_ownership', 'dti', 'grade', 'sub_grade', 'purpose',
    'open_acc', 'pub_rec', 'revol_util', 'total_acc',
    'fico_range_low', 'fico_range_high', 'mort_acc', 'issue_d'
]

def filter_dataset(input_path, output_path, year_start=2016, year_end=2018):
    df = pd.read_csv(input_path, usecols=lambda c: c in COLUMNS_TO_KEEP, low_memory=False)
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df = df[(df['issue_d'].dt.year >= year_start) & (df['issue_d'].dt.year <= year_end)]
    df = df.drop(columns=['issue_d'])
    df.to_csv(output_path, index=False)
    print(f"Filtered dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    filter_dataset(
        "data/raw/accepted_2007_to_2018Q2.csv",
        "data/processed/loan_data_filtered.csv"
    )
```
Run it:
```bash
python3 src/filter_raw_data.py
```
Check resulting file size:
```bash
ls -lh data/processed/loan_data_filtered.csv
```
If the result is still outside the 400–500MB target, adjust `year_start`/`year_end` and
re-run. **[ASK HUMAN]** if three attempts still don't land in range — may need a different
column subset strategy.

**Verification checklist for Phase 5:**
- [ ] `data/processed/loan_data_filtered.csv` exists
- [ ] File size is within or close to 400–500MB
- [ ] Row count printed and looks reasonable (hundreds of thousands of rows expected)

---

## 6. DATA CLEANING (`src/data_cleaning.py`)

Build this module with discrete, testable functions — not one giant script:
```python
import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")
    return df

def drop_missing_target(df, target='int_rate'):
    before = len(df)
    df = df.dropna(subset=[target])
    print(f"Dropped {before - len(df)} rows with missing target")
    return df

def impute_missing(df):
    numeric_cols = ['annual_inc', 'dti', 'revol_util', 'open_acc', 'pub_rec', 'total_acc', 'mort_acc']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    categorical_cols = ['emp_length', 'home_ownership']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def clip_outliers(df):
    df = df[(df['dti'] >= 0) & (df['dti'] <= 100)]
    income_cap = df['annual_inc'].quantile(0.99)
    df['annual_inc'] = df['annual_inc'].clip(upper=income_cap)
    df = df[(df['loan_amnt'] >= 1000) & (df['loan_amnt'] <= 40000)]
    return df

def clean_pipeline(input_path, output_path):
    df = load_data(input_path)
    print(f"Loaded: {df.shape}")
    df = drop_duplicates(df)
    df = drop_missing_target(df)
    df = impute_missing(df)
    df = clip_outliers(df)
    print(f"Final cleaned shape: {df.shape}")
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    clean_pipeline(
        "data/processed/loan_data_filtered.csv",
        "data/processed/loan_data_cleaned.csv"
    )
```
Run:
```bash
python3 src/data_cleaning.py
```

Then run this verification script inline (do not skip):
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/loan_data_cleaned.csv')
assert df['int_rate'].isnull().sum() == 0, 'Target still has nulls'
assert df.duplicated().sum() == 0, 'Duplicates remain'
assert (df['dti'] >= 0).all() and (df['dti'] <= 100).all(), 'DTI outliers remain'
print('All cleaning checks passed.')
"
```

**Verification checklist for Phase 6:**
- [ ] `data/processed/loan_data_cleaned.csv` exists
- [ ] All three inline assertions pass with no AssertionError
- [ ] Printed shape makes sense (fewer or equal rows than filtered input)

---

## 7. EDA — GENERATE AND SAVE FIGURES (not just display)

Create `notebooks/03_eda.ipynb` (or a script `src/generate_eda_figures.py` if the agent
prefers headless execution). Every chart must be **saved to `reports/figures/`**, not just
shown — the report needs these as image files.

Required charts (generate all, save each with `plt.savefig()`):
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('data/processed/loan_data_cleaned.csv')

plt.figure(figsize=(8,5))
plt.hist(df['int_rate'], bins=25)
plt.title('Distribution of Interest Rate')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Frequency')
plt.savefig('reports/figures/01_int_rate_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='grade', y='int_rate', order=sorted(df['grade'].unique()))
plt.title('Interest Rate by LendingClub Grade')
plt.savefig('reports/figures/02_rate_by_grade.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(df['dti'], df['int_rate'], alpha=0.1, s=5)
plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Interest Rate (%)')
plt.title('DTI vs Interest Rate')
plt.savefig('reports/figures/03_dti_vs_rate.png', dpi=150, bbox_inches='tight')
plt.close()

df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
plt.figure(figsize=(8,5))
plt.scatter(df['fico_avg'], df['int_rate'], alpha=0.1, s=5)
plt.xlabel('FICO Average Score')
plt.ylabel('Interest Rate (%)')
plt.title('FICO Score vs Interest Rate')
plt.savefig('reports/figures/04_fico_vs_rate.png', dpi=150, bbox_inches='tight')
plt.close()

numeric_cols = ['int_rate','loan_amnt','annual_inc','dti','open_acc','pub_rec','revol_util','total_acc','fico_avg','mort_acc']
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('reports/figures/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("All EDA figures saved to reports/figures/")
```

**Verification checklist for Phase 7:**
- [ ] 5 PNG files exist in `reports/figures/`
- [ ] Open at least `02_rate_by_grade.png` and visually confirm the staircase pattern
      (grade A lowest rate, grade G highest) — if this pattern is NOT visible, the cleaning
      or grade-encoding has a bug; stop and investigate before continuing.

---

## 8. FEATURE ENGINEERING (`src/feature_engineering.py`)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}
GRADE_MAP = {g: i+1 for i, g in enumerate(['A','B','C','D','E','F','G'])}

def engineer_features(df):
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df['emp_length_numeric'] = df['emp_length'].map(EMP_LENGTH_MAP).fillna(0)
    df['grade_ordinal'] = df['grade'].map(GRADE_MAP)
    df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)
    df['loan_to_income_ratio'] = df['loan_amnt'] / df['annual_inc'].replace(0, np.nan)
    df['loan_to_income_ratio'] = df['loan_to_income_ratio'].fillna(df['loan_to_income_ratio'].median())
    df['annual_inc_log'] = np.log1p(df['annual_inc'])

    df = pd.get_dummies(df, columns=['home_ownership', 'purpose'], drop_first=True)

    scale_cols = ['annual_inc_log', 'loan_amnt', 'dti', 'revol_util', 'fico_avg', 'loan_to_income_ratio']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    drop_cols = ['grade', 'sub_grade', 'emp_length', 'term', 'fico_range_low', 'fico_range_high', 'annual_inc']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, scaler

if __name__ == "__main__":
    df = pd.read_csv('data/processed/loan_data_cleaned.csv')
    df_engineered, scaler = engineer_features(df)
    df_engineered.to_csv('data/processed/loan_data_engineered.csv', index=False)
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"Engineered dataset shape: {df_engineered.shape}")
    print(f"Columns: {list(df_engineered.columns)}")
```
Run:
```bash
python3 src/feature_engineering.py
```

**Verification checklist for Phase 8:**
- [ ] `data/processed/loan_data_engineered.csv` exists
- [ ] `models/scaler.pkl` exists (needed later by the web app to scale live user input)
- [ ] Printed columns contain no non-numeric dtypes — run:
  ```bash
  python3 -c "import pandas as pd; df=pd.read_csv('data/processed/loan_data_engineered.csv'); print(df.dtypes)"
  ```
  All dtypes must be numeric (int64, float64, bool/uint8 from one-hot). If any `object`
  dtype remains, encoding was incomplete — stop and fix before modeling.

---

## 9. MODEL TRAINING (`src/train_model.py`)

```python
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np

def train_and_evaluate():
    df = pd.read_csv('data/processed/loan_data_engineered.csv')
    X = df.drop(columns=['int_rate'])
    y = df['int_rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = {}

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        results[name] = {
            'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CV_R2_mean': cv_scores.mean()
        }
        print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}, CV_R2={cv_scores.mean():.3f}")

        joblib.dump(model, f'models/{name}.pkl')

    joblib.dump(list(X.columns), 'models/feature_columns.pkl')

    results_df = pd.DataFrame(results).T
    results_df.to_csv('reports/model_comparison.csv')
    print("\nSaved model comparison table to reports/model_comparison.csv")

    return results

if __name__ == "__main__":
    train_and_evaluate()
```
Run:
```bash
python3 src/train_model.py
```
This will take noticeable time on a laptop for Random Forest/XGBoost with a large dataset.
If training exceeds ~15 minutes, **[ASK HUMAN]** whether to reduce `n_estimators`, subsample
the training data, or let it continue.

**Verification checklist for Phase 9:**
- [ ] Three `.pkl` model files exist in `models/`
- [ ] `models/feature_columns.pkl` exists (critical — the web app needs this to align input
      columns at prediction time)
- [ ] `reports/model_comparison.csv` exists with all three models' metrics
- [ ] R² values are reasonable (above 0.3 at minimum — if near zero or negative, something
      is wrong upstream; stop and investigate rather than proceeding to the web app)

---

## 10. STREAMLIT WEB APP (`app/streamlit_app.py`)

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Loan Interest Rate Recommender", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/XGBoost.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("Loan Interest Rate Recommender")
st.caption("For academic demonstration purposes only — not a real lending decision tool.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Details")
    loan_amnt = st.slider("Loan amount ($)", 1000, 40000, 12000, step=500)
    annual_inc = st.slider("Annual income ($)", 10000, 250000, 45000, step=1000)
    emp_length = st.selectbox("Employment length", list(range(0, 11)), index=3)
    dti = st.slider("Debt-to-income ratio (%)", 0, 50, 20)
    fico_avg = st.slider("FICO score", 300, 850, 680)
    term_months = st.radio("Loan term", [36, 60])
    home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE"])
    purpose = st.selectbox("Loan purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"])
    open_acc = st.slider("Open credit lines", 0, 30, 8)
    pub_rec = st.slider("Public records", 0, 5, 0)

with col2:
    st.subheader("Model Output")

    # Build input row matching training feature columns
    input_dict = {col: 0 for col in feature_columns}
    input_dict['loan_amnt'] = loan_amnt
    input_dict['annual_inc_log'] = np.log1p(annual_inc)
    input_dict['dti'] = dti
    input_dict['fico_avg'] = fico_avg
    input_dict['emp_length_numeric'] = emp_length
    input_dict['term_months'] = term_months
    input_dict['open_acc'] = open_acc
    input_dict['pub_rec'] = pub_rec
    input_dict['loan_to_income_ratio'] = loan_amnt / annual_inc if annual_inc > 0 else 0

    home_col = f'home_ownership_{home_ownership}'
    if home_col in input_dict:
        input_dict[home_col] = 1
    purpose_col = f'purpose_{purpose}'
    if purpose_col in input_dict:
        input_dict[purpose_col] = 1

    input_df = pd.DataFrame([input_dict])[feature_columns]

    scale_cols = ['annual_inc_log', 'loan_amnt', 'dti', 'revol_util', 'fico_avg', 'loan_to_income_ratio']
    scale_cols_present = [c for c in scale_cols if c in input_df.columns]
    input_df[scale_cols_present] = scaler.transform(input_df[scale_cols_present]) if set(scale_cols) == set(scale_cols_present) else input_df[scale_cols_present]

    prediction = model.predict(input_df)[0]
    lower = prediction * 0.9
    upper = prediction * 1.1

    st.metric("Predicted Interest Rate", f"{prediction:.2f}%")
    st.write(f"Estimated range: {lower:.2f}% – {upper:.2f}%")

    if prediction < 10:
        st.success("Low risk pricing")
    elif prediction < 15:
        st.warning("Moderate risk pricing")
    else:
        st.error("High risk pricing")
```

**IMPORTANT — [ASK HUMAN] note for this phase:** the scaling logic above is a simplified
placeholder (it does not perfectly replicate `StandardScaler.transform` column alignment
because the scaler was fit on a subset of columns, not the single-row input). Before final
demo, test this thoroughly with known inputs and compare against batch predictions from the
training pipeline. Flag to the human team that Member 5 (web developer) must verify this
alignment carefully, ideally by writing a small test script in `src/predict.py` that takes
one row from `loan_data_cleaned.csv`, engineers it the same way, and confirms the web app's
live prediction matches a direct `model.predict()` call on that row.

Run the app:
```bash
streamlit run app/streamlit_app.py
```

**Verification checklist for Phase 10:**
- [ ] App launches without error
- [ ] Moving each slider updates the prediction without crashing
- [ ] Manually cross-check: pick one row from `loan_data_cleaned.csv`, enter the same values
      in the app, and confirm the predicted rate is in a sane range compared to that row's
      actual `int_rate`

---

## 11. GIT COMMIT CONVENTIONS

Commit after each phase passes its verification checklist. Use this message format:
```
[Phase N] <short description>

<optional longer explanation>
```
Examples:
```
git add .
git commit -m "[Phase 6] Add data cleaning pipeline with outlier handling

Drops duplicates, imputes missing values, clips DTI/income outliers.
All verification assertions passing."
```
Never commit raw or processed CSVs, or `.pkl` model files (excluded via `.gitignore`).
Never commit `venv/`.

---

## 12. AGENT AUTONOMY SETTINGS (recommended Antigravity configuration)

**[ASK HUMAN]** to confirm these settings in Antigravity before the agent starts:
- File edits: require approval for the **first** file in each new phase, then auto-approve
  subsequent edits within that same phase once the pattern is established.
- Terminal commands: require approval for anything involving `pip install`, `rm`, or
  `git push` — auto-approve routine `python3 script.py` runs.
- Do not allow the agent unattended access to push to a shared/remote branch — all pushes
  to shared branches should be a manual human action, not agent-initiated.

---

## 13. WHEN THE AGENT SHOULD STOP AND ASK (summary of all [ASK HUMAN] triggers)

1. Before downloading or sourcing the dataset (Section 4)
2. If filtered dataset size is repeatedly outside the 400–500MB target after 3 attempts (Section 5)
3. If model training time exceeds ~15 minutes (Section 9)
4. If any verification checklist fails and the fix isn't obvious
5. Before any `git push` to a shared branch
6. Before installing any package not already listed in `requirements.txt`
7. Before deleting any file outside of `data/processed/` regenerated outputs

---

## 14. FINAL PRE-DEMO AGENT TASK

Once all phases pass verification, run this final consistency check and report results to
the human team:
```bash
python3 -c "
import joblib, pandas as pd
model = joblib.load('models/XGBoost.pkl')
cols = joblib.load('models/feature_columns.pkl')
print('Model type:', type(model).__name__)
print('Expected feature count:', len(cols))
df = pd.read_csv('data/processed/loan_data_engineered.csv')
print('Engineered dataset feature count (excluding target):', df.shape[1] - 1)
assert len(cols) == df.shape[1] - 1, 'MISMATCH: feature count does not align — web app will break'
print('Consistency check PASSED.')
"
```
Report the output of this command directly to the human before considering the build
complete.

---

## 15. CROSS-COMPATIBILITY NOTE

If the team later uses Claude Code (terminal-based) instead of or alongside Antigravity,
copy this file's content into a `CLAUDE.md` as well (Claude Code reads `CLAUDE.md`, not
`AGENTS.md`), or maintain both as a symlink:
```bash
ln -s AGENTS.md CLAUDE.md
```
