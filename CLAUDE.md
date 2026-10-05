# CLAUDE.md — Loan Interest Rate Recommender
### Full Project Specification & Build Plan
**Course:** Machine Learning | **Team Size:** 5 | **Semester:** 5th Sem B.Tech CSE-AI

---

## 1. PROJECT OVERVIEW

### 1.1 One-line pitch
A web-based system that predicts a fair loan interest rate for an applicant based on their financial profile, shows a confidence range instead of a single number, and explains *why* it made that prediction — modeled on real fintech risk-pricing engines.

### 1.2 Problem statement
Traditional loan pricing is opaque to applicants and often inconsistent across human underwriters. This project builds a **regression-based interest rate recommendation engine** trained on historical LendingClub loan data, wrapped in an interactive web interface that lets a user adjust applicant details and see the recommended rate update live, along with a prediction interval and a breakdown of which factors influenced the result.

### 1.3 Why this project (justification for report/viva)
- Directly maps onto the taught syllabus: Data Retrieval → Cleaning → EDA → Feature Engineering → Supervised Learning (Regression) → Train/Test Split → Cross-Validation.
- Uses **Linear Regression** as the primary taught algorithm (`sklearn.linear_model.LinearRegression`), matching lecture syntax exactly.
- Extends beyond syllabus with prediction intervals and explainability — demonstrating initiative without abandoning the core taught method.
- Real-world relevance: fintech risk-based pricing is an active, growing application domain.
- Dataset size (~400–500MB after filtering) fits laptop-based training (MacBook Air M5, 16GB RAM) without requiring heavy GPU compute.

### 1.4 Learning outcomes demonstrated
- Multi-source data handling (CSV retrieval, matching Module 1)
- Data cleaning and inspection (duplicates, missing values, outliers — Module 2)
- Exploratory Data Analysis with Matplotlib/Seaborn (Module 2)
- Feature engineering: encoding, scaling, transformation (Module 3)
- Supervised learning via regression (Module 4)
- Train/test splitting and K-Fold cross-validation (Module 4)
- End-to-end deployment of an ML model behind a live web interface

---

## 2. DATASET

### 2.1 Source
**Name:** LendingClub Loan Data
**Link:** https://www.kaggle.com/datasets/wordsforthewise/lending-club
**File to use:** `accepted_2007_to_2018Q2.csv` (ignore the rejected-loans file — too few usable columns)

### 2.2 Size management
Full file can exceed 1.5GB with 150 columns and 2M+ rows. To land in the 400–500MB target:
- Filter to loans issued in a 2–3 year window (e.g., 2016–2018)
- Drop all columns not listed in Section 2.3 below
- Save the filtered subset as a new working CSV (`loan_data_filtered.csv`) — this becomes your actual working file for the rest of the pipeline

### 2.3 Full feature dictionary

| # | Column | Type | Description | Role |
|---|--------|------|--------------|------|
| 1 | `int_rate` | float | Interest rate on the loan (%) | **TARGET VARIABLE** |
| 2 | `loan_amnt` | float | Amount requested by borrower | Input feature |
| 3 | `term` | categorical (36/60 months) | Loan repayment term | Input feature (convert to int) |
| 4 | `annual_inc` | float | Borrower's self-reported annual income | Input feature |
| 5 | `emp_length` | categorical | Years employed ("< 1 year" to "10+ years") | Input feature (convert to ordinal) |
| 6 | `home_ownership` | categorical | RENT / OWN / MORTGAGE | Input feature (one-hot encode) |
| 7 | `dti` | float | Debt-to-income ratio | Input feature |
| 8 | `grade` | categorical (A–G) | LendingClub's internal risk grade | Input feature (ordinal encode) |
| 9 | `sub_grade` | categorical (A1–G5) | Finer-grained risk grade | Optional — can drop if grade is sufficient |
| 10 | `purpose` | categorical | Reason for loan (debt_consolidation, credit_card, etc.) | Input feature (one-hot encode) |
| 11 | `open_acc` | int | Number of open credit lines | Input feature |
| 12 | `pub_rec` | int | Number of derogatory public records | Input feature |
| 13 | `revol_util` | float | Revolving credit utilization (%) | Input feature |
| 14 | `total_acc` | int | Total credit accounts ever opened | Input feature |
| 15 | `fico_range_low` | int | Lower bound of FICO credit score | Input feature |
| 16 | `fico_range_high` | int | Upper bound of FICO credit score | Input feature (or average the two into `fico_avg`) |
| 17 | `mort_acc` | int | Number of mortgage accounts | Input feature |

**Columns to explicitly drop:** `id`, `emp_title` (free text, too many uniques), `title`, `zip_code`, `url`, any post-origination columns (`total_pymnt`, `last_pymnt_amnt`, `recoveries`, etc. — these leak future information unavailable at prediction time).

### 2.4 Engineered features (new columns you create)
- `fico_avg` = (`fico_range_low` + `fico_range_high`) / 2
- `loan_to_income_ratio` = `loan_amnt` / `annual_inc`
- `emp_length_numeric` — convert "10+ years" → 10, "< 1 year" → 0, etc.
- `term_months` — convert "36 months" text → integer 36

---

## 3. FULL PIPELINE (mapped to syllabus modules)

### PHASE 1 — Data Retrieval (Module 1)
```python
import pandas as pd
filepath = 'data/loan_data_filtered.csv'
data = pd.read_csv(filepath)
print(data.iloc[:10])
print(data.shape)
```
**Team owner:** Member 1 (Data Engineer)
**Deliverable:** Confirmed clean load, shape printed, column list verified against Section 2.3.

### PHASE 2 — Data Cleaning & Inspection (Module 2)
Steps, in order:
1. `df = data.copy()` — always work on a copy, never mutate raw import
2. `df.info()` — confirm dtypes of every column
3. `df.duplicated().sum()` → `df = df.drop_duplicates()`
4. `df.isnull().sum()` — identify missing value counts per column
5. Handle missing values:
   - Numeric columns (`annual_inc`, `dti`, `revol_util`): impute with median
   - Categorical columns (`emp_length`, `home_ownership`): impute with mode or "Unknown" category
   - Drop rows with missing `int_rate` (can't train without the target): `df = df.dropna(subset=["int_rate"])`
6. Handle outliers using IQR method (as taught):
   - `annual_inc`: cap at 99th percentile (extreme reported incomes are likely data errors)
   - `dti`: remove negative or implausibly high values (>100)
   - `loan_amnt`: sanity check against LendingClub's known min/max ($1,000–$40,000)
7. Fix inconsistent text formatting (e.g., `.str.strip()`, `.str.title()` on categorical text columns, as shown in lecture's Garage column example)

**Team owner:** Member 1 (Data Engineer)
**Deliverable:** Clean `df`, a short written log of every row/column dropped and why (goes into report's Data Cleaning section).

### PHASE 3 — Exploratory Data Analysis (Module 2)
Required visualizations (minimum set):
1. Histogram of `int_rate` distribution — `plt.hist(df['int_rate'], bins=25)`
2. Box plot: `int_rate` grouped by `grade` — should show clean staircase (A=low rate, G=high rate)
3. Scatter plot: `dti` vs `int_rate`
4. Scatter plot: `fico_avg` vs `int_rate`
5. Correlation heatmap of all numeric features against `int_rate` (`sns.heatmap`)
6. `sns.pairplot()` on a reduced feature subset (don't pairplot all 17 features — pick top 5 correlated ones, pairplot is O(n²) and will be unreadable/slow otherwise)
7. `df.groupby('grade')['int_rate'].mean().plot()` — mirrors the exact `groupby().mean().plot()` pattern from lecture

**Team owner:** Member 2 (EDA/Feature Engineer)
**Deliverable:** A notebook section with all 7 visualizations + 3–4 sentences of written interpretation under each one (this is what evaluators specifically check for — not just "a chart exists" but "did you understand what it shows").

### PHASE 4 — Feature Engineering (Module 3)
1. **Encoding:**
   - One-hot encode: `home_ownership`, `purpose` → `pd.get_dummies(df, columns=[...])`
   - Ordinal encode: `grade` (A=1...G=7), `emp_length_numeric` (already ordinal by construction)
2. **Scaling:**
   - Apply `StandardScaler` to: `annual_inc`, `loan_amnt`, `dti`, `revol_util`, `fico_avg`
   - If `annual_inc` is right-skewed (check via histogram from Phase 3) — apply `np.log1p()` transform **before** scaling, matching the lecture's log-transform example exactly
3. **New feature creation:** compute `loan_to_income_ratio`, `fico_avg` as described in Section 2.4

**Team owner:** Member 2 (EDA/Feature Engineer)
**Deliverable:** Fully numeric, scaled feature matrix `X`, target vector `y = df['int_rate']`, ready for modeling.

### PHASE 5 — Modeling (Module 4)
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

LR = LinearRegression()
LR = LR.fit(X_train, y_train)
y_predict = LR.predict(X_test)
```
**Baseline model:** Linear Regression (primary, matches syllabus exactly)

**Comparison models (secondary, framed as "we also tried"):**
- Random Forest Regressor
- XGBoost Regressor (or Gradient Boosting Regressor if XGBoost isn't installed)

**Team owner:** Member 3 (Model Builder #1 — Linear Regression + Random Forest)

### PHASE 6 — Cross-Validation (Module 4)
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LR, X, y, cv=5)
print("Average CV R² score:", scores.mean())
```
Run cross-validation for **all three models** (Linear Regression, Random Forest, XGBoost) and build a comparison table.

**Team owner:** Member 4 (Model Builder #2 — XGBoost + evaluation)

### PHASE 7 — Evaluation
Metrics to compute for every model:
- **RMSE** (Root Mean Squared Error) — primary metric, same units as interest rate
- **MAE** (Mean Absolute Error) — easier to explain to non-technical evaluators
- **R²** — variance explained

Final comparison table (example structure):

| Model | RMSE | MAE | R² | 5-Fold CV R² |
|-------|------|-----|-----|--------------|
| Linear Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

**Team owner:** Member 4

### PHASE 8 — Explainability Layer (extension, beyond syllabus)
- Use **SHAP** (`shap.TreeExplainer` for tree models, `shap.LinearExplainer` for Linear Regression)
- For any single prediction, extract top 3–4 contributing features and their direction (+/-)
- This feeds directly into the "Top Factors" panel of the web app (Section 5)

**Team owner:** Member 4

### PHASE 9 — Prediction Intervals (extension, beyond syllabus)
- Use quantile regression (`GradientBoostingRegressor(loss='quantile', alpha=0.1)` and `alpha=0.9` for a 80% interval), OR
- Bootstrap resampling: train the model on 50–100 bootstrap samples of the training data, predict on the same input each time, take the 10th and 90th percentile of predictions as the interval bounds
- This feeds the "confidence range" shown in the web app

**Team owner:** Member 3

### PHASE 10 — Web Interface
See Section 5 for full UI specification.
**Team owner:** Member 5 (Web Developer)

---

## 4. TEAM STRUCTURE & RESPONSIBILITIES

| # | Role | Phases Owned | Key Deliverables |
|---|------|--------------|-------------------|
| 1 | Data Engineer | Phase 1, 2 | Clean dataset, cleaning log |
| 2 | EDA / Feature Engineer | Phase 3, 4 | Visualizations + interpretations, encoded/scaled feature matrix |
| 3 | Model Builder #1 | Phase 5 (Linear Reg + RF), Phase 9 | Trained baseline models, prediction intervals |
| 4 | Model Builder #2 | Phase 5 (XGBoost), Phase 6, 7, 8 | Cross-validation results, evaluation table, SHAP explainability |
| 5 | Web Developer | Phase 10 | Full working web app (Streamlit or Flask) |

**Cross-cutting responsibility (all members):** Report writing — each member writes the section corresponding to their phase; one member (rotate or assign) compiles the final report and rehearses the demo.

---

## 5. WEB APPLICATION — FULL UI SPECIFICATION

### 5.1 Technology choice
**Recommended:** Streamlit (pure Python, fastest to build for a 5-person team with limited frontend experience)
**Alternative:** Flask backend + HTML/CSS/JS frontend (more setup time, looks more "production")

### 5.2 Overall layout
Two-panel layout, side by side on desktop (stacks vertically on mobile):
- **Left panel:** Input form
- **Right panel:** Model output

### 5.3 Left panel — Input form (every field specified)

| Field | Widget type | Range/Options | Default value |
|-------|-------------|----------------|----------------|
| Loan amount | Slider | $1,000 – $40,000 | $12,000 |
| Annual income | Slider | $10,000 – $250,000 | $45,000 |
| Employment length | Dropdown | "< 1 year" to "10+ years" | "3 years" |
| Debt-to-income ratio | Slider | 0% – 50% | 20% |
| FICO score | Slider | 300 – 850 | 680 |
| Loan term | Toggle/radio | 36 months / 60 months | 36 months |
| Home ownership | Dropdown | Rent / Own / Mortgage | Rent |
| Loan purpose | Dropdown | Debt consolidation, Credit card, Home improvement, Major purchase, Other | Debt consolidation |
| Open credit lines | Slider | 0 – 30 | 8 |
| Public records | Slider | 0 – 5 | 0 |

Below all fields: a **"Predict Rate"** button (or live-update on every slider change — recommended for the teacher demo, since no click is needed to see the effect).

### 5.4 Right panel — Output display (every element specified)

**Element 1 — Big number display**
- Predicted interest rate shown in large font (e.g., 32px), e.g., "**11.4%**"
- Color-coded: green (<10%), yellow (10–15%), red (>15%)

**Element 2 — Confidence range**
- Smaller text below the big number: "Range: 10.1% – 12.8% (80% confidence)"
- Optionally render as a horizontal bar/gauge showing where the point estimate sits within the range

**Element 3 — Grade equivalent badge**
- Small badge: "≈ LendingClub Grade C" — maps your predicted rate back to the nearest real LendingClub grade bucket for intuitive anchoring

**Element 4 — Top factors (SHAP bar chart)**
- Horizontal bar chart, top 4 features
- Each bar labeled with feature name and signed contribution, e.g.:
  - "High DTI: +1.2%"
  - "Strong FICO score: −0.8%"
  - "Short employment history: +0.5%"
  - "Low loan-to-income ratio: −0.3%"
- Bars colored red (increases rate) / green (decreases rate)

**Element 5 — Model comparison toggle (optional stretch feature)**
- Small expandable section: "See how other models would price this loan"
- Shows Linear Regression vs Random Forest vs XGBoost predictions side by side for the same input

### 5.5 Page header
- Title: "Loan Interest Rate Recommender"
- Subtitle: "Enter applicant details to get a data-driven, explainable rate recommendation"
- Small disclaimer text: "For academic demonstration purposes only — not a real lending decision tool"

### 5.6 Visual style guidance
- Clean, minimal layout — avoid clutter, this is a demo tool not a production bank app
- Use a consistent color scheme (e.g., deep blue/teal for primary elements, matching a "fintech" feel)
- Ensure sliders have visible current-value labels (not just a track) so the teacher can read exact values while dragging

---

## 6. TECH STACK SUMMARY

| Layer | Tool |
|-------|------|
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML modeling | scikit-learn (LinearRegression, RandomForestRegressor), xgboost |
| Explainability | shap |
| Web interface | streamlit (or flask + html/css/js) |
| Environment | Python 3.10+, Jupyter Notebook for development, `.py` scripts for final pipeline |
| Version control | Git + GitHub (recommended for team collaboration tracking) |

---

## 7. TIMELINE (8-week plan)

| Week | Milestone |
|------|-----------|
| 1 | Dataset downloaded, filtered to size target, Phase 1 complete |
| 2 | Phase 2 complete — full cleaning log documented |
| 3 | Phase 3 complete — all 7 EDA visualizations + interpretations |
| 4 | Phase 4 complete — encoded, scaled feature matrix ready |
| 5 | Phase 5 + 6 complete — all 3 models trained, cross-validated |
| 6 | Phase 7 + 8 complete — evaluation table, SHAP integration |
| 7 | Phase 9 + 10 — prediction intervals, web app built and connected to trained model |
| 8 | Integration testing, report writing, demo rehearsal, buffer for fixes |

---

## 8. REPORT STRUCTURE (suggested)

1. Abstract
2. Introduction & Motivation
3. Related Work (brief — mention real-world risk-based pricing systems)
4. Dataset Description (use Section 2.3 table directly)
5. Methodology
   - 5.1 Data Cleaning
   - 5.2 EDA (include key charts)
   - 5.3 Feature Engineering
   - 5.4 Model Training
   - 5.5 Cross-Validation
6. Results & Evaluation (comparison table from Phase 7)
7. Explainability & Prediction Intervals (extension section)
8. Web Application (screenshots + architecture)
9. Conclusion & Future Scope
10. References (dataset citation, scikit-learn docs, SHAP paper)

---

## 9. VIVA / DEMO PREPARATION CHECKLIST

- [ ] Can every team member explain why Linear Regression was chosen as the baseline?
- [ ] Can every team member explain what RMSE/MAE/R² mean in plain language?
- [ ] Can every team member explain what K-Fold cross-validation does and why it matters?
- [ ] Can the team explain why certain columns were dropped (data leakage risk)?
- [ ] Is the live demo tested on the actual presentation laptop beforehand (not just a dev machine)?
- [ ] Does the team have a fallback (screenshots/recorded video) in case of live demo failure (no internet, crashed app, etc.)?
- [ ] Can the team explain the SHAP output for at least one specific example prediction?

---

## 10. FUTURE SCOPE (for report's closing section)
- Extend to a classification head predicting loan default risk alongside interest rate (same cleaned dataset, dual-output model)
- Deploy live on a cloud platform (Streamlit Community Cloud / Render) instead of local-only demo
- Incorporate macroeconomic indicators (interest rate environment, inflation) as additional features
- Explore fairness auditing — check whether predicted rates correlate unfairly with protected attributes not included as features but potentially inferable
