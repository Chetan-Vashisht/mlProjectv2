"""
Reusable prediction helper that exactly mirrors the feature engineering pipeline.
Used by the Streamlit app to ensure consistent predictions.
"""
import pandas as pd
import numpy as np
import joblib

# Must match src/feature_engineering.py exactly
EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}
GRADE_MAP = {g: i+1 for i, g in enumerate(['A','B','C','D','E','F','G'])}

SCALE_COLS = ['annual_inc_log', 'loan_amnt', 'dti', 'revol_util', 'fico_avg', 'loan_to_income_ratio']


def load_artifacts(model_name='XGBoost'):
    """Load model, scaler, and feature columns."""
    try:
        model = joblib.load(f'models/{model_name}.pkl')
    except FileNotFoundError:
        model = joblib.load('models/LinearRegression.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, feature_columns


def build_input_row(loan_amnt, annual_inc, emp_length, dti, fico_avg, term_months,
                    home_ownership, purpose, open_acc, pub_rec, revol_util=0.0,
                    total_acc=10, mort_acc=0, feature_columns=None):
    """
    Build a single-row DataFrame matching the training feature columns.
    This mirrors the feature engineering pipeline exactly.
    """
    if feature_columns is None:
        feature_columns = joblib.load('models/feature_columns.pkl')

    # Start with all zeros for all feature columns
    input_dict = {col: 0.0 for col in feature_columns}

    # Raw numeric features (these will be scaled below)
    input_dict['loan_amnt'] = float(loan_amnt)
    input_dict['dti'] = float(dti)
    input_dict['revol_util'] = float(revol_util)
    input_dict['fico_avg'] = float(fico_avg)
    input_dict['annual_inc_log'] = float(np.log1p(annual_inc))
    input_dict['loan_to_income_ratio'] = float(loan_amnt / annual_inc) if annual_inc > 0 else 0.0

    # Non-scaled numeric features
    input_dict['open_acc'] = float(open_acc)
    input_dict['pub_rec'] = float(pub_rec)
    input_dict['total_acc'] = float(total_acc)
    input_dict['mort_acc'] = float(mort_acc)
    input_dict['emp_length_numeric'] = float(emp_length) if isinstance(emp_length, (int, float)) else float(EMP_LENGTH_MAP.get(emp_length, 0))
    input_dict['term_months'] = float(term_months)

    # Grade ordinal — not available as direct user input in the web app,
    # but needed for the model. We'll estimate from FICO + other features.
    # For the web app, we don't ask for grade (it's what the model predicts around).
    # Set a neutral middle value or compute from FICO mapping.
    if 'grade_ordinal' in input_dict:
        # Approximate grade from FICO score
        if fico_avg >= 750:
            input_dict['grade_ordinal'] = 1  # A
        elif fico_avg >= 700:
            input_dict['grade_ordinal'] = 2  # B
        elif fico_avg >= 670:
            input_dict['grade_ordinal'] = 3  # C
        elif fico_avg >= 640:
            input_dict['grade_ordinal'] = 4  # D
        elif fico_avg >= 600:
            input_dict['grade_ordinal'] = 5  # E
        elif fico_avg >= 570:
            input_dict['grade_ordinal'] = 6  # F
        else:
            input_dict['grade_ordinal'] = 7  # G

    # One-hot encoded columns
    home_col = f'home_ownership_{home_ownership}'
    if home_col in input_dict:
        input_dict[home_col] = 1.0

    purpose_col = f'purpose_{purpose}'
    if purpose_col in input_dict:
        input_dict[purpose_col] = 1.0

    # Build DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])[feature_columns]
    return input_df


def scale_input(input_df, scaler):
    """Apply the same StandardScaler used during training to the scale columns."""
    input_df = input_df.copy()
    scale_cols_present = [c for c in SCALE_COLS if c in input_df.columns]
    if len(scale_cols_present) == len(SCALE_COLS):
        input_df[SCALE_COLS] = scaler.transform(input_df[SCALE_COLS])
    return input_df


def predict_rate(model, scaler, feature_columns,
                 loan_amnt, annual_inc, emp_length, dti, fico_avg, term_months,
                 home_ownership, purpose, open_acc, pub_rec,
                 revol_util=0.0, total_acc=10, mort_acc=0):
    """Full prediction pipeline: build input → scale → predict."""
    input_df = build_input_row(
        loan_amnt, annual_inc, emp_length, dti, fico_avg, term_months,
        home_ownership, purpose, open_acc, pub_rec, revol_util,
        total_acc, mort_acc, feature_columns
    )
    input_df = scale_input(input_df, scaler)
    prediction = model.predict(input_df)[0]
    return prediction


def predict_all_models(scaler, feature_columns,
                       loan_amnt, annual_inc, emp_length, dti, fico_avg, term_months,
                       home_ownership, purpose, open_acc, pub_rec,
                       revol_util=0.0, total_acc=10, mort_acc=0):
    """Predict with all 3 models for comparison."""
    input_df = build_input_row(
        loan_amnt, annual_inc, emp_length, dti, fico_avg, term_months,
        home_ownership, purpose, open_acc, pub_rec, revol_util,
        total_acc, mort_acc, feature_columns
    )
    input_df = scale_input(input_df, scaler)

    results = {}
    for name in ['LinearRegression', 'RandomForest', 'XGBoost']:
        try:
            m = joblib.load(f'models/{name}.pkl')
            results[name] = m.predict(input_df)[0]
        except FileNotFoundError:
            results[name] = None
    return results


if __name__ == "__main__":
    # Quick test with a sample applicant
    model, scaler, feature_columns = load_artifacts('XGBoost')
    rate = predict_rate(
        model, scaler, feature_columns,
        loan_amnt=12000, annual_inc=45000, emp_length=3, dti=20,
        fico_avg=680, term_months=36, home_ownership='RENT',
        purpose='debt_consolidation', open_acc=8, pub_rec=0
    )
    print(f"Sample prediction: {rate:.2f}%")

    all_rates = predict_all_models(
        scaler, feature_columns,
        loan_amnt=12000, annual_inc=45000, emp_length=3, dti=20,
        fico_avg=680, term_months=36, home_ownership='RENT',
        purpose='debt_consolidation', open_acc=8, pub_rec=0
    )
    print("All models:", {k: f"{v:.2f}%" for k, v in all_rates.items() if v is not None})
