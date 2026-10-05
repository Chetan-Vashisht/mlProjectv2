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
