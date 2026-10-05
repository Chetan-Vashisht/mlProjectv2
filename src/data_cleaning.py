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
