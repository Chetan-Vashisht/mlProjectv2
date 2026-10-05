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
        "data/raw/accepted_2007_to_2018Q4.csv",
        "data/processed/loan_data_filtered.csv"
    )
