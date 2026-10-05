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
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  Computing 5-fold CV for {name}...")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        results[name] = {
            'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CV_R2_mean': cv_scores.mean()
        }
        print(f"  {name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}, CV_R2={cv_scores.mean():.3f}")

        joblib.dump(model, f'models/{name}.pkl')
        print(f"  Saved models/{name}.pkl")

    joblib.dump(list(X.columns), 'models/feature_columns.pkl')
    print(f"\nSaved models/feature_columns.pkl ({len(X.columns)} features)")

    results_df = pd.DataFrame(results).T
    results_df.to_csv('reports/model_comparison.csv')
    print("\nModel Comparison Table:")
    print(results_df.to_string())
    print("\nSaved to reports/model_comparison.csv")

    return results

if __name__ == "__main__":
    train_and_evaluate()
