import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def compare_models(file_path='dataset/crop_yield.csv'):
    # Load and preprocess
    df = pd.read_csv(file_path).dropna()

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    target_col = 'Yield' if 'Yield' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, lr_preds)
    lr_mse = mean_squared_error(y_test, lr_preds)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mse = mean_squared_error(y_test, rf_preds)

    print("--- Model Performance Comparison ---")
    print(f"Linear Regression -> R2 Score: {lr_r2:.4f} | MSE: {lr_mse:.4f}")
    print(f"Random Forest     -> R2 Score: {rf_r2:.4f} | MSE: {rf_mse:.4f}")

    return {"Linear Regression": lr_r2, "Random Forest": rf_r2}


if __name__ == "__main__":
    compare_models()