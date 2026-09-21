import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_cropcast_models(file_path='dataset/crop_yield.csv'):
    # 1. Load Data
    df = pd.read_csv(file_path).dropna()

    # 2. Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # 3. Split Features and Target
    target_col = 'Yield' if 'Yield' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Apply Scalarisation (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Evaluate Linear Regression (Baseline Model)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, lr_preds)
    lr_mse = mean_squared_error(y_test, lr_preds)

    # 6. Evaluate Random Forest Regressor (Core Model)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mse = mean_squared_error(y_test, rf_preds)

    # 7. Output Results for Teacher Review / Terminal Log
    print("=" * 45)
    print("   CROPCAST - MODEL EVALUATION & COMPARISON")
    print("=" * 45)
    print(f"Linear Regression -> R2 Score: {lr_r2:.4f} | MSE: {lr_mse:.4f}")
    print(f"Random Forest     -> R2 Score: {rf_r2:.4f} | MSE: {rf_mse:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    evaluate_cropcast_models()