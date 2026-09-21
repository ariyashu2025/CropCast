from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from preprocessing import load_and_preprocess_data


def train_random_forest_yield(file_path='dataset/crop_yield.csv'):
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)

    # Random Forest Regressor for Yield Prediction
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    predictions = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"Random Forest Regressor MSE: {mse:.4f}")
    return rf_regressor, feature_names