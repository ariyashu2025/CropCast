from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import load_and_preprocess_data


def train_models():
    # Update path to your dataset inside the dataset folder
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data('dataset/crop_yield.csv')

    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_r2 = r2_score(y_test, rf_preds)
    print(f"Random Forest R2 Score: {rf_r2:.4f}")

    return rf_model


if __name__ == "__main__":
    train_models()