import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(file_path):
    """
    Loads dataset, handles missing values, encodes categories,
    and applies Scalarisation (StandardScaler) necessary for gradient-based and distance-based models.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()

    # Categorical encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Splitting features and target
    target_col = 'Yield' if 'Yield' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scalarisation for numerical stability in gradient optimization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns