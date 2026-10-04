from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocess, FEATURES


def train_disease(method="logistic"):
    df = preprocess()
    X = df[FEATURES]
    y = df.Disease_Risk

    a, b, c, d = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    sc = StandardScaler()
    a = sc.fit_transform(a)
    b = sc.transform(b)

    if method == "ridge":
        model = LogisticRegression(penalty="l2", C=1, max_iter=1500)
        name = "Logistic Regression (L2)"
    elif method == "lasso":
        # Changed solver from 'liblinear' to 'saga' to support multi-class L1 penalty
        model = LogisticRegression(penalty="l1", solver="saga", C=1, max_iter=1500)
        name = "Logistic Regression (L1)"
    else:
        model = LogisticRegression(penalty=None, max_iter=1500)
        name = "Logistic Regression"

    model.fit(a, c)
    p = model.predict(b)

    return {
        "model": name,
        "accuracy": round(accuracy_score(d, p), 4),
        "precision": round(precision_score(d, p, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(d, p, average="weighted", zero_division=0), 4),
        "f1": round(f1_score(d, p, average="weighted", zero_division=0), 4)
    }