from flask import Flask, render_template, request, jsonify
from load_data import load_dataset, dataset_summary
from eda import generate_charts, eda_stats
from outlier_fix import iqr_report
from linear_regression import train_yield
from logistic_regression import train_disease
from tree_based import train_tree

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", data=dataset_summary())

@app.route("/eda")
def eda():
    generate_charts()
    return render_template("eda.html", s=eda_stats())

@app.route("/preprocessing")
def preprocessing():
    df = load_dataset()
    return render_template(
        "preprocessing.html",
        rows=len(df),
        cols=len(df.columns),
        missing=int(df.isna().sum().sum()),
        dup=int(df.duplicated().sum()),
        out=iqr_report(df, "Yield")
    )

@app.route("/yield")
def yield_page():
    return render_template("yield.html")

@app.route("/api/yield")
def yield_api():
    try:
        return jsonify(train_yield(request.args.get("method", "linear")))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/disease")
def disease_page():
    return render_template("disease.html")

@app.route("/api/disease")
def disease_api():
    try:
        return jsonify(train_disease(request.args.get("method", "logistic")))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/trees")
def trees():
    return render_template("trees.html")

@app.route("/api/tree")
def tree_api():
    try:
        return jsonify(train_tree(request.args.get("model", "random_forest")))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/api/predict")
def predict_api():
    try:
        rain = float(request.args["rain"])
        temp = float(request.args["temp"])
        hum = float(request.args["hum"])
        moist = float(request.args["moist"])
        n = float(request.args["n"])
        pest = float(request.args["pest"])
        sev = float(request.args["sev"])
        ndvi = float(request.args["ndvi"])
        irrigation = request.args.get("irrigation", "Yes")
        y = max(.3, round(1.4 + .002 * rain + .028 * moist + .004 * n + .9 * ndvi + (.45 if irrigation == "Yes" else 0) - .008 * pest - .012 * sev - .014 * abs(temp - 27), 2))
        score = .025 * hum + .02 * moist + .035 * pest + .045 * sev
        risk = "High" if score >= 7.5 else ("Medium" if score >= 5 else "Low")
        rec = {
            "High": "Inspect immediately and consult a local agriculture expert for disease control.",
            "Medium": "Monitor the field frequently and maintain balanced irrigation and nutrition.",
            "Low": "Continue routine scouting and normal crop management."
        }[risk]
        return jsonify(yield_ton_ha=y, disease_risk=risk, recommendation=rec)
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    print("Starting CropCast AI on http://127.0.0.1:5000 ...")
    app.run(debug=True)