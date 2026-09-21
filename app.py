from flask import Flask, render_template, request
import numpy as np
import joblib
from model_training import train_models

app = Flask(__name__)

# Train and load model upon startup (or load a saved model using joblib)
model = train_models()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs (ensure your HTML form matches these names)
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]

        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"Estimated Crop Yield: {output} tons/hectare")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)