from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model and expected columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html', columns=model_columns)

@app.route('/predict', methods=['POST'])
def predict():
    features = {}
    for col in model_columns:
        val = request.form.get(col)
        try:
            features[col] = float(val)
        except (ValueError, TypeError):
            features[col] = 0.0

    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    output = "Customer will Churn" if prediction == 1 else "Customer will Stay"

    return render_template('index.html', columns=model_columns, prediction_text=output)

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug_mode)
