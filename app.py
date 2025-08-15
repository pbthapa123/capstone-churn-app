from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and expected columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    # Pass columns to HTML so form builds automatically
    return render_template('index.html', columns=model_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Read form data into dictionary
    features = {}
    for col in model_columns:
        val = request.form.get(col)
        try:
            features[col] = float(val)
        except ValueError:
            features[col] = 0.0  # fallback if input missing

    # Create DataFrame for prediction
    input_df = pd.DataFrame([features])

    # Predict
    prediction = model.predict(input_df)[0]
    output = "Customer will Churn" if prediction == 1 else "Customer will Stay"

    return render_template('index.html', columns=model_columns, prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
