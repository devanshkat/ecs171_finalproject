import sys
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Allows for calling models from Models folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'Models'))

app = Flask(__name__)
model = joblib.load('../Models/poly_regression.joblib')
scaler = joblib.load('../Models/scaler.joblib')
poly = joblib.load('../Models/poly.joblib')
# df = pd.read_csv('../Models/instagram_usage_lifestyle.csv')
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        age = float(request.form.get("age"))
        minutes = float(request.form.get("minutes"))
        happiness = float(request.form.get("happiness"))
        df = pd.read_csv('../Models/instagram_usage_lifestyle.csv')
        df = df[(df['age'] == age) & (df['daily_active_minutes_instagram']== minutes) & (df['self_reported_happiness'] == happiness)]
        print(len(df))
        print(df['perceived_stress_score'])
        processed_data = scaler.transform(poly.transform([[age, minutes, happiness]]))
        prediction = model.predict(processed_data)[0].round()
        print(model.predict(processed_data), prediction)
        if prediction < 1:
            prediction = 1
        elif prediction > 10:
            prediction = 10
        return render_template("result.html", prediction=prediction)
    else:
        return "Invalid request method"

if __name__ == "__main__":
    app.run(debug=True)
