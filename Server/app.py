import sys
import os
import joblib
from flask import Flask, render_template, request

# Allows for calling models from Models folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'Models'))

app = Flask(__name__)
linear_model = joblib.load('../Models/linear_regression.joblib')
scaler = joblib.load('../Models/scaler_linear.joblib')
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        age = float(request.form.get("age"))
        minutes = float(request.form.get("minutes"))
        happiness = float(request.form.get("happiness"))
        processed_data = scaler.transform([[age, minutes, happiness]])
        prediction = linear_model.predict(processed_data)
        return render_template("result.html", prediction=prediction)
    else:
        return "Invalid request method"

if __name__ == "__main__":
    app.run(debug=True)
