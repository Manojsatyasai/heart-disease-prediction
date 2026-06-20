from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset
df = pd.read_csv("../Heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = [
        request.form['age'],
        request.form['sex'],
        request.form['cp'],
        request.form['trestbps'],
        request.form['chol'],
        request.form['fbs'],
        request.form['restecg'],
        request.form['thalach'],
        request.form['exang'],
        request.form['oldpeak'],
        request.form['slope'],
        request.form['ca'],
        request.form['thal']
    ]

    data = [float(x) for x in data]

    sample = pd.DataFrame([data], columns=X.columns)
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    result = "High Risk" if prediction[0] == 1 else "Low Risk"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)