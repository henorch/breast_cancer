from flask import Flask, request, jsonify
import joblib
import numpy as np


#loading the model 
model = joblib.load("breast_cancer.pkl")


app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to Breast Cancer Machine Learning Home"


@app.route("/predict", method="POST")
def prediction():
    data = 


if __name__ == "__main__":
    app.run(debug=True)
