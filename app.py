import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# create flask app
app = Flask(__name__)

# load the model
model = pickle.load(open('log_regression.pkl', "rb"))



@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template('index.html', Result='RESULT : Heart disease present'if prediction[0] == 0 else 'RESULT : Heart Disease free')

if __name__ == "__main__":
    app.run(debug=True)