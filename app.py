#! /usr/bin/env python

from turtle import st
from  flask import Flask, render_template, request, jsonify, make_response, Markup, url_for
import json
import dataPrep
import pickle as pk
import numpy as np
import pandas as pd



app = Flask(__name__)
port = 8000

@app.route('/')
def root():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    
    req = request.get_json()

    # Process data
    payload = req["payload"]
    features = dataPrep.extract_features(payload)
    X = [list(features.values())]

    # Make prediction
    prediction = model.predict(X)
    predictionString = labelEncoder.inverse_transform(prediction)
    probability = model.predict_proba(X)
    print(f"Prediction: {predictionString}")
    print(f"Probability: {probability}")

    confidence = max(probability[0]) * 100

    if predictionString == "xss":
        mal = "True"
    else:
        mal = "False"

    #sanitize payload
    sanitized = dataPrep.sanitize(req["payload"])
    print(sanitized)

    # Send response
    response_body = {
        "message": "OK",
        "payload": req["payload"],
        "malicious": mal, 
        "confidence": confidence,
        "sanitized": sanitized
    }
    print(response_body)
    with open(responseLog, "a") as rl:
        rl.writelines(json.dumps(response_body))
        rl.write("\n")

    res = make_response(jsonify(response_body), 200)
    return res


if __name__ == "__main__":

    # np settings
    np.set_printoptions(threshold=np.inf)
    pd.set_option('display.max_colwidth', None)


    # Set file for logs
    responseLog = "data/app/responseLog.txt"

    #load models
    file = "trained_models/MLPClassifier_2022_03_13-12-10.pkl"
    with open(file, "rb") as f:
        model = pk.load(f)

    # load Encoder
    le = "trained_models/labelEncoder.pkl"
    with open(le, "rb") as l:
        labelEncoder = pk.load(l)
    
    #model stats
    stats = [model.classes_, model.t_, model.n_features_in_, model.n_layers_, model.intercepts_, model.coefs_]
    # print(f"Classes: {model.classes_}")
    # print(f"Training Samples Seen: {model.t_}")
    # print(f"Number of features seen: {model.n_features_in_}")
    # print(f"Number of layers: {model.n_layers_}")
    # print(f"Coefs(wieghts): {model.coefs_[0]}")
    # print(f"Biases: {model.intercepts_}")

    # dfa = pd.DataFrame(model.coefs_[0])

    # texfile = "data/app/tex.txt"
    # with open(texfile, "w") as tf:
    #     tf.write(dfa.to_latex(index = False, float_format="%.4f"))


    # Record stats for loaded model
    statfile = "data/app/stats.txt"
    with open(statfile, "w") as sf:
        for s in stats:
            sf.writelines(str(s))
            sf.writelines("\n\n")


    #Run App
    app.run(host="localhost", port=port, debug=True)
