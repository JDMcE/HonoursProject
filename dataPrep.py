#! /usr/bin/env python

import numpy as np
import pandas as pd
import re
from urllib.parse import unquote
import bleach


def extract_features(payload, label=""):
    """ 
    Accepts payload as string
    Returns a dictionary of features
    """

    urlDecodedPayload = unquote(payload)

    features = {
        "input size": 0,
        "alert": 0,
        "script": 0,
        "onerror": 0,
        "confirm": 0,
        "img": 0,
        "onload": 0,
        "eval": 0,
        "prompt": 0,
        "src": 0,
        "href": 0,
        "javascript": 0,
        "window": 0,
        "fromcharcode": 0,
        "document": 0,
        "onmouseover": 0,
        "cookie": 0,
        "domain": 0,
        "onfocus": 0,
        "iframe": 0,
        "onclick": 0,
        "singleQuoteMark": 0,
        "doubleQuoteMark": 0,
        "leftBracket": 0,
        "rightBracket": 0,
        "leftAngleBracket": 0,
        "rightAngleBracket": 0,
        "backslant": 0,
        "coma": 0,
        "plus": 0,
        "colon": 0,
        "http": 0,
        "file": 0
        }

    # Check if seperate label has been supplied
    if label:
        features["label"] = label

    
    # Check input length (pre URL decoding)
    features["input size"] += len(payload)

    # Check for single character features
    for c in urlDecodedPayload:
        if c == "\'":
            features["singleQuoteMark"] += 1
        elif c == "\"":
            features["doubleQuoteMark"] += 1
        elif c == "(":
            features["leftBracket"] += 1
        elif c == ")":
            features["rightBracket"] += 1
        elif c == "<":
            features["leftAngleBracket"] += 1
        elif c == ">":
            features["rightAngleBracket"] += 1
        elif c == "\\":
            features["backslant"] +=1
        elif c == ",":
            features["coma"] +=1
        elif c == "+":
            features["plus"] +=1
        elif c == ":":
            features["colon"] +=1

    # Check for keywords
    for key in features:
        count = re.findall(key, urlDecodedPayload, re.IGNORECASE)
        if len(count) > 0:
            features[key] += len(count)


    return features


def extract_dataset(raw_dataset, filename):
    """ 
    Accepts raw_dataset as pandas dataframe with "payload" and "label" column
    Saves extracted dataset as csv
    """
    dataSet = []

    print("Extracting features: ")
    for i, row in raw_dataset.iterrows():
        feat = extract_features(row[0])
        feat["label"] = row[1]
        dataSet.append(list(feat.values()))
        #print(i)
    
    headers = list(feat.keys())

    try:
        dataSet = np.array(dataSet)
        df = pd.DataFrame(dataSet)
        df.to_csv(filename, header = headers, index = False)
    except Exception as e:
        print("Unable to save csv: " + str(e))


def sanitize(payload):
    return bleach.clean(payload)
    




if __name__ == "__main__":
    rawData  = pd.read_csv("data/testing/smallxss.csv")
    # test = extract_features(rawData["payloads"][0])
    # print(test)
    #rawData  = pd.read_csv("data/trainingData.csv")
    #test = extract_dataset(rawData, "data/extracted_TrainingData.csv")
    output = sanitize(rawData["payloads"][0])
    print(output)
