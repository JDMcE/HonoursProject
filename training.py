#! /usr/bin/env python

import numpy as np
import pandas as pd
import pickle as pk
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import learning_curve, ShuffleSplit



# read in raw data
rawData = pd.read_csv("data/extracted_TrainingData.csv")
data = rawData.to_numpy()

# Separate features and labels
feature_names = list(rawData.columns)
features = data[:,0:-1]
labels = data[:,-1]

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
print(encoder.classes_)
# encoder.inverse_transform()
# with open("labelEncoder.pkl",'wb') as le: 
#     pk.dump(encoder, le)

# standardize data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into train/test
train_feat, test_feat, train_label, test_label = train_test_split(features, labels, test_size=0.2)

# PCA
doPCA = False
if doPCA:
    pca = PCA(32)
    pca.fit(train_feat)
    train_feat = pca.transform(train_feat)
    test_feat = pca.transform(test_feat)
    print(f"Reduced to: {len(pca.components_)}")
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    print(pca.singular_values_)

# Train all models
RandomForestClassifier_model = RandomForestClassifier(verbose=True).fit(train_feat, train_label)
LogisticRegression_model = LogisticRegression(max_iter=500).fit(train_feat, train_label)
Perceptron_model = Perceptron().fit(train_feat, train_label)
SVC_model = SVC().fit(train_feat, train_label)
GaussianNB_model = GaussianNB().fit(train_feat, train_label)
DecisionTreeClassifier_model = DecisionTreeClassifier().fit(train_feat, train_label)
MLPClassifier_model = MLPClassifier(
    hidden_layer_sizes=(32,32, 32),
    activation='relu', 
    solver='adam',
    learning_rate='constant',
    verbose=True).fit(train_feat, train_label)

all_models = [RandomForestClassifier_model, LogisticRegression_model, Perceptron_model, SVC_model, GaussianNB_model, DecisionTreeClassifier_model, MLPClassifier_model]
top3_models = [RandomForestClassifier_model, DecisionTreeClassifier_model, MLPClassifier_model]

plt.plot(MLPClassifier_model.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

ax = plt.gca()
results = []
for model in top3_models:
    testLabelPredictions = model.predict(test_feat)
    #print(testLabelPredictions)
    name = type(model).__name__

    print(f"{name}:")
    trainAcc = model.score(train_feat, train_label)
    print(f"Training accuracy: {trainAcc*100}")
    accScore = accuracy_score(test_label, testLabelPredictions)
    print(f"accuracy: {accScore*100}")
    precScore = precision_score(test_label, testLabelPredictions)
    print(f"Precision score: {precScore*100}")
    recallScore = recall_score(test_label, testLabelPredictions)
    print(f'Recall: {recallScore*100}')
    f1Score = f1_score(test_label, testLabelPredictions)
    print(f"f1 Score: {f1Score} \n")
    cr = classification_report(test_label, testLabelPredictions)
    print(cr)
    cm = confusion_matrix(test_label, testLabelPredictions)
    print(f"Confulsion Matrix: {cm} \n")

    # add model results
    results.append([name, accScore, precScore, recallScore, f1Score,])# pca.components_, pca.explained_variance_ratio_])

    # plot RocCurve
    disp = RocCurveDisplay.from_estimator(model, test_feat, test_label)
    disp.plot(ax=ax, alpha=0.8)

    # save model
    date = datetime.now().strftime("%Y_%m_%d-%I-%M")
    filename = f"trained_models/{name}_{date}.pkl"
    with open(filename,'wb') as file: 
        pk.dump(model, file)

plt.show()

headers = ["model", "accScore", "precScore", "recallScore", "f1Score", "PCAcomp", "PCAexplained_variance_ratio_"]
results = pd.DataFrame(results)
results.to_csv(f"data/results/results_{date}.csv", header=headers, index=False)
print(results)


