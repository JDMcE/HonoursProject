#! /usr/bin/env python

#Combines datasets into one labeled dataset

import csv


newxss = []
with open("xssed.csv", encoding="utf8") as csvFile:
	reader = csv.reader(csvFile) #, dialect="excel"
	for i, row in enumerate(reader):
		newxss.append(row)

newnomal = []
with open("dmzo_nomal.csv", encoding="utf8") as csvFile:
	reader = csv.reader(csvFile) #, dialect="excel"
	for i, row in enumerate(reader):
		newnomal.append(row)

# Append labels
for i in newxss:
	i.append("xss")

for i in newnomal:
	i.append("nomal")


dataLabels = [["payload", "label"]]
dataSet = dataLabels + newxss + newnomal

with open("trainingData.csv", "w", newline="") as f:
	writer = csv.writer(f)
	writer.writerows(dataSet)
