import  csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.linear_model import BayesianRidge

#Gets Data from file
def readData(filename):
    file = open(filename)
    reader = csv.reader(file)

    header = next(reader)
    trainData = []

    for x in reader:
        trainData.append(x)

    return header, trainData

#Splits the data into distinct lists for each feature and label
def splitData(trainData):
    fileNames = []
    focus = []
    eyes = []
    face = []
    near = []
    action = []
    accessory = []
    group = []
    collage = []
    human = []
    occlussion = []
    info = []
    blur = []
    labels =[]

    x = 0

    for y in trainData:
        fileNames.append(trainData[x][0])
        focus.append(trainData[x][1])
        eyes.append(trainData[x][2])
        face.append(trainData[x][3])
        near.append(trainData[x][4])
        action.append(trainData[x][5])
        accessory.append(trainData[x][6])
        group.append(trainData[x][7])
        collage.append(trainData[x][8])
        human.append(trainData[x][9])
        occlussion.append(trainData[x][10])
        info.append(trainData[x][11])
        blur.append(trainData[x][12])
        labels.append(trainData[x][13])
        x = x + 1
    return fileNames, focus, eyes, face, near, action, accessory, group, collage, human, occlussion, info, blur, labels

#Converts the list into numpy arrays
def numpyArrayCreation(focus, eyes, face, near, action, accessory, group, collage, human, occlussion, info, blur, labels):
    focus = np.array(focus).astype(dtype = float)
    eyes = np.array(eyes).astype(dtype = float)
    face = np.array(face).astype(dtype = float)
    near = np.array(near).astype(dtype = float)
    action = np.array(action).astype(dtype = float)
    accessory = np.array(accessory).astype(dtype = float)
    group = np.array(group).astype(dtype = float)
    collage = np.array(collage).astype(dtype = float)
    human = np.array(human).astype(dtype = float)
    occlussion = np.array(occlussion).astype(dtype = float)
    info = np.array(info).astype(dtype = float)
    blur = np.array(blur).astype(dtype = float)
    labels = np.array(labels).astype(dtype = float)

    return focus, eyes, face, near, action, accessory, group, collage, human, occlussion, info, blur, labels


header, trainData = readData('./petfinder-pawpularity-score/train.csv')
fileNames, focus, eyes, face, near, action, accessory, group, collage, human, occlussion, info, blur, labels = splitData(trainData)
focus, eyes, face, near, action, accessory, group, collage, human, occlussion, info, blur, labels = numpyArrayCreation(focus, eyes, face, near, action, accessory, group, collage, human, occlussion, info, blur, labels)

print("Testing Bayesian on group, accessory, human, near, and face")

featuresForModel = []
featuresForModel.append(group)
featuresForModel.append(accessory)
featuresForModel.append(human)
featuresForModel.append(near)
featuresForModel.append(face)
featuresForModel = np.asarray(featuresForModel)
featuresForModel = featuresForModel.transpose()

xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)

regressionModel = BayesianRidge()
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Bayesian")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

print("Testing Bayesian on group, accessory, human, and face")

featuresForModel = []
featuresForModel.append(group)
featuresForModel.append(accessory)
featuresForModel.append(face)
featuresForModel.append(human)
featuresForModel = np.asarray(featuresForModel)
featuresForModel = featuresForModel.transpose()

xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)

regressionModel = BayesianRidge()
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Bayesian")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)
