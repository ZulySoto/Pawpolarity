import  csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def readData(filename):
    file = open(filename)
    reader = csv.reader(file)

    header = next(reader)
    trainData = []

    for x in reader:
        trainData.append(x)

    return header, trainData

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


header, trainData = readData('./petfinder-pawpularity-score/train.csv')
fileNames, focus, eyes, face, near, action, accessory, group, collage, human, occlussion, info, blur, labels = splitData(trainData)

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

featuresForModel = []
featuresForModel.append(group)
featuresForModel.append(accessory)
featuresForModel.append(human)
featuresForModel.append(near)
featuresForModel.append(face)
featuresForModel = np.asarray(featuresForModel)

featuresForModel = featuresForModel.transpose()
xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)

#Experiment 13: Random Forest with group, accessory, human, near, face
regressionModel = RandomForestRegressor(criterion = "poisson")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Experiment 13: Random Forest with group, accessory, human")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

#Experiment 14: Random Forest with group, accessory, human, near, face five times
count = 0
accuracyR2Score = 0
accuracyVariance = 0
accuracyMeanSquared = 0
regressionModel = RandomForestRegressor(criterion = "poisson")
while(count < 5):
    xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)
    regressionModel.fit(xtrain, ytrain)
    prediction = regressionModel.predict(xtest)
    accuracyMeanSquared = accuracyMeanSquared + metrics.mean_squared_error(ytest, prediction)
    accuracyVariance = accuracyVariance + metrics.explained_variance_score(ytest, prediction)
    accuracyR2Score = accuracyR2Score + metrics.r2_score(ytest,prediction)
    count = count + 1

print("Experiment 14: Random Forest with group, accessory, human, near, face five times")
accuracyMeanSquared = accuracyMeanSquared / 5
accuracyVariance = accuracyVariance / 5
accuracyR2Score = accuracyR2Score / 5

print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

#Experiment 15: Random Forest with group, accessory, human, near, face 10 times
count = 0
accuracyR2Score = 0
accuracyVariance = 0
accuracyMeanSquared = 0
regressionModel = RandomForestRegressor(criterion = "poisson")
while(count < 10):
    xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)
    regressionModel.fit(xtrain, ytrain)
    prediction = regressionModel.predict(xtest)
    accuracyMeanSquared = accuracyMeanSquared + metrics.mean_squared_error(ytest, prediction)
    accuracyVariance = accuracyVariance + metrics.explained_variance_score(ytest, prediction)
    accuracyR2Score = accuracyR2Score + metrics.r2_score(ytest,prediction)
    count = count + 1

accuracyMeanSquared = accuracyMeanSquared / 10
accuracyVariance = accuracyVariance / 10
accuracyR2Score = accuracyR2Score / 10
print("Experiment 15: Random Forest with group, accessory, human, near, face 10 times")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)


featuresForModel = []
featuresForModel.append(group)
featuresForModel.append(accessory)
featuresForModel.append(face)
featuresForModel.append(human)
featuresForModel = np.asarray(featuresForModel)

featuresForModel = featuresForModel.transpose()
xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)

#Experiment 16: Random Forest with group, accessory, face, human
regressionModel = RandomForestRegressor(criterion = "poisson")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Experiment 16: Random Forest with group, accessory, face, human")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

#Experiment 17: Random Forest with group, accessory, face, human five times
count = 0
accuracyR2Score = 0
accuracyVariance = 0
accuracyMeanSquared = 0
regressionModel = RandomForestRegressor(criterion = "poisson")
while(count < 5):
    xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)
    regressionModel.fit(xtrain, ytrain)
    prediction = regressionModel.predict(xtest)
    accuracyMeanSquared = accuracyMeanSquared + metrics.mean_squared_error(ytest, prediction)
    accuracyVariance = accuracyVariance + metrics.explained_variance_score(ytest, prediction)
    accuracyR2Score = accuracyR2Score + metrics.r2_score(ytest,prediction)
    count = count + 1

print("Experiment 17: Random Forest with group, accessory, face, human five times")
accuracyMeanSquared = accuracyMeanSquared / 5
accuracyVariance = accuracyVariance / 5
accuracyR2Score = accuracyR2Score / 5

print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

#Experiment 18: Random Forest with group, accessory, face, human 10 times
count = 0
accuracyR2Score = 0
accuracyVariance = 0
accuracyMeanSquared = 0
regressionModel = RandomForestRegressor(criterion = "poisson")
while(count < 10):
    xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)
    regressionModel.fit(xtrain, ytrain)
    prediction = regressionModel.predict(xtest)
    accuracyMeanSquared = accuracyMeanSquared + metrics.mean_squared_error(ytest, prediction)
    accuracyVariance = accuracyVariance + metrics.explained_variance_score(ytest, prediction)
    accuracyR2Score = accuracyR2Score + metrics.r2_score(ytest,prediction)
    count = count + 1

accuracyMeanSquared = accuracyMeanSquared / 10
accuracyVariance = accuracyVariance / 10
accuracyR2Score = accuracyR2Score / 10
print("Experiment 18: Random Forest with group, accessory, face, human 10 times")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)










#Plot a plot a tree
#fig = plt.figure(figsize = (15,10))
#plot_tree(regressionModel.estimators_[0], rounded = True, impurity = True, class_names = labels)
#fig.savefig('regression.png')
