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

def printOfInformation():
    zero10 = 0
    ten20 = 0
    twenty30 = 0
    thirty40 = 0
    forty50 = 0
    fifty60 = 0
    sixty70 = 0
    seventy80 = 0
    eighty90 = 0
    ninedy100 = 0

    x = 0

    #Count the percentages
    while(x < 9912):
        #print(labels[x])
        if(labels[x] >= 90):
            ninedy100 = ninedy100 + 1
        elif(labels[x] >= 80):
            eighty90 = eighty90 + 1
        elif(labels[x] >= 70):
            seventy80 = seventy80 + 1
        elif(labels[x] >= 60):
            sixty70 =sixty70 + 1
        elif(labels[x] >= 50):
            fifty60 = fifty60 + 1
        elif( labels[x] >= 40):
            forty50 = forty50  + 1
        elif(labels[x] >= 30):
            thirty40 = thirty40 + 1
        elif(labels[x] >= 20):
            twenty30 = twenty30 + 1
        elif(labels[x] >= 10):
            ten20 = ten20 + 1
        else:
            zero10 = zero10 + 1
        x = x + 1

    # To double check that no data point is counter twice
    #total = zero10 + ten20 + twenty30 + thirty40 + forty50 + fifty60 + sixty70 + seventy80 + eighty90 + ninedy100

    #Check percentage of each x compared to the labels
    zero10 = 0
    ten20 = 0
    twenty30 = 0
    thirty40 = 0
    forty50 = 0
    fifty60 = 0
    sixty70 = 0
    seventy80 = 0
    eighty90 = 0
    ninedy100 = 0
    x = 0
    while(x < 9912):
        if(blur[x] == 1):
            if(labels[x] >= 90):
                ninedy100 = ninedy100 + 1
            elif(labels[x] >= 80):
                eighty90 = eighty90 + 1
            elif(labels[x] >= 70):
                seventy80 = seventy80 + 1
            elif(labels[x] >= 60):
                sixty70 =sixty70 + 1
            elif(labels[x] >= 50):
                fifty60 = fifty60 + 1
            elif( labels[x] >= 40):
                forty50 = forty50  + 1
            elif(labels[x] >= 30):
                thirty40 = thirty40 + 1
            elif(labels[x] >= 20):
                twenty30 = twenty30 + 1
            elif(labels[x] >= 10):
                ten20 = ten20 + 1
            else:
                zero10 = zero10 + 1

        x = x + 1
    total = zero10 + ten20 + twenty30 + thirty40 + forty50 + fifty60 + sixty70 + seventy80 + eighty90 + ninedy100
    print(zero10)
    print(ten20)
    print(twenty30)
    print(thirty40)
    print(forty50)
    print(fifty60)
    print(sixty70)
    print(seventy80)
    print(eighty90)
    print(ninedy100)
    print(total)

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

features = []
features.append(focus)
features.append(eyes)
features.append(face)
features.append(near)
features.append(action)
features.append(accessory)
features.append(group)
features.append(collage)
features.append(human)
features.append(occlussion)
features.append(info)
features.append(blur)

npFeatures = np.asarray(features)
npFeaturesT = np.asarray(features)
print(npFeatures.shape)
print(labels.shape)
npFeatures = npFeatures.transpose()

f = []
f.append(face)
f.append(group)
f.append(accessory)

f = np.asarray(f)
f = f.transpose()

model = LinearRegression(positive = True)
#model.fit(npFeatures, labels)
model.fit(npFeatures, labels)

importance = model.coef_

for i,v in enumerate(importance):
    print('Feature %0d, Score %0.5f' % (i,v))

featuresForModel = []
featuresForModel.append(group)
featuresForModel.append(accessory)
featuresForModel.append(face)
featuresForModel.append(human)
featuresForModel.append(near)
featuresForModel = np.asarray(featuresForModel)

featuresForModel = featuresForModel.transpose()
#xtrain, xtest, ytrain, ytest = train_test_split(npFeatures, labels)
xtrain, xtest, ytrain, ytest = train_test_split(featuresForModel, labels)
model = LinearRegression(positive = True)
#model.fit(npFeatures, labels)
model.fit(xtrain, ytrain)
prediction = model.predict(xtest)

accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)

accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Linear Regression")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)


regressionModel = RandomForestRegressor(criterion = "mse")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Random Forest with squared-error")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = RandomForestRegressor(criterion = "mae")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Random Forest with absolute_error")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = RandomForestRegressor(criterion = "poisson", max_depth = 2)
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Random Forest with poisson")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)
fig = plt.figure(figsize=(40, 40))
plot_tree(regressionModel.estimators_[0], filled = True, rounded = True)
fig.savefig("RandomForest.png");

regressionModel = SVR(kernel = 'rbf')
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("SVR with rbf")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = SVR(kernel = 'linear')
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)

accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("SVR with linear")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = SVR(kernel = 'poly')
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)

accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("SVR with poly")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = SVR(kernel = 'sigmoid')
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("SVR with sigmoid")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)


regressionModel = DecisionTreeRegressor(criterion = "friedman_mse")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Decision Tree Regression with friedman_mse")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = DecisionTreeRegressor(criterion = "mse")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Decision Tree Regression with squared_error")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = DecisionTreeRegressor(criterion = "mae")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Decision Tree Regression with absolute_error")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)

regressionModel = DecisionTreeRegressor(criterion = "poisson")
regressionModel.fit(xtrain, ytrain)
prediction = regressionModel.predict(xtest)
accuracyMeanSquared = metrics.mean_squared_error(ytest, prediction)
accuracyVariance = metrics.explained_variance_score(ytest, prediction)
accuracyR2Score = metrics.r2_score(ytest,prediction)
print("Decision Tree Regression with poisson")
print(accuracyMeanSquared)
print(accuracyVariance)
print(accuracyR2Score)
