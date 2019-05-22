#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:03:08 2019

@author: hkyeremateng-boateng
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import csv
from ARPSimulator import ARPSimulator as arp
import sklearn.svm as svm

numberOfSamples = 5000
N = numberOfSamples
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length
lambda1 = 0.99;
lambda2 = 0.99;
wLen = 5*dLen;
N_test = dLen*N//2; #test data length
N = N_train+N_test; #total data length
sample_len = 10
max_iteration = 250000
fSteps = dLen; #tracks number of future steps to predict
typeOfOperation = 2 #if 1 then running just one lambda value; 2 running a performance analysis of a range of lambda values

def load_data(fileName):
    data = pd.read_csv(fileName, header=None,sep='\n')
    data = data[data.columns[0:]].values
    return data.transpose()

def plotSVM(totalPwrLvl, prediction,sampleSize):
    dLen = 100 #length of the energy detector
    
    score = prediction[1,:]
    
    totalPwr_score = np.zeros((N));
    prediction_score = np.zeros((N_test-wLen));
    
    
    totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
    
    prediction_score = 10*np.log10(np.abs(score))-30
    subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
    
    plt.figure(figsize=(20,10))
    
    plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
    plt.title('One-step ahead prediction')
    plt.legend(['Input Signal','Prediction'])
    plt.xlabel('Samples')
    plt.ylabel('Magnitude (dBm)')
    plt.show()

def calculateAccuracy(score, powerLvl,sampleSize):
    
    totalPwr = 10*np.log10(np.abs(np.real(powerLvl[0,N_train+wLen:N])))-30
    prediction = 10*np.log10(np.abs(score))-30
    rmse = mean_squared_error(prediction,totalPwr)
    return rmse;  
  
def saveARPData(data):
    with open("svmDataSet_01.csv", 'w') as arp_dataset:
        wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerows([[lst] for  lst in data])

def chooseBestPredictionScore(prediction,formattedData):
    errorScore = 999999 #Initial error score
    for i in np.arange(0,sample_len):
        score = prediction[i,:]
        predictedErrScore = calculateAccuracy(score, formattedData, numberOfSamples)
        if predictedErrScore < errorScore:
            errorScore = predictedErrScore;
    return errorScore
    
    
def generatereqByLambda(lambda1, lambda2,numberOfSamples):
    print("Lambda 1",lambda1,"Lambda 2",lambda2, "Sample Size",numberOfSamples,"Max Iteration",max_iteration)
    lambda_range=0.01
    lambda_length = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda range
    color_range = np.zeros(lambda_length*lambda_length)

    a = 0;
    
    x_range=np.zeros(lambda_length*lambda_length)
    y_range=np.zeros(lambda_length*lambda_length)
    testData = np.zeros((wLen,N_test-wLen));
    fig = plt.figure(figsize=(40,30))
    model_file="models/linearsvr_155kitr_5000s_lambda9_v2.sav"
    model = pickle.load(open(model_file,'rb'));
    for i in np.arange(lambda_range,lambda1,lambda_range):
        for j in np.arange(lambda_range,lambda2, lambda_range):
            
            totalPwrLvl = arp.generateFreqEnergy(arp,i,j,numberOfSamples)
            powerData = pd.DataFrame(np.array(totalPwrLvl).reshape(-1,1),totalPwrLvl)
            powerData = powerData[powerData.columns[0:]].values
            powerLvlLambda = powerData.transpose();
            for b in np.arange(N_test-wLen):
                for t in  np.arange(wLen):
                    testData.itemset((t,b),np.real(powerLvlLambda[0,N_train+t+b]));            
            nData = testData;

            prediction = np.zeros((fSteps, N_test-wLen));
            for a in np.arange(0,sample_len):
                prediction[a] = model.predict(nData.transpose());
                #nData = np.concatenate((testData[1:wLen:,],prediction[i:i+1,:]));
                nData = np.concatenate((testData[a+1:wLen:,],prediction[0:a+1,:]));
                nData = np.concatenate((prediction[0:a+1,:],testData[a+1:wLen:,]));# New prediction concatenation
            errorScore = chooseBestPredictionScore(prediction,powerLvlLambda)

            print("Lambda 1",i,"Lambda 2",j, "Sample Size",numberOfSamples,"Error  ",errorScore)
            color_range.itemset(a,errorScore/100)
            x_range.itemset(a,i);
            y_range.itemset(a,j);
            a=a+1;

    plt.scatter(x_range,y_range,c=color_range, s=errorScore*20,alpha=0.55)
    plt.colorbar()
    fig.savefig("lambda0.9range0.01_500samples_05102019.png")


def predictor(formattedData):
    trainData = np.zeros((wLen,N_train-wLen));
    testData = np.zeros((wLen,N_test-wLen));
    trainLbl = np.zeros((1,N_train-wLen));
    
    trainFeatures = 0;
    for i in np.arange(wLen,N_train-1):
        trainLbl.itemset(trainFeatures,formattedData[0,i]);
        trainFeatures = trainFeatures +1;
       
    for s in np.arange(trainFeatures):
        for k in  np.arange(wLen-1):
            trainData.itemset((k,s),np.real(formattedData[0,s+k]))
    for b in np.arange(N_test-wLen):
        for t in  np.arange(wLen):
            testData.itemset((t,b),np.real(formattedData[0,N_train+t+b]));    

    label_reshape = trainLbl.reshape((N_train-wLen))
    train_reshape = trainData.transpose()
    nData = testData;

    clf = svm.LinearSVR(epsilon=10e-89, C=100000000, max_iter=250000, dual=True,random_state=0,loss='squared_epsilon_insensitive',tol=10e-6).fit(train_reshape,label_reshape)
    fSteps = dLen; #tracks number of future steps to predict

    predicted = np.zeros((fSteps, N_test-wLen));

    #nData = testData;
    for i in np.arange(0,sample_len):
        predicted[i] = clf.predict(nData.transpose());
        nData = np.concatenate((predicted[0:i+1,:],testData[i+1:wLen:,]));
        #nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));
    
    predSet = np.zeros((fSteps, N_test-wLen));
    setCnt = 0;

    for i in np.arange(0,N_test-wLen):
        predSet[setCnt,i-setCnt:i-setCnt+fSteps] = predicted[:,i].transpose()
        if (setCnt+1)==fSteps:
            #obsSample[i-setCnt] = 1;
            setCnt = 0;
        else:
            setCnt = setCnt + 1;
    return predicted
if typeOfOperation == 2:
    generatereqByLambda(lambda1,lambda2,numberOfSamples)
else:
    totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
    saveARPData(totalPwrLvl)
    trainData = np.zeros((wLen,N_train-wLen));
    testData = np.zeros((wLen,N_test-wLen));
    trainLbl = np.zeros((1,N_train-wLen));
    formattedData = load_data("svmDataSet_01.csv")
    
    trainFeatures = 0;
    for i in np.arange(wLen,N_train-1):
        trainLbl.itemset(trainFeatures,formattedData[0,i]);
        trainFeatures = trainFeatures +1;
       
    for s in np.arange(trainFeatures):
        for k in  np.arange(wLen-1):
            trainData.itemset((k,s),np.real(formattedData[0,s+k]))
    for b in np.arange(N_test-wLen):
        for t in  np.arange(wLen):
            testData.itemset((t,b),np.real(formattedData[0,N_train+t+b]));
            
    label_reshape = trainLbl.reshape((N_train-wLen))
    train_reshape = trainData.transpose()
    
    model_file="models/linearsvr_100kitr_3000s_lambda99_v1.sav"
    model = pickle.load(open(model_file,'rb'));
    nData = testData;
    fSteps = dLen; #tracks number of future steps to predict
    prediction = np.zeros((fSteps, N_test-wLen));
    for i in np.arange(0,sample_len):
        prediction[i] = model.predict(nData.transpose());
        #nData = np.concatenate((testData[1:wLen:,],prediction[i:i+1,:]));
        nData = np.concatenate((prediction[0:i+1,:],testData[i+1:wLen:,]));# New prediction concatenation
    
    for i in np.arange(0,sample_len):
        score = prediction[i,:]
        accuracy = calculateAccuracy(score, formattedData, numberOfSamples)
        print("Error Rate: ",accuracy);
  
    


