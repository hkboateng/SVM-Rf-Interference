#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:58:22 2019

@author: Hubert Kyeremateng-Boateng
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ARPSimulator import ARPSimulator as arp
import csv
from sklearn.metrics import mean_squared_error
import pickle

numberOfSamples = 1000
lambda1 = 0.009;
lambda2 = 0.001;
dLen = 100 #length of the energy detector
wLen = 5*dLen; 


def generateTestDataSet(start,end,dataSource,testData,trainData):
    for s in np.arange(start):
        for k in  np.arange(end):
            testData.itemset((k,s),np.real(dataSource[0,trainData+s+k]))
    return testData;

def calculateAccuracy(score, powerLvl,sampleSize):
    dLen = 100 #length of the energy detector
    N = sampleSize
    N_train = dLen*N//2-dLen+1; #training data length - 14901
    #Training label ground truth/target
    wLen = 5*dLen; 
    N_test = dLen*N//2; #test data length - 15000
    N = N_train+N_test;
    
    totalPwr = 10*np.log10(np.abs(np.real(powerLvl[0,N_train+wLen:N])))-30
    prediction = 10*np.log10(np.abs(score))-30
    rmse = mean_squared_error(prediction,totalPwr)
    return rmse;
def saveARPData(data):
    with open("svmDataSet.csv", 'w') as arp_dataset:
        wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerows([[lst] for  lst in data])

def load_data(fileName):
    data = pd.read_csv(fileName, header=None,sep='\n')
    data = data[data.columns[0:]].values
    return data.transpose()
def predictor(filename,powerAvgLvl):
    dLen = 100 #length of the energy detector
    N = numberOfSamples
    N_train = dLen*N//2-dLen+1; #training data length - 14901
    #Training label ground truth/target
    wLen = 5*dLen; 
    N_test = dLen*N//2; #test data length - 15000
    N = N_train+N_test;
    
    testData = np.zeros((wLen,N_test-wLen));
    for b in np.arange(N_test-wLen):
        for t in  np.arange(wLen):
            testData.itemset((t,b),np.real(powerAvgLvl[0,N_train+t+b]));  

    svm_model = pickle.load(open(filename, 'rb'))

    fSteps = dLen
    predicted = np.zeros((fSteps, N_test-wLen));
 
    nData = testData;
    for i in np.arange(0,prediction_width):
        predicted[i] = svm_model.predict(nData.transpose());
        #nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));
        nData = np.concatenate((predicted[0:i+1,:],testData[i+1:wLen:,]));
        #nData = np.concatenate((testData[i+1:wLen:,],predicted[0:i+1,:]));

    
    return predicted
    pass

def plotSVM(totalPwrLvl, prediction,sampleSize):
    dLen = 100 #length of the energy detector
    N = sampleSize
    N_train = dLen*N//2-dLen+1; #training data length - 14901
    #Training label ground truth/target
    wLen = 5*dLen; 
    N_test = dLen*N//2; #test data length - 15000
    N = N_train+N_test;
    
    
    totalPwr_score = np.zeros((N));
    prediction_score = np.zeros((N_test-wLen));
    
    
    totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
    
    prediction_score = 10*np.log10(np.abs(prediction))-30
    subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
    
    plt.figure(figsize=(10,10))
    
    plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
    plt.title('one-step ahead prediction')
    plt.legend(['Input Signal','Prediction'])
    plt.xlabel('Samples')
    plt.ylabel('Magnitude (dBm)')
    plt.show()
 
prediction_width = 50 
totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
saveARPData(totalPwrLvl)
totalPwrLvl = load_data("svmDataSet.csv")
filename="linearSVRModel_500s_150000itr.sav"
prediction = predictor(filename,totalPwrLvl)

score_accuracy =1000
totalPwr_score = np.zeros((numberOfSamples));
for i in range(prediction_width):
    accuracy = calculateAccuracy(prediction[i,:], totalPwrLvl, numberOfSamples)
    if accuracy < score_accuracy:
        score_accuracy = accuracy
        totalPwr_score = prediction[i,:]
    print("Error Rate: ",accuracy);
plotSVM(totalPwrLvl,totalPwr_score,numberOfSamples)
print("Error Rate: ",score_accuracy);
