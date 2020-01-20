#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 23:05:16 2019

@author: Hubert Kyeremateng-Boateng
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from ARP import ARPSimulator as arp
import csv
from sklearn.metrics import mean_squared_error
import pickle


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
    with open(dataSource, 'w') as arp_dataset:
        wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerows([[lst] for  lst in data])

def load_data(fileName):
    data = pd.read_csv(fileName, header=None,sep='\n')
    data = data[data.columns[0:]].values
    return data.transpose()

def plotSVM(totalPwrLvl, prediction,sampleSize):
    dLen = 100 #length of the energy detector
    N = sampleSize
    N_train = dLen*N//2-dLen+1; #training data length - 14901
    #Training label ground truth/target
    wLen = 5*dLen; 
    N_test = dLen*N//2; #test data length - 15000
    N = N_train+N_test;
    
    score = prediction[1,:]
    
    totalPwr_score = np.zeros((N));
    prediction_score = np.zeros((N_test-wLen));
    
    
    totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
    
    prediction_score = 10*np.log10(np.abs(score))-30
    subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
    
    plt.figure(figsize=(30,30))
    
    plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
    plt.title('one-step ahead prediction')
    plt.legend(['Input Signal','Prediction'])
    plt.xlabel('Samples')
    plt.ylabel('Magnitude (dBm)')
    plt.show()
   
numberOfSamples = 1000
lambda1 = 0.8;
lambda2 = 0.8;
dLen = 100 #length of the energy detector
wLen = 5*dLen; 
dataSource = "svmDataSet_01.csv"
N = numberOfSamples

N_train = dLen*N//2-dLen+1; #training data length

   
N_test = dLen*N//2; #test data length
N = N_train+N_test; #total data length
sample_len = 5
#input window length
trainLbl = np.zeros((1,N_train-wLen));

totalPwrLvls,totalPwrLvl_cumulants = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
saveARPData(totalPwrLvls)
totalPwrLvl = load_data(dataSource)

counter = 0
for i in np.arange(wLen,N_train-1):
    trainLbl.itemset(counter,totalPwrLvl[0,i]);
    counter= counter+1;

#Traing and test input data
trainData = np.zeros((wLen,N_train-wLen));
testData = np.zeros((wLen,N_test-wLen));
###### totalAvgPwr is one-dimensional array#####
###### trainData in a multi-dimensional array######
for s in np.arange(N_test-wLen):
    for k in  np.arange(wLen):
        testData.itemset((k,s),np.real(totalPwrLvl[0,N_train+s+k]))
for s in np.arange(counter):
    for k in  np.arange(wLen-1):
        trainData.itemset((k,s),np.real(totalPwrLvl[0,s+k]))
                

                
label_reshape = trainLbl.reshape((N_train-wLen))
train_reshape = trainData.transpose()
clf = svm.LinearSVR(epsilon=10e-9, C=10e6,verbose=6,random_state=0,loss='squared_epsilon_insensitive',tol=10e-6,max_iter=105000).fit(train_reshape,label_reshape)
#clf = svm.LinearSVR(epsilon=10e-90, C=10e6, max_iter=10000, dual=True,random_state=0,loss='squared_epsilon_insensitive',tol=10e-5).fit(train_reshape,label_reshape)
fSteps = dLen; #tracks number of future steps to predict
#model_name = 'linearSVRModel_500s_150000itr.sav'
#with open(model_name,'wb') as model_file:
#    pickle.dump(clf,model_file)
    
prediction = np.zeros((fSteps, N_test-wLen));
 
nData = testData;
for i in np.arange(0,sample_len):
    prediction[i] = clf.predict(nData.transpose());
    nData = np.concatenate((testData[1:wLen:,],prediction[i:i+1,:]));
    #nData = np.concatenate((predicted[0:i+1,:],testData[i+1:wLen:,]));
    #nData = np.concatenate((testData[i+1:wLen:,],prediction[0:i+1,:]));
score = prediction[1,:]
accuracy = calculateAccuracy(score, totalPwrLvl, numberOfSamples)
print("Error Rate: ",accuracy);
plotSVM(totalPwrLvl,prediction,numberOfSamples)
#svmp.generatereqByLambda(lambda1,lambda2,numberOfSamples)
