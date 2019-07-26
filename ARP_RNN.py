# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:46:55 2019

@author: hubert.kyeremateng
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from ARP import ARPSimulator as arp
from keras.layers import Dense, LSTM, Dropout,LSTMCell,Flatten
from keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from keras import optimizers

def saveARPData(data,fileName):
    with open(fileName, 'w') as arp_dataset:
        wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerows([[lst] for  lst in data])

def load_data(fileName):
    data = pd.read_csv(fileName, header=None,sep='\n')
    data = data[data.columns[0:]].values
    return data.transpose()

def getTrainLabl(dataSource,trainVec,wLen,trainLen ):
    counter = 0;
    for i in np.arange(wLen,trainLen):
        for r in np.arange(wLen):
            trainVec.itemset((r,i),dataSource[0,i+r]);
        counter = counter +1;
    return trainVec,counter;

def generateTestDataSet(start,end,dataSource,testData,trainData):
    for s in np.arange(start):
        for k in  np.arange(end):
            testData.itemset((k,s),np.real(dataSource[0,trainData]))
    return testData;

def generateTrainDataSet(start,end,dataSource,trainData):
    for s in np.arange(start):
        for k in  np.arange(end):
            trainData.itemset((k,s),np.real(dataSource[0,s+k]))
    return trainData;

def predictor(numberOfSamples, datas):
        N = numberOfSamples
        dLen = 100 #length of the energy detector
        N_train = dLen*N//2; #training data length
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length 
        N = N_train+N_test; #total data length
        numOfFeatures = 24500
        sample_length = 512
        #input window length
        trainLbl = np.zeros((sample_length,N_train-wLen));

        counter = 0
        for i in np.arange(sample_length,N_train):
            trainLbl.itemset(counter,datas[0,i]);
            counter= counter+1;

        #Traing and test input data
        trainData = np.zeros((sample_length,N_train-wLen));
        testData = np.zeros((sample_length,N_test-wLen));
        ###### totalAvgPwr is one-dimensional array#####
        ###### trainData in a multi-dimensional array######
        trainData = generateTrainDataSet(counter,sample_length-1,datas,trainData);
        testData = generateTestDataSet(N_test-wLen,wLen,datas, testData, N_train)
        
        train_reshape = trainData.transpose()
        nData = testData;
        
        size=testData.shape[1]

        # reshape input to be [samples, time steps, features]
        train_reshape = np.reshape(trainData, (trainData.shape[0],1, trainData.shape[1]))
        test_reshape = np.reshape(testData, (testData.shape[0], 1,testData.shape[1]))
        #testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        print(trainLbl.shape)
        trainLbls = trainLbl.transpose();
        print("Test size",test_reshape.shape)
        print("Train Data size",train_reshape.shape)
        print("Target size",trainLbls.shape)
        model = Sequential()
        model.add(LSTM(sample_length,input_shape=(train_reshape.shape[1:]),return_sequences=True,batch_size=32,stateful=False))
#        model.add(Dropout(0.1))
        model.add(LSTM(256,input_shape=(train_reshape.shape[0:])))
#        model.add(Dropout(0.1))        
#        model.add(LSTM(500,activation='relu'))
#        model.add(Dropout(0.5))      
#        model.add(Flatten(data_format=None,input_shape=(train_reshape.shape[0:])))
        opts = optimizers.Adam(lr=10e-5)
        model.add(Dense(numOfFeatures,activation='linear'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=opts)
        
        model.fit(train_reshape, trainLbl, epochs=10, batch_size=32,shuffle=False)
        return model, test_reshape

def calculateAccuracy(score, powerLvl,sampleSize):
    dLen = 100 #length of the energy detector
    N = sampleSize
    N_train = dLen*N//2; #training data length - 14901
    #Training label ground truth/target
    wLen = 5*dLen; 
    N_test = dLen*N//2; #test data length - 15000
    N = N_train+N_test;
    
    totalPwr = 10*np.log10(np.abs(np.real(powerLvl[0,N_train+wLen:N])))-30
    prediction = 10*np.log10(np.abs(score))-30
    rmse = mean_squared_error(prediction,totalPwr)
    return rmse;

def chooseBestPredictionScore(prediction,formattedData,numberOfSamples):
    errorScore = 999999 #Initial error score

    for i in np.arange(0,5):
        score = prediction[i,:]
        predictedErrScore = calculateAccuracy(score, formattedData, numberOfSamples)
        if predictedErrScore < errorScore:
            errorScore = predictedErrScore;
            score = prediction[i,:]
       
    return score,errorScore
    
def plotSVM(totalPwrLvl, prediction,sampleSize):
    dLen = 100 #length of the energy detector
    N = sampleSize
    N_train = dLen*N//2-dLen+1; #training data length - 14901
    #Training label ground truth/target
    wLen = 5*dLen; 
    N_test = dLen*N//2; #test data length - 15000
    N = N_train+N_test;
    
    score, errorScore = chooseBestPredictionScore(prediction,totalPwrLvl,sampleSize)
    #error_score = float(format(errorScore,'.4f'))
    totalPwr_score = np.zeros((N));
    prediction_score = np.zeros((N_test-wLen));
    print(errorScore)
    
    totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
    
    prediction_score = 10*np.log10(np.abs(score))-30
    subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
    caption = "\nError Score: "+str(errorScore)
    fig = plt.figure(figsize=(20,10))
    fig.text(.5, .05, caption, ha='center')
    plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
    plt.title('One-step ahead prediction')
    plt.legend(['Input Signal','Prediction'])
    plt.xlabel('Samples')
    plt.ylabel('Magnitude (dBm)')
    plt.show()
        
numberOfSamples = 500
fileName = "svmDataSet_01.csv"
lambda1 = 0.8
lambda2 = 0.8
totalPwrLvl,totalPwrLvl_cumulants = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
saveARPData(totalPwrLvl_cumulants,fileName)
totalPwrLvl = load_data(fileName)
norm = Normalizer(norm='max')
totalPwrLvls = norm.fit_transform(totalPwrLvl)
model, test_reshape = predictor(numberOfSamples,totalPwrLvls)

y_pred = model.predict(test_reshape)
plotSVM(totalPwrLvl,y_pred,numberOfSamples)
