#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 16:17:12 2018

@author: hkyeremateng-boateng
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from ARPSimulator import ARPSimulator as arp
import csv
from sklearn.metrics import mean_squared_error
import pickle

numberOfSamples = 500
lambda1 = 0.2;
lambda2 = 0.2;
dLen = 100 #length of the energy detector
wLen = 5*dLen; 

class SVMPredictor:
    actualTotalSamples = 0;
    N_train=0

    N_test =0
    def getTrainLabl(self, dataSource,trainVec,wLen,trainLen ):
        counter = 0;
        for i in np.arange(wLen,trainLen-1):
            trainVec.itemset(counter,dataSource[0,i]);
            counter = counter +1;
        return trainVec,counter;
        
    def generateTestDataSet(self,start,end,dataSource,testData,trainData):
        for s in np.arange(start):
            for k in  np.arange(end):
                testData.itemset((k,s),np.real(dataSource[0,trainData+s+k]))
        return testData;
    
    def generateTrainDataSet(self,start,end,dataSource,trainData):
        for s in np.arange(start):
            for k in  np.arange(end):
                trainData.itemset((k,s),np.real(dataSource[0,s+k]))
        return trainData;
    
    def predictor(self,numberOfSamples, datas):
        
        #N = numberOfSamples

        N_train = dLen*numberOfSamples//2-dLen+1; #training data length 
        N_test = dLen*numberOfSamples//2; #test data length 

        trainLbl = np.zeros((1,N_train-wLen));
        
        counter = 0
        for i in np.arange(wLen,N_train-1):
            trainLbl.itemset(counter,datas[0,i]);
            counter= counter+1;
        
        #Traing and test input data
        trainData = np.zeros((wLen,N_train-wLen));
        testData = np.zeros((wLen,N_test-wLen));
        ###### totalAvgPwr is one-dimensional array#####
        ###### trainData in a multi-dimensional array######
        trainData = self.generateTrainDataSet(counter,wLen-1,datas,trainData);
        
        testData = self.generateTestDataSet(N_test-wLen,wLen,datas, testData, N_train)

        label_reshape = trainLbl.reshape((N_train-wLen))
        train_reshape = trainData.transpose()
        clf = svm.LinearSVR(epsilon=10e-90, C=10e6, max_iter=40000, dual=True,random_state=0,loss='squared_epsilon_insensitive',tol=10e-5).fit(train_reshape,label_reshape)
        fSteps = dLen; #tracks number of future steps to predict
        model_name = 'linearSVRModel_500s_30000itr.sav'
        with open(model_name,'wb') as model_file:
            pickle.dump(clf,model_file)
            
        predicted = np.zeros((fSteps, N_test-wLen));
 
        nData = testData;
        for i in np.arange(0,prediction_width):
            predicted[i] = clf.predict(nData.transpose());
            #nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:])); #  - Original One-step ahead prediction
            #nData = np.concatenate((testData[i+1:wLen:,],predicted[0:i+1,:])); # -Using predicted datapoint as the last datapoint to use in next prediction
            nData = np.concatenate((predicted[0:i+1,:],testData[i+1:wLen:,])); # - Using predicted datapoint as the initial datapoint to use in next prediction

  
        return predicted,nData
    
    def plotSVM(self, totalPwrLvl, score,sampleSize):

        N_train = dLen*sampleSize//2-dLen+1; #training data length - 14901
        N_test = dLen*sampleSize//2; #test data length - 15000
        totalSampleSize = N_train+N_test;
        
        totalPwr_score = np.zeros((totalSampleSize));
        prediction_score = np.zeros((N_test-wLen));
        
        
        totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:totalSampleSize])))-30
        
        prediction_score = 10*np.log10(np.abs(score))-30
        subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
        
        plt.figure(figsize=(10,10))
        
        plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
        plt.title('one-step ahead prediction')
        plt.legend(['Input Signal','Prediction'])
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (dBm)')
        plt.show()

    def generatereqByLambda(self,lambda1, lambda2,numberOfSamples):
        lambda_range=0.1
        
        lambda1_length = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda 1 range
        lambda2_length = len(np.arange(lambda_range,lambda2,lambda_range))         

        index = 0;
        color_range = np.zeros(lambda1_length*lambda2_length)
        fig = plt.figure(figsize=(20,10))
        x_range = np.zeros(lambda1_length*lambda2_length)
        y_range = np.zeros(lambda1_length*lambda2_length)
        for i in np.arange(lambda1,lambda_range,-lambda_range):

            for j in np.arange(lambda2,lambda_range, -lambda_range):

                totalPwrLvl = arp.generateFreqEnergy(arp,i,j,numberOfSamples)
                powerData = pd.DataFrame(np.array(totalPwrLvl).reshape(-1,1),totalPwrLvl)
                powerData = powerData[powerData.columns[0:]].values
                powerLvlLambda = powerData.transpose()
                
                prediction = svmp.predictor(numberOfSamples,powerLvlLambda)

                score = prediction[1,:]
                
                accuracy = svmp.calculateAccuracy(score, powerLvlLambda, numberOfSamples)
                print("Lambda 1",i,"lambda2",j," -: Error Rate: ",accuracy);

                color_range.itemset(index,(accuracy/100))
                
                x_range.itemset(index,i);
                y_range.itemset(index,j);

                index = index+1;
            
        plt.scatter(x_range,y_range,c=color_range, s=(accuracy)*10,alpha=0.55)
        plt.colorbar()
        fig.savefig("lambda0.8range0.01_300samples.png")
    
    def saveARPData(self, data):
        with open(powerLvl_dataSource, 'w') as arp_dataset:
            wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerows([[lst] for  lst in data])
    
    def load_data(self, fileName):
        data = pd.read_csv(fileName, header=None,sep='\n')
        data = data[data.columns[0:]].values
        return data.transpose()
    

    def calculateAccuracy(self, score, powerLvl,sampleSize):
        N_train = dLen*sampleSize//2-dLen+1; #training data length 
        N_test = dLen*sampleSize//2; #test data length
        sampleSize = N_train+N_test;
        
        totalPwr = 10*np.log10(np.abs(np.real(powerLvl[0,N_train+wLen:sampleSize])))-30
        prediction = 10*np.log10(np.abs(score))-30
        rmse = mean_squared_error(prediction,totalPwr)
        return rmse;

prediction_width = 50   
sample_len = 5
svmp = SVMPredictor()
powerLvl_dataSource = "svmDataSet.csv"
totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
svmp.saveARPData(totalPwrLvl)
totalPwrLvl = svmp.load_data(powerLvl_dataSource)
prediction,text_X = svmp.predictor(numberOfSamples,totalPwrLvl)

for i in range(prediction_width):
    accuracy = svmp.calculateAccuracy(prediction[i,:], totalPwrLvl, numberOfSamples)
    print("Error Rate: ",accuracy);
svmp.plotSVM(totalPwrLvl,prediction[sample_len-1,:],numberOfSamples)
#svmp.generatereqByLambda(lambda1,lambda2,numberOfSamples)
