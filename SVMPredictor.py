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
from datetime import datetime
from sklearn.preprocessing import Normalizer
import pickle
#numberOfSamples = 3500
lambda1 = 0.9;
lambda2 = 0.9;
sample_len = 5
max_iteration = 180
powerLvl=-30
obsSample = [];
class SVMPredictor:

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
        
        N = numberOfSamples
        dLen = 100 #length of the energy detector
        N_train = dLen*N//2-dLen+1; #training data length
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length 
        N = N_train+N_test; #total data length

        
        #input window length
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

        nData = testData;

        clf = svm.LinearSVR(epsilon=10e-5, C=10e6,  verbose=6,max_iter=200,random_state=0,loss='squared_epsilon_insensitive',tol=10e-6).fit(train_reshape,label_reshape)

        fSteps = dLen; #tracks number of future steps to predict

        predicted = np.zeros((fSteps, N_test-wLen));
#        filename="models/linearsvr_500itr_5000s_lambda99_average_v1.sav"
#        pickleDmp = open(filename,'wb')
#        pickle.dump(clf,pickleDmp)

        for i in np.arange(0,sample_len):
            predicted[i] = clf.predict(nData.transpose());
            nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));
            #nData = np.concatenate((predicted[0:i+1,:],testData[i+1:wLen:,]));

        predSet = np.zeros((fSteps, N_test-wLen));
        setCnt = 0;
        obsSample = np.zeros((0,N_test-wLen));
        for i in np.arange(0,N_test-wLen):
            predSet[setCnt,i-setCnt:i-setCnt+fSteps] = predicted[:,i].transpose()
            if (setCnt+1)==fSteps:
                obsSample[i-setCnt] = 1;
                setCnt = 0;
            else:
                setCnt = setCnt + 1;
        return predicted
    
    def chooseBestPredictionScore(self,prediction,formattedData,cumulant,numberOfSamples):
        errorScore = 999999 #Initial error score

        for i in np.arange(0,sample_len):
            score = prediction[i,:]
            predictedErrScore = svmp.calculateAccuracy(score, formattedData, numberOfSamples)
            if predictedErrScore < errorScore:
                errorScore = predictedErrScore;
                score = prediction[i,:]
            print("Error Rate: ",cumulant,"Number of Samples",numberOfSamples,"Score: ",errorScore);
        return score

    def plotSVM(self, totalPwrLvl, prediction,sampleSize,cumulant):
        dLen = 100 #length of the energy detector
        N = sampleSize
        N_train = dLen*N//2-dLen+1; #training data length - 14901
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length - 15000
        N = N_train+N_test;
        
        score = svmp.chooseBestPredictionScore(prediction,totalPwrLvl,cumulant)
        
        totalPwr_score = np.zeros((N));
        prediction_score = np.zeros((N_test-wLen));
        
        
        totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
        
        prediction_score = 10*np.log10(np.abs(score))-30
        subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
        
        fig = plt.figure(figsize=(20,10))
        
        plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
        plt.title('One-step ahead prediction')
        plt.legend(['Input Signal','Prediction'])
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (dBm)')
        plt.show()

    def generatereqByLambda(self,lambda1, lambda2,numberOfSamples):
        lambda_range=0.1
        lambda_length = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda range
        color_range = np.zeros(lambda_length*lambda_length)

        a = 0;
        
        x_range=np.zeros(lambda_length*lambda_length)
        y_range=np.zeros(lambda_length*lambda_length)
        
        fig = plt.figure(figsize=(20,20))

        for i in np.arange(lambda_range,lambda1,lambda_range):
            for j in np.arange(lambda_range,lambda2, lambda_range):
                
                totalPwrLvl = arp.generateFreqEnergy(arp,i,j,numberOfSamples)
                powerData = pd.DataFrame(np.array(totalPwrLvl).reshape(-1,1),totalPwrLvl)
                powerData = powerData[powerData.columns[0:]].values
                powerLvlLambda = powerData.transpose()
                
                prediction = svmp.predictor(numberOfSamples,powerLvlLambda)
                
                score = prediction[1,:]
                accuracy = svmp.calculateAccuracy(score, powerLvlLambda, numberOfSamples)

                color_range.itemset(a,accuracy/100)
                x_range.itemset(a,i);
                y_range.itemset(a,j);
                a=a+1;

        plt.scatter(x_range,y_range,c=color_range, s=accuracy*10,alpha=0.55)
        plt.colorbar()
        fig.savefig("lambda0.99range0.01_300samples.png")
        return 0;
    
    def saveARPData(self,fileName ,data):
        with open(fileName, 'w') as arp_dataset:
            wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerows([[lst] for  lst in data])
    
    def load_data(self, fileName):
        data = pd.read_csv(fileName, header=None,sep='\n')
        data = data[data.columns[0:]].values
        return data.transpose()
    
    def calculateAccuracy(self, score, powerLvl,sampleSize):
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
    
lambda_list = np.arange(0.1,0.2,0.1)
powerList = [-30]
numOfSamples =1000
snr_list=[]
errorList = []
errorList_20 = []
errorList_10 = []
errorList_0 = []
errorList_5N = []
errorList_10N = []
lamb_list = []


for  power in powerList:

    for lamd in lambda_list:
        svmp = SVMPredictor()
        filename = "svmDataSet_02.csv"
        filename_cumulant = "svmDataSet_02_cumulant.csv"
        avgTotalPwrLvl, totalAvgPwr_cumulant,snr,thres = arp.generateFreqEnergy(arp,lamd,lamd,numOfSamples,power)
        snr_s="{:.0f}".format(snr)
        snr_list.append(int(snr_s))

        
        
        svmp.saveARPData(filename,avgTotalPwrLvl)
        totalPwrLvl = svmp.load_data(filename)
        
        svmp.saveARPData(filename_cumulant,totalAvgPwr_cumulant)
        totalPwrLvl_cumulants = svmp.load_data(filename_cumulant)
    
        normalize = Normalizer()
        normalize_cumulants = Normalizer()
        avgTotalPwrLvl = normalize.fit_transform(totalPwrLvl)
        totalPwrLvl_cumulant = normalize_cumulants.fit_transform(totalPwrLvl_cumulants)
        #svmp.generatereqByLambda(lambda1,lambda2,numberOfSamples)
        prediction = svmp.predictor(numOfSamples,avgTotalPwrLvl)
        #prediction_cumulants = svmp.predictor(num,totalPwrLvl_cumulant)
        
        error_rate = svmp.calculateAccuracy(prediction[1,:],avgTotalPwrLvl,numOfSamples)
        
        if int(snr_s) == 20:
            errorList_20.append(error_rate)
        elif int(snr_s) == 10:
            errorList_10.append(error_rate)
        elif int(snr_s) == 0:
            errorList_0.append(error_rate)
        elif int(snr_s) == -5:
            errorList_5N.append(error_rate)
        elif int(snr_s) == -10:
            errorList_10N.append(error_rate)
        else:
            errorList.append(error_rate)

        #svmp.chooseBestPredictionScore(prediction,avgTotalPwrLvl,"Average",numOfSamples)
        #svmp.chooseBestPredictionScore(prediction_cumulants,totalPwrLvl_cumulant,"cumulants",num)
    print('--------------------------------------------------------------------------------------')

plt.plot(lambda_list,errorList_20,'rs-',lambda_list,errorList_10,'b*--',lambda_list,errorList_0,'gp-',lambda_list,errorList_5N,'m^:',lambda_list,errorList_10N,'yX--')  
plt.title('Conventional Energy Detector Performance')
plt.legend(['20 dB SNR','10 dB SNR','0 dB SNR','-5 dB SNR','-10 dB SNR'])
plt.xlabel('Activity Statistic')
plt.ylabel('Detection Accuracy')
plt.show()