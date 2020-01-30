# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:08:35 2019

@author: hubert.kyeremateng
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from ARP import ARPSimulator as arp
import csv
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import Normalizer,StandardScaler
from matplotlib.ticker import FormatStrFormatter
#numberOfSamples = 500

sample_len = 5
max_iteration = -1

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


       #print(grid.best_params_)
        clf = svm.LinearSVR(epsilon=10e-8, C=10e6,verbose=6, random_state=0,max_iter=500,loss='squared_epsilon_insensitive',tol=10e-6).fit(train_reshape,label_reshape)
        fSteps = dLen; #tracks number of future steps to predict

        predicted = np.zeros((fSteps, N_test-wLen));
#        filename="models/linearsvr_155kitr_5000s_lambda9_v2.sav"
#        pickleDmp = open(filename,'wb')
#        pickle.dump(clf,pickleDmp)

        for i in np.arange(0,sample_len):
            predicted[i] = clf.predict(nData.transpose());
            #nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));
            #nData = np.concatenate((predicted[i:i+1,:],testData[1:wLen:,]));
            nData = np.concatenate((testData[i+1:wLen:,],predicted[0:i+1,:]));
            #nData = np.concatenate((predicted[0:i+1,:],testData[i+1:wLen:,]));

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
    
    def chooseBestPredictionScore(self,prediction,formattedData,numberOfSamples):
        errorScore = 999999 #Initial error score

        for i in np.arange(0,sample_len):
            score = prediction[i,:]
            predictedErrScore = svmp.calculateAccuracy(score, formattedData, numberOfSamples)
            print("Error Score ",predictedErrScore)
            if predictedErrScore < errorScore:
                errorScore = predictedErrScore;
                score = prediction[i,:]
           
        return score,errorScore

    def plotSVM(self, totalPwrLvl, prediction,sampleSize):
        dLen = 100 #length of the energy detector
        N = sampleSize
        N_train = dLen*N//2-dLen+1; #training data length - 14901
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length - 15000
        N = N_train+N_test;
        
        score, errorScore = svmp.chooseBestPredictionScore(prediction,totalPwrLvl,sampleSize)
        error_score = float(format(errorScore,'.4f'))
        totalPwr_score = np.zeros((N));
        prediction_score = np.zeros((N_test-wLen));
        
        
        totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
        
        prediction_score = 10*np.log10(np.abs(score))-30
        subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
        caption = "Sample Size: "+str(sampleSize)+ " Number of Iterations: "+str(max_iteration)+" MSE: "+str(error_score)
        fig = plt.figure(figsize=(20,10))
        fig.text(.5, .05, caption, ha='center')
        plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
        plt.title('One-step ahead prediction')
        plt.legend(['Input Signal','Prediction'])
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (dBm)')
        plt.show()

    def generatereqByLambda(self,lambda1, lambda2,numberOfSamples):
        lambda_range=0.01
        lambda_length = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda range
        color_range = np.zeros(lambda_length*lambda_length)

        a = 0;
        
        x_range=np.zeros(lambda_length*lambda_length)
        y_range=np.zeros(lambda_length*lambda_length)
        
        fig = plt.figure(figsize=(30,30))

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
    
    def saveARPData(self,fileName, data):
        with open(fileName, 'w') as arp_dataset:
            wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerows([[lst] for  lst in data])
    
    def load_data(self, fileName):
        data = pd.read_csv(fileName, header=None,sep='\n')
        data = data[data.columns[0:]].values
        return data.transpose()
    
    def plotByPowerLevel(self, powerList,error_scoreList_norm,error_scoreList_cum):
        if len(powerList) == 0 or len(error_scoreList_norm) == 0:
            print("List is empty")
        caption = "Maximum Iteration: 150 Sample size: "+str(numberOfSamples)           
        fig = plt.figure(figsize=(10,10))
        fig.text(.5, .05, caption, ha='center')
        plt.plot(powerList,error_scoreList_norm,'bo-',powerList,error_scoreList_cum,'rs-')
        for s, val in zip(powerList,error_scoreList_norm):
            label = "{:.4f}".format(val)
            plt.annotate(label, # this is the text
                 (s,val), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
        for s, val in zip(powerList,error_scoreList_cum):
            label = "{:.4f}".format(val)
            plt.annotate(label, # this is the text
                 (s,val), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')            
            #diagram.annotate(val,powerList[s])
        plt.title('One-step ahead prediction')
        plt.legend(['Moving Average','Cumulants - First-order'])
        plt.xlabel('On/Off State')
        plt.ylabel("Prediction Error Rate")
        plt.show() 
        
    def calculateAccuracy(self, score, powerLvl,sampleSize,thresVal):
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
        leng = score.shape[0]
        for s in np.arange(leng):
            if thresVal > score[s]:
                print('yes')
        
        return rmse;

fileName = "svmDataSet.csv"
fileName_cumulants = "svmDataSet_cumulants.csv"
sampleList = [500]
lambdaList = [0.9]
powerLvlList = [-40]
error_scoreList = []
error_scoreList_norm = []
error_scoreList_cum = []
error_score_lambda_norm = []
error_score_lambda_cum = []
sNR_list=[]
lambs = []
numberOfSamples= 1000
lambda1 = 0.8;
lambda2 = 0.8;
snr_list_N10=[]
snr_list_N5=[]
snr_list_0=[]
snr_list_10=[]
snr_list_20=[]
lamb_list_N10=[]
lamb_list_N5=[]
lamb_list_0=[]
lamb_list_10=[]
lamb_list_20=[]
for power in powerLvlList:
    for lamb in lambdaList:
        svmp = SVMPredictor()
        
        accuracy,SNR,acc = arp.generateFreqEnergy(arp,lamb,lamb,numberOfSamples,power)
#        svmp.saveARPData(fileName,totalPwrLvl)
#        totalPwrLvl = svmp.load_data(fileName)
        #print(np.round(SNR,0), acc)
        sNR_list.append(SNR)
        lambs.append(lamb)
#        minmax = Normalizer(norm='l2')
#        #norm_thresh = minmax.fit(thresh)
#        #totalPwrLvl_minmax = minmax.fit_transform(totalPwrLvl)
#        totalPwrLvl_minmax = totalPwrLvl
#        prediction_minmax = svmp.predictor(numberOfSamples,totalPwrLvl_minmax)
#        prediction_norm_acc = prediction_minmax[1,:]
#        accuracy_norm = svmp.calculateAccuracy(prediction_norm_acc, totalPwrLvl_minmax, numberOfSamples,thresVal)
        if np.round(SNR,0) == 10:
            snr_list_10.append(accuracy)
            lamb_list_10.append(lamb)
        elif np.round(SNR,0) == 20:
            snr_list_20.append(accuracy)
            lamb_list_20.append(lamb)
        elif np.round(SNR,0) == 0:
            snr_list_0.append(accuracy)
            lamb_list_0.append(lamb)
        elif np.round(SNR,0) == -5:
            snr_list_N5.append(accuracy)
            lamb_list_N5.append(lamb)     
        elif np.round(SNR,0) == -10:
            snr_list_N10.append(accuracy)
            lamb_list_N10.append(lamb)
#           

fig = plt.figure(figsize=(20,30))
plt.plot(lambdaList,snr_list_N5,'ys-',lambdaList,snr_list_N10,'rs-',lambdaList,snr_list_0,'bs-',lambdaList,snr_list_10,'gs-',lambdaList,snr_list_20,'cs-')          
plt.legend(['-5 dB SNR','20 dB SNR'])
plt.xlabel('Activity Statistic')
plt.ylabel("Prediction Accuracy Rate")
plt.show() 