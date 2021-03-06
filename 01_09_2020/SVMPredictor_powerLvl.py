# -*- coding: utf-8 -*-
"""
Created on Thurs Jul  11 2019

@author: hubert.kyeremateng
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd

#from ARP_Simulator import ARPSimulator as arp

from ARP_simulator import ARP_Simulator

import csv
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import Normalizer,StandardScaler
from matplotlib.ticker import FormatStrFormatter
#numberOfSamples = 500

sample_len = 5
max_iteration = 500

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
    
    def predictor(self,numberOfSamples, datas,maxt_iter=max_iteration):
        
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
        clf = svm.LinearSVR(epsilon=10e-9, C=10e6,verbose=6,random_state=0,loss='squared_epsilon_insensitive',tol=10e-6,max_iter=maxt_iter).fit(train_reshape,label_reshape)
        fSteps = dLen; #tracks number of future steps to predict

        predicted = np.zeros((fSteps, N_test-wLen));

        for i in np.arange(0,sample_len):
            predicted[i] = clf.predict(nData.transpose());
            nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));

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

    def generatereqByLambda(self,lambda1, lambda2,numberOfSamples,isCumulant=False):
        #print("Lambda 1",lambda1,"Lambda 2",lambda2, "Sample Size",numberOfSamples)
        lambda_range=0.05
        lambda_length = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda range
        color_range = np.zeros(lambda_length*lambda_length)

        a = 0;
        
        x_range=np.zeros(lambda_length*lambda_length)
        y_range=np.zeros(lambda_length*lambda_length)
        
        fig = plt.figure()
        arp = ARP_Simulator()
        for i in np.arange(lambda_range,lambda1,lambda_range):
            for j in np.arange(lambda_range,lambda2, lambda_range):                
                acc,prec_accuracy,snr = arp.arpSimulatorGenerator(i,j,-40)

                color_range.itemset(a,acc)
                x_range.itemset(a,i);
                y_range.itemset(a,j);
                a=a+1;
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (dBm)')
        plt.title('Simulated Prediction Accuracy')
        plt.scatter(x_range,y_range,c=color_range, s=acc*10000,alpha=0.95,edgecolors='None')
        plt.colorbar()
        fig.savefig("lambda0.99range0.1_1000samples_02_09_2020.jpeg")
        fig.savefig("lambda0.99range0.1_1000samples_02_09_2020.png")
        return 0;
    
    def saveARPData(self,fileName, data):
        with open(fileName, 'w') as arp_dataset:
            wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerows([[lst] for  lst in data])
    
    def load_data(self, fileName):
        data = pd.read_csv(fileName, header=None,sep='\n')
        data = data[data.columns[0:]].values
        return data.transpose()
    
    def plotByPowerLevel(self, powerList,error_scoreList_norm,error_scoreList_cum,image="test"):
        if len(powerList) == 0 or len(error_scoreList_norm) == 0:
            print("List is empty")
        caption = "Sample size: "+str(numberOfSamples)           
        plt.rc('font',size=22)
        fig = plt.figure(figsize=(25,20))
        #fig.text(.5, .05, caption, ha='center')
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
                 xytext=(0,-20), # distance from text to points (x,y)
                 ha='center')            
            #diagram.annotate(val,powerList[s])
        image_file = image+".png"
        
        plt.title('One-step ahead prediction')
        plt.legend(['Moving Average','Cumulants - First-order'])
        plt.xlabel('Power Level')
        plt.ylabel("Prediction - RMSE Rate")
        fig.savefig(image_file)
        plt.show() 
        
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

fileName = "svmDataSet.csv"
fileName_cumulants = "svmDataSet_cumulants.csv"
sampleList = np.arange(500,5500,500)
lambdaList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
powerLvlList = np.arange(-150,0,10)

iterList = np.arange(50,550,50)
error_scoreList = []
error_scoreList_norm = []
error_scoreList_cum = []
error_score_lambda_norm = []
error_score_lambda_cum = []
error_score_lambda_scale = []
numberOfSamples= 1000

lambda1 = 1;
lambda2 = 1;
svmp = SVMPredictor()
svmp.generatereqByLambda(lambda1,lambda2,numberOfSamples,False)
#for power in powerLvlList:
#    svmp = SVMPredictor()
#    arp = None
#    arp = ARP_Simulator()   
#    totalPwrLvl,totalPwrLvl_cumulants = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples,power)
#    svmp.saveARPData(fileName,totalPwrLvl)
#    totalPwrLvl = svmp.load_data(fileName)
#    
#    svmp.saveARPData(fileName_cumulants,totalPwrLvl_cumulants)
#    totalPwrLvl_cumulants = svmp.load_data(fileName_cumulants)    
#    
#    scale = StandardScaler(with_mean=False)
#    norm = Normalizer(norm='l2')
#    norm_cumulants = Normalizer(norm='l2')
#    
##    totalPwrLvl_norm = norm.fit_transform(totalPwrLvl)
#    #totalPwrLvl_cumulants = norm_cumulants.fit_transform(totalPwrLvl_cumulants)
#    totalPwrLvl_scale = norm.fit_transform(totalPwrLvl)
#    
##    prediction_norm = svmp.predictor(numberOfSamples,totalPwrLvl_norm,iters)
#    #prediction = svmp.predictor(numberOfSamples,totalPwrLvl)
#    prediction_scale = svmp.predictor(numberOfSamples,totalPwrLvl_scale)
#    prediction_cumulants = svmp.predictor(numberOfSamples,totalPwrLvl_cumulants)
#    
#    accuracy_scale = svmp.calculateAccuracy(prediction_scale[1,:], totalPwrLvl_scale, numberOfSamples)
#    #accuracy = svmp.calculateAccuracy(prediction[1,:], totalPwrLvl, numberOfSamples)
##    accuracy_norm = svmp.calculateAccuracy(prediction_norm[1,:], totalPwrLvl_norm, numberOfSamples)
#    accuracy_cumulants = svmp.calculateAccuracy(prediction_cumulants[1,:], totalPwrLvl_cumulants, numberOfSamples)
##    for i in np.arange(0,sample_len):
##        score = prediction_norm[i,:]
##        accuracy = svmp.calculateAccuracy(score, totalPwrLvl_norm, numberOfSamples)
##        print("Error Rate: Total Average Power",numberOfSamples,accuracy);
##    print('   -----------------------        ')
##    for j in np.arange(0,sample_len):
##        score_cumulants = prediction_cumulants[j,:]
##        accuracy_cumulants = svmp.calculateAccuracy(score_cumulants, totalPwrLvl_cumulants, numberOfSamples)
##        print("Error Rate: Cumulants",numberOfSamples,accuracy_cumulants);  
##    print('------------------------------------------------------------------')
##    error_scoreList.append(accuracy)
##    error_scoreList_norm.append(accuracy_norm)
##    error_scoreList_cum.append(accuracy_cumulants)
##    error_score_lambda_norm.append(accuracy_norm)
#    error_score_lambda_scale.append(accuracy_scale)
#    error_score_lambda_cum.append(accuracy_cumulants)
#        #svmp.plotSVM(totalPwrLvl,prediction,numberOfSamples)
#        #svmp.plotSVM(totalPwrLvl_norm,prediction_norm,numberOfSamples)
#        #svmp.plotSVM(totalPwrLvl_cumulants,prediction_cumulants,numberOfSamples)
##svmp.plotByPowerLevel(powerLvlList,error_scoreList_norm,error_scoreList_cum)
#svmp.plotByPowerLevel(powerLvlList,error_score_lambda_scale,error_score_lambda_cum,"powerLvl")



