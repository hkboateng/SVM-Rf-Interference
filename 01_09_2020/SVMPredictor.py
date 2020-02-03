# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:43:12 2019

@author: asus
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
import csv
import pickle

class SVM_Predictor:
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
        def predictor_cumulant(self,numberOfSamples, datas):
        
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

    
            nData = testData;
    
            label_reshape = trainLbl.reshape((N_train-wLen))
            train_reshape = trainData.transpose()    
            #model = svm.LinearSVR(epsilon=10e-9, C=10e6,verbose=6,random_state=0,loss='squared_epsilon_insensitive',tol=10e-6,max_iter=250).fit(train_reshape,label_reshape)
            filename="Models/linearsvr_500itr_5000s_lambda99_cumulants_v1.sav"
            model = pickle.load(open(filename,'rb'));
            fSteps = dLen; #tracks number of future steps to predict
    
            predicted = np.zeros((fSteps, N_test-wLen));

            for i in np.arange(0,5):
                predicted[i] = model.predict(nData.transpose());
                nData = np.concatenate((testData[i+1:wLen:,],predicted[0:i+1,:]));
    
            return predicted
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

    
            nData = testData;
    
            label_reshape = trainLbl.reshape((N_train-wLen))
            train_reshape = trainData.transpose()    
           #print(grid.best_params_)
            #model = svm.LinearSVR(epsilon=10e-90,C=10e12,verbose=6, random_state=0,max_iter=25,tol=10e-6).fit(train_reshape,label_reshape)
            fSteps = dLen; #tracks number of future steps to predict
    
            predicted = np.zeros((fSteps, N_test-wLen));
            filename="Models/linearsvr_95kitr_1000s_lambda99.sav"
            model = pickle.load(open(filename,'rb'));
            
            for i in np.arange(0,5):
                predicted[i] = model.predict(nData.transpose());
                nData = np.concatenate((testData[i+1:wLen:,],predicted[0:i+1,:]));
                #nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));
    
            return predicted
        def plotSVM(self,totalPwrLvl, prediction,sampleSize,thresh,withPrediction):
            dLen = 100 #length of the energy detector
            N = sampleSize
            dLen = 100 #length of the energy detector
            N_train = dLen*N//2-dLen+1; #training data length
            wLen = 5*dLen;
            N_test = dLen*N//2; #test data length
            N = N_train+N_test; #total data length           
            
            
            totalPwr_score = np.zeros((N));
            prediction_score = np.zeros((N_test-wLen));
            
            thresh_score = 10*np.log10(thresh*np.ones((np.size(totalPwrLvl[0,N_train+wLen:N]))))-30
            totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
            
            
            subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
            
            plt.figure(figsize=(20,10))
            plt.plot(subplot_range,totalPwr_score)
                      
            if withPrediction:
                score = prediction[1,:]
                prediction_score = 10*np.log10(np.abs(score))-30
                plt.plot(subplot_range,prediction_score)
                plt.legend(['Input Signal','Prediction'])
                plt.title('One-step ahead prediction')
                plt.ylabel('Magnitude (dBm)')
            else:
                plt.plot(subplot_range,thresh_score)  
                plt.legend(['Input Signal','Threshold'])
                plt.title('Simulated RF Input')
                plt.ylabel('Total Average Power (dBM)')
            
            plt.xlabel('Samples')
            plt.show()
            plt.savefig("simulated_totalaverage_01_28_2020.png")
        
        def saveARPData(self,fileName, data):
            with open(fileName, 'w') as arp_dataset:
                wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
                wr.writerows([[lst] for  lst in data])
    
        def load_data(self, fileName):
            data = pd.read_csv(fileName)
            data = data[data.columns[0:]].values
            return data