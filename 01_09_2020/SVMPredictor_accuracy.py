# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:08:35 2019

@author: hubert.kyeremateng
"""

import numpy as np
import matplotlib.pyplot as plt
from ARP_simulator import ARP_Simulator


sample_len = 5
max_iteration = -1


numberOfSamples= 1000
N = numberOfSamples
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length
#Training label ground truth/target
wLen = 5*dLen; 
N_test = dLen*N//2; #test data length 
N = N_train+N_test; #total data length
lambdaList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
powerLvlList = [-70,-65,-50,-60,-45,-40]
#lambdaList = [0.1]
#powerLvlList = [-40]
sNR_list=[]
lambs = []


snr_acc_N10=[]
snr_acc_N5=[]
snr_acc_0=[]
snr_acc_10=[]
snr_acc_20=[]
snr_acc_svm_N10=[]
snr_acc_svm_N5=[]
snr_acc_svm_0=[]
snr_acc_svm_10=[]
snr_acc_svm_20=[]
lamb_list_N10=[]
lamb_list_N5=[]
lamb_list_0=[]
lamb_list_10=[]
lamb_list_20=[]
pStates = []


for power in powerLvlList:
    for lamb in lambdaList:
        arp = None
        arp = ARP_Simulator()
        
        acc,acc_svm,SNR = arp.arpSimulatorGenerator(lamb,lamb,numberOfSamples,power,True)

        snr = np.round(SNR,0)

        if snr == 10:
            snr_acc_10.append(acc)
            snr_acc_svm_10.append(acc_svm)
            lamb_list_10.append(lamb)
        elif snr == 20:
            snr_acc_20.append(acc)
            snr_acc_svm_20.append(acc_svm)
            lamb_list_20.append(lamb)
        elif snr == 0:
            snr_acc_0.append(acc)
            snr_acc_svm_0.append(acc_svm)
            lamb_list_0.append(lamb)
        elif snr == -5:
            snr_acc_N5.append(acc)
            snr_acc_svm_N5.append(acc_svm)
            lamb_list_N5.append(lamb)     
        elif snr == -10:
            snr_acc_N10.append(acc)
            snr_acc_svm_N10.append(acc_svm)
            lamb_list_N10.append(lamb)
#            
fig = plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(lamb_list_N5,snr_acc_N5,'ko-',lamb_list_N10,snr_acc_N10,'ys-',lamb_list_0,snr_acc_0,'go-',lamb_list_10,snr_acc_10,'bo-',lamb_list_20,snr_acc_20,'rs-')          

plt.title('Conventional Energy Detector Performance')
plt.legend(['-5 dB SNR','-10 dB SNR','0 dB SNR','10 dB SNR','20 dB SNR'])
plt.xlabel('Activity Statistic (λ₁ = λ₂)')
plt.ylabel("Detection Accuracy")
plt.subplot(212)
plt.plot(lamb_list_N10,snr_acc_svm_N10,'ys-',lamb_list_0,snr_acc_svm_0,'go-',lamb_list_10,snr_acc_svm_10,'bo-',lamb_list_20,snr_acc_svm_20,'rs-')          

plt.title('Predictive Energy Detector Performance')
plt.legend(['-10 dB SNR','0 dB SNR','10 dB SNR','20 dB SNR'])
plt.xlabel('Activity Statistic (λ₁ = λ₂)')
plt.ylabel("Prediction Accuracy")
plt.subplots_adjust(hspace=0.5)
plt.show() 
plt.savefig("accuracy_cumulants_01_20_2020.png")