# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:08:35 2019

@author: hubert.kyeremateng
"""

import numpy as np
import matplotlib.pyplot as plt
from ARP_simulator import ARP_Simulator
import time

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
powerLvlList_40 = [-40]
powerLvlList_50 = [-50]
powerLvlList_60 = [-60]
powerLvlList_65 = [-65]
powerLvlList_70 = [-70]
#lambdaList = [0.1]
powerLvlList = [-40,-50,-60,-65,-70]
sNR_list=[]
lambs = []
#lambdaList = [0.1]
#powerLvlList = [-70]

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
snr_svm_accuracy_list = []
#t1 = time.time()
#arp = ARP_Simulator()
#vectFunc = np.vectorize(arp.arpSimulatorGenerator,otypes=[np.ndarray],cache=True)
#snr_acc_svm_20.append(vectFunc(lambdaList,lambdaList,powerLvlList_40))
#snr_acc_svm_10.append(vectFunc(lambdaList,lambdaList,powerLvlList_50))
#snr_acc_svm_0.append(vectFunc(lambdaList,lambdaList,powerLvlList_60))
#snr_acc_svm_N5.append(vectFunc(lambdaList,lambdaList,powerLvlList_65))
#snr_acc_svm_N10.append(vectFunc(lambdaList,lambdaList,powerLvlList_70))
#t2 = time.time()
#print((t2-t1)/60)
t1 = time.time()
for power in powerLvlList:
    for lamb in lambdaList:
        arp = None
        arp = ARP_Simulator()
        
        acc,acc_svm,SNR = arp.arpSimulatorGenerator(lamb,lamb,power)
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
            
fig = plt.figure(figsize=(20,20))
plt.subplot(211)
plt.rc('font',size=16)

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
plt.subplots_adjust(hspace=1.0)
plt.savefig("accuracy_cumulants_convention_02_25_2020.png")
plt.savefig("accuracy_cumulants_convention_02_25_2020.jpeg")
plt.show() 
t2 = time.time()
print((t2-t1)/60)