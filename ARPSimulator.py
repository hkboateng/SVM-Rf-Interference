#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:51:50 2018

@author: hkyeremateng-boateng
"""

import random as rnd
import math as m
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import time as tic
from scipy import stats

class ARPSimulator:


    def generateFreqEnergy(self,lambda1, lambda2, numberOfSamples,power=-40):

        pi = m.pi

        tic.clock()
        
        kParam1 = 2 #k-parameter for Erlang/gamma distribution (ON)
        kParam2 = 2 #k-parameter for Erlang/gamma distribution (OFF)
        vScale1 = 1 #scales variance relative to lambda1 (optional) 
        vScale2 = 1 #scales variance relative to lambda2 (optional)
        var1 = vScale1*lambda1 #variance parameter for log-normal distribution (ON)
        var2 = vScale2*lambda2 #variance parameter for log-normal distribution (OFF)
        N = numberOfSamples #number of samples
        occupancy = [0]*N
        stateTrans = [] #tracks alternating states [1,0,1,0,1,0,...]
        intTimes = [] #tracks intervals
        upTimes = []
        downTimes = []
        intTimesSeq = [] #counts and tracks intervals
        upDist = "lnorm"    #'exp', 'erl', or 'lnorm'
        downDist = "lnorm"  #'exp', 'erl', or 'lnorm'
        
        #process initialized to "on"
        
        totalTime = 0 #tracks total time generated by the ARP
        seqState = 1 #tracks next state to generate
        
        while totalTime < N:
            #generates on sequence
            if seqState:
                #generates random on period
                if upDist=="exp":
                    period = m.ceil(rnd.expovariate(lambda1))
                elif upDist=="erl":
                    period = m.ceil(rnd.gammavariate(kParam1,1/lambda1)) #assumes k=2
                elif upDist=="lnorm":
                    trueMu = m.log(((1/lambda1)**2)/m.sqrt((1/var1)+(1/lambda1)**2))
                    trueSig = m.sqrt(m.log((1/var1)/((1/lambda1)**2)+1))
                    period = m.ceil(rnd.lognormvariate(trueMu,trueSig)) 
                #period = 5
                
                if (totalTime+period) > N: #makes sure total time isn't exceeded
                    occupancy[totalTime:N] = [1]*(N-totalTime)
                else: #appends proper sequence of 1s
                    occupancy[totalTime:totalTime+period] = [1]*period
                    
                #tracks state transitions and on/off durations    
                stateTrans.append(1)
                intTimes.append(period)
                upTimes.append(period)
                intTimesSeq.append(list(range(1,period+1)))
                seqState = 0
                
            #generates off sequence
            else:
                #generates random off period
                if downDist=="exp":
                    period = m.ceil(rnd.expovariate(lambda2))
                elif downDist=="erl":
                    period = m.ceil(rnd.gammavariate(kParam2,1/lambda2)) #assumes k=2
                elif downDist=="lnorm":
                    period = m.ceil(rnd.lognormvariate(np.log(((1/lambda2)**2)/np.sqrt((1/var2)+(1/lambda2)**2)),np.sqrt(np.log(1/var2)/(((1/lambda2)**2)+1))))

                #period = 10
                
                if (totalTime+period) > N: #makes sure total time isn't exceeded
                    occupancy[totalTime:N] = [0]*(N-totalTime)
                else: #appends proper sequence of 1s
                    occupancy[totalTime:totalTime+period] = [0]*period
                
                #tracks state transitions and on/off durations    
                stateTrans.append(0)
                intTimes.append(period)
                downTimes.append(period)
                intTimesSeq.append(list(range(1,period+1)))
                seqState = 1
                
            totalTime += period
            
        seqSize = len(stateTrans) #total number of on and off states
        traffic_intensity = sum(occupancy)/N #measures traffic intensity
#        #measures mean signal interarrival
#        mean_int = sum(intTimes[0:seqSize-(seqSize%2)]) / ((seqSize-(seqSize%2))/2) 
#        actual_int = 1/lambda1+1/lambda2 #calculates theoretical interarrival
#        
#        #reactive predictor "accuracy/error"
#        predicted = occupancy[0:N-1]
#        #theoretical accuracy based on lambda parameters
#        theoAcc = 1-(2/actual_int-1/N)
#        #accuracy based on measured mean interarrival
#        expAcc = 1-(2/mean_int-1/N)
#        #observed accuracy
#        obsAcc = sum([predicted[i]==occupancy[i+1] for i in range(N-1)]) / (N-1)
        
        
        ###input RF signal generation###
        dLen = 100 #length of the energy detector
        fs = 100e6
        time = np.linspace(0,N*dLen/fs,N*dLen)

        amp = m.sqrt((10**(power/10))/1000 * (2*50)) #sinusoid amplitude
        '''
        noiseVar - the noise floor is the measure of the signal created from the sum of all
        the noise sources and unwanted signals within a measurement system, where
        noise is defined as any signal other than the one being monitored.         
        '''
        noiseVar = 1e-6 #noisefloor variance (1e-7 places noisefloor around -100 dBm)
        #noisefloor = [m.sqrt(noiseVar)*rnd.gauss(0,1) for i in range(N*dLen)]
        noisefloor = m.sqrt(noiseVar)*np.random.randn(N*dLen)
        sineWave = amp*np.exp(1j*2*pi*(10e6)*time) #sine wave at 10 MHz
        #SNR of the signal
        SNR = 10*np.log10((sum(np.abs(sineWave)**2)/(dLen*N))/(sum(np.abs(noisefloor)**2)/(dLen*N)))

        #Modulates the sine wave with the occupancy state where each state has dLen samples
        occSwitch = np.repeat(occupancy,dLen)
        inputRF = sineWave*occSwitch+noisefloor
        
        P_fa = 0.01 #probability of false alarm
        #energy detector threshold
        thresh = noiseVar/m.sqrt(dLen)*(-norm.ppf(P_fa))+noiseVar
        print("Threshold",thresh)
        #calculates total average power over a sliding window
        totalAvgPwr = np.zeros((dLen*N-dLen+1))
        totalAvgPwr_cumulants = np.zeros((dLen*N-dLen+1))
        trueState = np.zeros((dLen*N-dLen+1))
        #pwrStates = np.zeros((dLen, dLen*N-dLen+1))

        
        for i in range(dLen*N-dLen+1):
            cumulant = stats.kstat(np.abs(inputRF[i:i+dLen-1]),1)
            totalAvgPwr_cumulants.itemset(i,cumulant)
            totalAvgPwr[i] = sum(np.abs(inputRF[i:i+dLen])**2)/dLen
            #pwrStates[:,i] = np.arange(i,i+dLen)
            trueState[i] = int(sum(occSwitch[i:i+dLen]) > 0) 

        #return totalAvgPwr,totalAvgPwr_cumulants, SNR,thresh;
        return totalAvgPwr

#arp = ARPSimulator()
#arp.generateFreqEnergy(0.2,0.2,500,-55)
