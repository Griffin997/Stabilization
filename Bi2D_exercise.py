#This is the python script for replicating the Bi,Spencer 2D stabilization technique

#Preparing all libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import addcopyfighandler

#Initial Options
histPlot = True

#Assumed time points of interest
tdata = np.arange(8, 512, 8) #ms units

#Defining the function of interest
def biExp2D(tdata,c1, c2, T21, T22):
    exp1 = c1*np.exp(-tdata/T21)
    exp2 = c2*np.exp(-tdata/T22)
    return exp1 + exp2

#How many iterations are we going to do of this fitting test
iterCount = 1000
ParamTitle = ['c1','c2','T21','T22']
paramStore = np.zeros([iterCount,np.size(ParamTitle)])
SNRStore = np.zeros(iterCount)

#Parameter varied to observe identifiability
T22_range = np.arange(50,51,1)

stdStore = np.zeros([np.size(T22_range),np.size(ParamTitle)])

#Parameters held constant
c1 = 0.3
c2 = 0.7
T21 = 60


for j in range(np.size(T22_range)):

    T22 = T22_range[j]

    realParams = (c1,c2,T21,T22)

    trueDat = biExp2D(tdata,*realParams)

    for i in range(iterCount):

        #Determining the noise
        noiseSigma = np.mean(trueDat)/800
        noise = np.random.normal(0,noiseSigma,tdata.size)

        noiseDat = trueDat + noise

        #More accurate SNR calculation - found online
        # np.sum(np.abs(np.fft.fft(cleanSound,sampling_rate//2)/Nsamples)**2)
        # mSNR = 10*np.log10(trueDat,noiseDat)

        #Experimental Signal to Noise Ratio Calculation
        mSNR = noiseDat/noise
        avgMSNR = np.absolute(np.mean(mSNR))
        SNRStore[i] = avgMSNR

        popt,pcov = curve_fit(biExp2D, tdata, noiseDat, bounds = [(0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)])

        if (popt[0] > popt[1]):
            p_hold = popt[0]
            popt[0] = popt[1]
            popt[1] = p_hold
            p_hold = popt[2]
            popt[2] = popt[3]
            popt[3] = p_hold

        # popt = popt.shape(1,4) #reshaping popt if necessary

        paramStore[i,:] = popt
        
    runStd = np.std(paramStore, axis = 1)

    stdStore[j,:] = runStd

    # fig, ax = plt.subplots(2,2)

if histPlot:
    for i in range(2):
        plt.hist(x=paramStore[:,2*i], bins='auto', color='#0504aa')
        plt.hist(x=paramStore[:,2*i+1], bins='auto', color='#454B1B')
        plt.xlabel('Param Value')
        plt.ylabel('Count')
        plt.title(ParamTitle[2*i] + ' against ' + ParamTitle[2*i+1])
        plt.show()

