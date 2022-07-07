#This is the python script for replicating the Bi,Spencer 2D stabilization technique

#Preparing all libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import addcopyfighandler

#Assumed time points of interest
TE = np.arange(8, 512, 8) #ms units
TI = np.arange(50,4850,200) #ms units

#Defining the function of interest
def biExp2D(TI,TE,c1, T11, T21, c2, T12, T22):
    exp1 = c1*(1-2*np.exp(-TI/T11))*np.exp(-TE/T21)
    exp2 = c2*(1-2*np.exp(-TI/T12))*np.exp(-TE/T22)
    return exp1 + exp2

#How many iterations are we going to do of this fitting test
iterCount = 1000
ParamTitle = ['c1','c2','T21','T22']
paramStore = np.zeros([iterCount,np.size(ParamTitle)])
SNRStore = np.zeros(iterCount)

#Parameter varied to observe identifiability
T12 = 800

#Parameters held constant
c1 = 0.3
c2 = 0.7
T11 = 1000
T21 = 60
T22 = 45



realParams = (c1,T11,T21,c2,T12,T22)

trueDat = biExp2D(TI,TE,*realParams)

plt.plot(TI,TE,trueDat)
plt.show()


# for i in range(iterCount):

#     #Determining the noise
#     noiseSigma = np.mean(trueDat)/800
#     noise = np.random.normal(0,noiseSigma,tdata.size)

#     noiseDat = trueDat + noise

#     #More accurate SNR calculation - found online
#     # np.sum(np.abs(np.fft.fft(cleanSound,sampling_rate//2)/Nsamples)**2)
#     # mSNR = 10*np.log10(trueDat,noiseDat)

#     #Experimental Signal to Noise Ratio Calculation
#     mSNR = noiseDat/noise
#     avgMSNR = np.absolute(np.mean(mSNR))
#     SNRStore[i] = avgMSNR

#     popt,pcov = curve_fit(biExp2D, tdata, noiseDat, bounds = [(0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)])

#     if (popt[0] > popt[1]):
#         p_hold = popt[0]
#         popt[0] = popt[1]
#         popt[1] = p_hold
#         p_hold = popt[2]
#         popt[2] = popt[3]
#         popt[3] = p_hold

#     # popt = popt.shape(1,4) #reshaping popt if necessary

#     paramStore[i,:] = popt


# # fig, ax = plt.subplots(2,2)

# for i in range(2):
#     plt.hist(x=paramStore[:,2*i], bins='auto', color='#0504aa')
#     plt.hist(x=paramStore[:,2*i+1], bins='auto', color='#454B1B')
#     plt.xlabel('Param Value')
#     plt.ylabel('Count')
#     plt.title(ParamTitle[2*i] + ' against ' + ParamTitle[2*i+1])
#     plt.show()

