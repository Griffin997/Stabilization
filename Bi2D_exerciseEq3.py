#This is the python script for replicating the Bi,Spencer 2D stabilization technique
#Thisfile makes it possible to estimate parameter values for the two dimensional and two
#compartamental model

#Preparing all libraries
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
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

#Defining the function of interest
def biExp2D_cf(X,c1, T11, T21, c2, T12, T22):
    TI, TE = np.hsplit(X, 2)
    exp1 = c1*(1-2*np.exp(-TI/T11))*np.exp(-TE/T21)
    exp2 = c2*(1-2*np.exp(-TI/T12))*np.exp(-TE/T22)
    result = np.hstack(exp1+exp2)
    return result

#How many iterations are we going to do of this fitting test
iterCount = 1000
ParamTitle = ['c1','T11','T21','c2','T12','T22']
paramStore = np.zeros([iterCount,np.size(ParamTitle)])
SNRStore = np.zeros(iterCount)

#Parameters held constant
c1 = 0.3
c2 = 0.7
T11 = 1000
T12 = 500
T21 = 60
T22 = 40

realParams = (c1,T11,T21,c2,T12,T22)

TI,TE = np.meshgrid(TI,TE)
truDat = biExp2D(TI,TE,*realParams)

xdata = np.stack((TI,TE), axis=2).reshape(-1, 2)
ydata = np.hstack(truDat)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(TI,TE,truDat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('TI')
plt.ylabel('TE')
plt.title('Signal')
plt.show()


for i in range(iterCount):

    #Determining the noise
    SNR = 800
    # noiseSigma = 1/SNR
    noiseSigma = np.mean(ydata)/SNR
    noise = np.random.normal(0,noiseSigma,np.size(ydata))

    noiseDat = ydata + noise

    #Experimental Signal to Noise Ratio Calculation
    mSNR = noiseDat/noise
    avgMSNR = np.absolute(np.mean(mSNR))
    SNRStore[i] = avgMSNR

    popt,pcov = curve_fit(biExp2D_cf, xdata, noiseDat, bounds = [(0,0,0,0,0,0),(1,np.inf,np.inf,1,np.inf,np.inf)])

    if (popt[0] > popt[3]):
        p_hold1 = popt[0]
        p_hold2 = popt[1]
        p_hold3 = popt[2]
        popt[0] = popt[3]
        popt[1] = popt[4]
        popt[2] = popt[5]
        popt[3] = p_hold1
        popt[4] = p_hold2
        popt[5] = p_hold3

    # popt = popt.shape(1,4) #reshaping popt if necessary

    paramStore[i,:] = popt


# fig, ax = plt.subplots(2,2)

for i in range(np.size(ParamTitle)//2):
    plt.hist(x=paramStore[:,i], bins='auto', color='#0504aa')
    plt.hist(x=paramStore[:,i+np.size(ParamTitle)//2], bins='auto', color='#454B1B')
    plt.xlabel('Param Value')
    plt.ylabel('Count')
    plt.title(ParamTitle[i] + ' against ' + ParamTitle[i+np.size(ParamTitle)//2])
    plt.show()

