#This is the python script for replicating the Bi,Spencer 2D stabilization technique
#in only a single dimension of TE time. The intention of this experiment is to look 
#at how varying TE translates into an identifiability issue - we are replicating the 
# figures from the Bi, Spencer paper that show histograms of the estimated values - this
#code has the potential to look at a histogram for each TE value or to look at the
#standard deviation across the population for a range of TE values

#Preparing all libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import addcopyfighandler

#Initial Options
histPlot = True
stdPlot = False

###Parameter Control

tdata = np.arange(8, 512, 8)    #Echo Time in ms units
SNR = 10000                       #Signal to Noise Ratio
c1 = 0.05                        #Amplitude 1
c2 = 1-c1                       #Amplitude 2
T21 = 60                        #T2 for amplitude 1
iterCount = 1000                 #How many times will curve_fit be run

#Parameter varied to observe identifiability
# T22_range = np.arange(40,71,2.5)
T22_range = [45]                #T2 for amplitude 2

#Defining the function of interest
def biExp2D(tdata,c1, c2, T21, T22):
    exp1 = c1*np.exp(-tdata/T21)
    exp2 = c2*np.exp(-tdata/T22)
    return exp1 + exp2


#Initializing all arrays
ParamTitle = ['c1','c2','T21','T22']
paramStore = np.zeros([iterCount,np.size(ParamTitle)])
SNRStore = np.zeros(iterCount)
stdStore = np.zeros([np.size(T22_range),np.size(ParamTitle)])


for j in range(np.size(T22_range)):

    T22 = T22_range[j]

    realParams = (c1,c2,T21,T22)

    trueDat = biExp2D(tdata,*realParams)

    for i in range(iterCount):

        #Determining the noise and add noise to data
        noiseSigma = 1/SNR
        noise = np.random.normal(0,noiseSigma,tdata.size)
        noiseDat = trueDat + noise

        #Experimental Signal to Noise Ratio Calculation
        mSNR = noiseDat/noise
        avgMSNR = np.absolute(np.mean(mSNR))
        SNRStore[i] = avgMSNR

        lb = (0,0,0,0)
        ub = (1,1,np.inf,np.inf)

        init_c1 = np.random.uniform(lb[0],ub[0])
        init_c2 = np.random.uniform(init_c1,ub[1])
        init_T21 = np.random.randint(2,100)
        init_T22 = np.random.randint(1,init_T21)
        init_p = [init_c1,init_c2,init_T21,init_T22]

        popt,pcov = curve_fit(biExp2D, tdata, noiseDat, p0 = init_p, bounds = [lb,ub])

        if (popt[0] > popt[1] and popt[3] > popt[2]):
            p_hold = popt[0]
            popt[0] = popt[1]
            popt[1] = p_hold
            p_hold = popt[2]
            popt[2] = popt[3]
            popt[3] = p_hold

        #Collect all parameter values to compare
        paramStore[i,:] = popt
        
    #Collect the standard deviation in parameter values
    runStd = np.std(paramStore, axis = 0)
    stdStore[j,:] = runStd

    if histPlot:
        fig, ax = plt.subplots(1,2, figsize=(9.5,5))
        for i in range(2):

            #Get a good bin size
            binData = paramStore[:,[2*i,2*i+1]]
            binData = np.reshape(binData,-1)
            binW = (np.max(binData) - np.min(binData))/50
            binW = float('%.1g' % binW) #converts everything to a single significant figure
            binMin = 10**(np.floor(np.log10(np.min(binData))))
            binArray = np.arange(binMin, np.max(binData) + binW, binW)

            #Construct a nice figure of each of the two 
            ax[i].hist(x=paramStore[:,2*i], bins='auto', color='b', label = ParamTitle[2*i], alpha = 0.7)
            ax[i].hist(x=paramStore[:,2*i+1], bins='auto', color='g', label = ParamTitle[2*i+1], alpha = 0.7)
            ax[i].set_xlabel('Param Value')
            ax[i].set_ylabel('Count')
            ax[i].axvline(x=realParams[2*i], linewidth=1, label= 'True ' + ParamTitle[2*i], color='red')
            ax[i].axvline(x=realParams[2*i+1], linewidth=1, label= 'True ' + ParamTitle[2*i+1], color='orange')
            ax[i].legend()
            ax[i].set_title('Parameter Histogram Comparison\n' +
                        '\nTrue ' + ParamTitle[2*i] + '=' + str(round(realParams[2*i],2)) + ' :: ' 
                        'True ' + ParamTitle[2*i+1] + '=' + str(round(realParams[2*i+1],2)))
        plt.show()


if stdPlot:
    plt.plot(T22_range,stdStore[:,0])
    plt.plot(T22_range,stdStore[:,1])
    plt.plot(T22_range,stdStore[:,2])
    plt.plot(T22_range,stdStore[:,3])
    plt.xlabel('T_{2,2} Value')
    plt.ylabel('Standard Deviation')
    plt.title('T_{2,2} Influence on Standard Deviation')
    plt.legend(['c1','c2','T_{2,1}','T_{2,2}'])
    plt.show()