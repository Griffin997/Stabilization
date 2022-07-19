#This is the python script for replicating the Bi,Spencer 2D stabilization technique
#with the intention of varying TI for Eq.3 to observe the TI related nullpoint - this
#code has the potential to look at a histogram for each TI value or to look at the
#standard deviation across the population for a range of TI values

#Preparing all libraries
from lzma import CHECK_SHA256
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import addcopyfighandler

#Initial Options
histPlot = False
stdPlot = True

#Assumed echo time points
tdata = np.arange(8, 512, 8) #ms units

######All Fixed parameters for code
#Parameters held constant
c1 = 0.3
c2 = 0.7
T21 = 60
T11 = 400
T12 = 1200
#Parameter varied to observe identifiability
# T22_range = np.arange(40,71,2.5)
T22_range = [20]
#Information determing TI range
TI_numPoints = 101
TI_lb = 0.5
TI_ub = 1.5
assert(TI_ub>TI_lb)
#Process related parameters
iterCount = 1000 #number of iterations that curveFit is applied
SNR = 10000


#Preparing TI array to loop through
TI_array = np.linspace(TI_lb,TI_ub,TI_numPoints)
TI1star = np.log(2)*(T11)
TI_array = TI_array*TI1star

#Defining the function of interest
def biExp2D(tdata, TI, T11, T12, c1, c2, T21, T22):
    exp1 = c1*(1-2*np.exp(-TI/T11))*np.exp(-tdata/T21)
    exp2 = c2*(1-2*np.exp(-TI/T12))*np.exp(-tdata/T22)
    return exp1 + exp2

#How many iterations are we going to do of this fitting test
ParamTitle = ['c1','c2','T21','T22']
paramStore = np.zeros([iterCount,np.size(ParamTitle)])
SNRStore = np.zeros([iterCount,np.size(TI_array)])
stdStore = np.zeros([np.size(TI_array),np.size(ParamTitle)])
std2Store = np.zeros([np.size(TI_array),np.size(ParamTitle)])
# cNquickStore = np.zeros([np.size(TI_array),np.size(ParamTitle)]) #To be implemented later

#Looping through all TI values in the TI_array
for k in range(np.size(TI_array)):

    TI = TI_array[k]

    true_d1 = c1*(1-2*np.exp(-TI/T11))
    true_d2 = c2*(1-2*np.exp(-TI/T12))

    #Looping through all T22 values in the T22_range
    for j in range(np.size(T22_range)):

        T22 = T22_range[j]

        realParams = [c1, c2, T21, T22]

        trueDat = biExp2D(tdata,TI,T11,T12,*realParams)

        for i in range(iterCount):

            #Determining the noise and add noise to data
            noiseSigma = 1/SNR
            noise = np.random.normal(0,noiseSigma,tdata.size)
            noiseDat = trueDat + noise

            #Experimental Signal to Noise Ratio Calculation
            mSNR = noiseDat/noise
            avgMSNR = np.absolute(np.mean(mSNR))
            SNRStore[i,k] = avgMSNR

            lb = (0,0,0,0)
            ub = (1,1,np.inf,np.inf)

            init_c1 = np.random.uniform(lb[0],ub[0])
            init_c2 = np.random.uniform(init_c1,ub[1])
            init_T21 = np.random.randint(2,100)
            init_T22 = np.random.randint(1,init_T21)
            init_p = [init_c1,init_c2,init_T21,init_T22]

            popt,pcov = curve_fit(lambda t_dat,p1,p2,p3,p4 : biExp2D(t_dat,TI,T11,T12,p1,p2,p3,p4), tdata, noiseDat, p0 = init_p, bounds = [lb,ub])

            #Reshaping of array to ensure that the parameter pairs all end up in the appropriate place
            if (popt[0] > popt[1] and popt[3] > popt[2]):
                p_hold = popt[0]
                popt[0] = popt[1]
                popt[1] = p_hold
                p_hold = popt[2]
                popt[2] = popt[3]
                popt[3] = p_hold

            paramStore[i,:] = popt
            
        

        if histPlot:
            fig, ax = plt.subplots(1,2, figsize=(9.5,5))
            for i in range(2):

                #Get a good bin size
                binData = paramStore[:,[2*i,2*i+1]]
                binData = np.reshape(binData,-1)
                binW = (np.max(binData) - np.min(binData))/10
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
                ax[i].set_title('Parameter Histogram Comparison\n' + 'TI1* = ' + str(round(TI1star,2)) + ' :: TI value (' + str(k) + '): ' + str(round(TI,2)) +
                            '\nTrue ' + ParamTitle[2*i] + '=' + str(round(realParams[2*i],2)) + ' :: ' 
                            'True ' + ParamTitle[2*i+1] + '=' + str(round(realParams[2*i+1],2)))
            plt.show()
    #Collect the standard deviation in parameter values
    runStd = np.std(paramStore, axis = 0)
    stdStore[k,:] = runStd

    std2 = np.var(paramStore,axis = 0)**(1/2)
    std2Store[k,:] = std2


if stdPlot:
    for i in range(np.size(ParamTitle)):
        plt.plot(TI_array,stdStore[:,i], label = ParamTitle[i])
    plt.axvline(x=TI1star, linewidth=1, label= 'TI1 nullpoint', color='k')
    plt.xlabel('TI Value')
    plt.ylabel('Standard Deviation of Parameter')
    plt.title('TI Influence on Parameter Standard Deviation')
    plt.legend()
    plt.show()

    # Rely on Variance to calculate distribution
    # for i in range(np.size(ParamTitle)):
    #     plt.plot(TI_array,std2Store[:,i], label = ParamTitle[i])
    # plt.xlabel('TI Value')
    # plt.ylabel('sqrt(var) of Parameter')
    # plt.title('TI Influence on ParameterStandard Deviation')
    # plt.legend()
    # plt.show()