#This is the python script for replicating the Bi,Spencer 2D stabilization technique

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
T11 = 400
T12 = 800
T21 = 60
T22 = 45

T11star = np.log(2)*T11
T12star = np.log(2)*T12

print(T11star)
print(T12star)

realParams = (c1,T11,T21,c2,T12,T22)

TImg,TEmg = np.meshgrid(TI,TE)
truDat = biExp2D(TImg,TEmg,*realParams)

xdata = np.stack((TImg,TEmg), axis=2).reshape(-1, 2)
ydata = np.hstack(truDat)

zMax = np.max(abs(truDat)*1.5)

(TEpat,sigPat) = np.meshgrid(np.linspace(8,512,2),np.linspace(-zMax,zMax,2))
T1pat = np.full_like(TEpat,T11star)
T2pat = np.full_like(TEpat,T12star)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(TImg,TEmg,truDat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_surface(T1pat,TEpat,sigPat,cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)
ax.plot_surface(T2pat,TEpat,sigPat,cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)
plt.xlabel('TI')
plt.ylabel('TE')
plt.title('Signal')
plt.show()

diffT11 = (TI-T11star)**2
indexStarT11 = np.argmin(diffT11)

diffT12 = (TI-T12star)**2
indexStarT12 = np.argmax(diffT12)

plt.plot(TE,biExp2D(T11star*2,TE,*realParams))
plt.plot(TE,biExp2D(T11star,TE,*realParams))
plt.plot(TE,biExp2D(np.mean([T11star,T12star]),TE,*realParams))
plt.plot(TE,biExp2D(T12star,TE,*realParams))
plt.plot(TE,biExp2D(T12star/2,TE,*realParams))
plt.legend(['above T11*','T11*','between T11* and T12*','T12*','below T12*'])
plt.show()