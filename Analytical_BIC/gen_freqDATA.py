############# Libaries ###############

import scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import statistics
import math
import time
import itertools
from itertools import product, zip_longest
import pickle
from tqdm import tqdm, trange
from datetime import date


import multiprocess as mp
from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
import functools

######All Fixed parameters for code
#Parameters held constant
#Parameters held constant
T11 = 92 #600
T12 = 353 #1200
c1 = 0.5 #0.4
c2 = 0.5 #0.6 
T21 = 31 #40
T22 = 41 #100

true_params = np.array([T11, T12, c1, c2, T21, T22])

multi_starts_BIC = 3


#Building the TE array - this should be a uniform array
n_TE = 64
TE_step = 8

TE_DATA = np.linspace(TE_step, TE_step*n_TE, n_TE) #ms units

#Adjusting the ratio of T21 and T22
# T2rat_array = [2.5]#np.arange(1.5, 2.51, 0.1)

#SNR Values to Evaluate
SNR_array = [100]#10**np.linspace(np.log10(25), np.log10(250), 15)

repetition = 1000

## TI index
TI_STANDARD = np.arange(0,3601,1)

ParamTitle_6p = [r'$T_{11}$', r'$T_{12}$', r'$c_1$', r'$c_2$', r'$T_{21}$', r'$T_{22}$']

target_iterator = [(a,b) for a in TI_STANDARD for b in SNR_array]

num_cpus_avail = np.min([len(target_iterator),4])

#### Important for Naming
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

data_path = "Analytical_BIC/Frequency_DATA"
add_tag = "standard"
data_tag = (f"BICfreq_{add_tag}_SNRsuite_{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)

#### Signaling Functions
def S_biX_6p(TE, T11, T12, c1, c2, T21, T22, TI_val = 0):
    exp1 = c1*(1-2*np.exp(-TI_val/T11))*np.exp(-TE/T21)
    exp2 = c2*(1-2*np.exp(-TI_val/T12))*np.exp(-TE/T22)
    return exp1 + exp2

#The one dimensional models 
def S_biX_4p(TE, d1, d2, T21, T22):
    exp1 = d1*np.exp(-TE/T21)
    exp2 = d2*np.exp(-TE/T22)
    return exp1 + exp2

def S_moX_2p(TE, d, T2):
    return d*np.exp(-TE/T2)

def calc_BIC(RSS, TE_array, p_list, sigma):
    BIC = 1/len(TE_array) * (RSS + np.log(len(TE_array))*len(p_list)*(sigma)**2)

    return BIC

def get_func_bounds(func):
    f_name = func.__name__
    if f_name == "S_biX_6p":
        lower_bound = (0, 0, 0, 0, 0, 0)
        upper_bound = (2000, 2000, 1, 1, 150, 150)
    elif f_name == "S_moX_3p":
        lower_bound = (0, 0, 0)
        upper_bound = (2000, 1, 150)
    elif f_name == "S_biX_4p":
        lower_bound = (-1, -1, 0, 0)
        upper_bound = (1, 1, 150, 150)
    elif f_name == "S_moX_2p":
        lower_bound = (-1, 0)
        upper_bound = (1, 150)
    else:
        raise Exception("Not a valid function: " + f_name)

    return lower_bound, upper_bound

def set_p0(func, random = True):
    true_params = [T11, T12, c1, c2, T21, T22]
    if random:
        lb, ub = get_func_bounds(func)
        if func.__name__.find("S_biX_6p") > -1:
            T11_est = np.random.uniform(lb[-6],ub[-6])
            T12_est = np.random.uniform(T11_est,ub[-5])
            c1_est = np.random.uniform(lb[-4],ub[-4])
            T21_est = np.random.uniform(lb[-2],ub[-2])
            T22_est = np.random.uniform(T21_est,ub[-1])
            p0 = [T11_est, T12_est, c1_est, 1-c1_est, T21_est, T22_est]
        else:
            p0 = [np.random.uniform(lb[i],ub[i]) for i in range(len(lb))]
    else:
        # f_name = func.__name__
        # if f_name.find("moX") > -1:
        #     p0 = [75, 0.5, 75]
        # elif f_name.find("biX") > -1:
        #     p0 = [75, 75, 0.5, 0.5, 75, 75]
        # else:
        p0 = true_params
            
    return p0

def check_param_order(popt):
    #Reshaping of array to ensure that the parameter pairs all end up in the appropriate place - ensures that T22 > T21
    if (popt[-1] < popt[-2]): #We want by convention to make sure that T21 is <= T22
        for pi in range(np.size(popt)//2):
            p_hold = popt[2*pi]
            popt[2*pi] = popt[2*pi+1]
            popt[2*pi+1] = p_hold
    return popt

#All curves get noise according to this equation
def add_noise(data, SNR):
    #returns a noised vector of data using the SNR given
    sigma = (c1+c2)/SNR #np.max(np.abs(data))/SNR
    noise = np.random.normal(0,sigma,data.shape)
    noised_data = data + noise
    return noised_data

def estP_oneCurve(func, noisey_data, TI_val = 0):

    f_name = func.__name__
    init_p = set_p0(func, random = True)
    lb, ub = get_func_bounds(func)

    if f_name.find("6p") > -1:
        popt, _, info_popt, _, _ = curve_fit(functools.partial(func, TI_val = TI_val), TE_DATA, noisey_data, p0 = init_p, bounds = [lb,ub], method = 'trf', maxfev = 1500, full_output = True)
    else:
        popt, _, info_popt, _, _ = curve_fit(func, TE_DATA, noisey_data, p0 = init_p, bounds = [lb,ub], method = 'trf', maxfev = 1500, full_output = True)
    popt = check_param_order(popt)
    RSS = np.sum(info_popt['fvec']**2)

    return popt, RSS

def evaluate_model(data, SNR, ms_iter = multi_starts_BIC):

    #Returns true if the moX is better than the biX --> returns a 1 for moX and a 0 for biX
    #Returns the parameters of the correct answer

    RSS_biX = np.inf
    RSS_moX = np.inf

    for ms in range(ms_iter):
        popt_biX_temp, RSS_biX_temp = estP_oneCurve(S_biX_4p, data)
        popt_moX_temp, RSS_moX_temp = estP_oneCurve(S_moX_2p, data)

        if RSS_biX_temp < RSS_biX:
            popt_biX = popt_biX_temp
            RSS_biX = RSS_biX_temp

        if RSS_moX_temp < RSS_moX:
            popt_moX = popt_moX_temp
            RSS_moX = RSS_moX_temp

    # popt_biX, RSS_biX = estP_oneCurve(S_biX_4p, data)
    # popt_moX, RSS_moX = estP_oneCurve(S_moX_2p, data)

    BIC_biX = calc_BIC(RSS_biX, TE_DATA, popt_biX, 1/SNR)
    BIC_moX = calc_BIC(RSS_moX, TE_DATA, popt_moX, 1/SNR)

    if BIC_moX < BIC_biX:
        return True, popt_moX, RSS_moX
    else:
        return False, popt_biX, RSS_biX
    
def calculate_frequency(i_param_combo):
    
    TI_value, SNR_value = target_iterator[i_param_combo]
    data = S_biX_6p(TE_DATA, *true_params, TI_val = TI_value)
    counter = 0
    for rep in range(repetition):
        noisy_curve = add_noise(data, SNR_value)
        moX_opt, _, _ = evaluate_model(noisy_curve, SNR_value)
        if moX_opt:
            counter += 1

    feature_df = pd.DataFrame(columns = ["frequency", "TI", "SNR"])        
    feature_df['SNR'] = [SNR_value]
    feature_df['TI'] = [TI_value]
    feature_df['frequency'] = counter/repetition

    return feature_df

#### Looping through Iterations of the brain - applying parallel processing to improve the speed
if __name__ == '__main__':
    freeze_support()

    print("Finished Assignments...")

    ##### Set number of CPUs that we take advantage of
    
    if num_cpus_avail >= 10:
        print("Using Super Computer")

    lis = []

    with mp.Pool(processes = num_cpus_avail) as pool:

        with tqdm(total = len(target_iterator)) as pbar:
            for estimates_dataframe in pool.imap_unordered(calculate_frequency, range(len(target_iterator))):
            
                lis.append(estimates_dataframe)

                pbar.update()

        pool.close()
        pool.join()
    

    print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    df = pd.concat(lis, ignore_index= True)

    df.to_pickle(data_folder + f'/' + data_tag +'.pkl')

############## Save General Code Code ################

hprParams = {
    "SNR_array": SNR_array,         #fourth iterator
    "true_params": true_params,
    "nTE": n_TE,
    "dTE": TE_step,
    "var_reps": repetition,
    "TI_range": TI_STANDARD
}

f = open(f'{data_folder}/hprParameter_{add_tag}_SNRsuite_{day}{month}{year}.pkl','wb')
pickle.dump(hprParams,f)
f.close()