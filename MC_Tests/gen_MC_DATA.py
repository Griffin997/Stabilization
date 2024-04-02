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

####### Options #######
randStart = True                  #Initial guess for parameter values in random locations

############# Global Params ###############

######All Fixed parameters for code
#Parameters held constant
T11 = 600
T12 = 1200
c1 = 0.4
c2 = 0.6 
T21 = 40
T22 = 100

true_params = np.array([T11, T12, c1, c2, T21, T22])

#Building the TE array - this should be a uniform array
n_TE = 64
TE_step = 8

TE_DATA = np.linspace(TE_step, TE_step*n_TE, n_TE) #ms units

TI_low = 250
TI_high = 1050
TI_res = 0.5

TI_DATA = np.arange(TI_low,TI_high+TI_res,TI_res)

TI1star = np.log(2)*T11
TI2star = np.log(2)*T12

assert(TI_low < TI1star and TI1star < TI_high)
assert(TI_low < TI2star and TI2star < TI_high)

#SNR Values to Evaluate
SNR_value = 500

#Number of noisy realizations
var_reps = 10000

#Number of tasks to execute
target_iterator = [(a,b) for a in TI_DATA for b in range(var_reps)]

#### Important for Naming
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

num_cpus_avail = np.min([len(target_iterator),40])
data_path = "InversionExp_Characteristics/MC_DATA"
add_tag = ""
data_head = "trueStart"
data_tag = (f"{data_head}_SNR{SNR_value}_iter{var_reps}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)

############# Functions ##############

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

#Function for calculating the d coeficient for a TI, c, T1 collection
def d_value(TI,c,T1):
    return c*(1-2*np.exp(-TI/T1))

#All curves get noise according to this equation
def add_noise(data, SNR):
    #returns a noised vector of data using the SNR given
    sigma = (c1+c2)/SNR #np.max(np.abs(data))/SNR
    noise = np.random.normal(0,sigma,data.shape)
    noised_data = data + noise
    return noised_data

#### Fitting helping functions

def get_func_bounds(func):
    f_name = func.__name__
    if f_name.find("6p") > -1:
        lower_bound = (0, 0, 0, 0, 0, 0)
        upper_bound = (2000, 2000, 1, 1, 150, 150)
    elif f_name.find("4p") > -1:
        lower_bound = (-1, -1, 0, 0)
        upper_bound = (1, 1, 150, 150)
    else:
        raise Exception("Not a valid function: " + f_name)

    return lower_bound, upper_bound

def get_param_list(func):
    f_name = func.__name__
    if f_name.find("6p") > -1:
        params = ("T11","T12","c1","c2","T21","T22")
    elif f_name.find("4p") > -1:
        params = ("d1","d2","T21","T22")
    else:
        raise Exception("Not a valid function: " + f_name)

    return params

def set_p0(func, TI = 0):
    f_name = func.__name__
    if f_name.find("6p") > -1:
        p0 = true_params
    else:
        p0 = [d_value(TI,c1,T11), d_value(TI,c2,T12), T21, T22]
            
    return p0

def check_param_order(popt):
    #Reshaping of array to ensure that the parameter pairs all end up in the appropriate place - ensures that T22 > T21
    if (popt[-1] < popt[-2]): #We want by convention to make sure that T21 is <= T22
        for pi in range(np.size(popt)//2):
            p_hold = popt[2*pi]
            popt[2*pi] = popt[2*pi+1]
            popt[2*pi+1] = p_hold
    return popt

def estP_oneCurve(func, TI_val, noisey_data):

    f_name = func.__name__
    init_p = set_p0(func, TI = TI_val)
    lb, ub = get_func_bounds(func)

    if f_name.find("6p") > -1:
        popt, _, info_popt, _, _ = curve_fit(functools.partial(func, TI_val = TI_val), TE_DATA, noisey_data, p0 = init_p, bounds = [lb,ub], method = 'trf', maxfev = 1500, full_output = True)
    else:
        popt, _, info_popt, _, _ = curve_fit(func, TE_DATA, noisey_data, p0 = init_p, bounds = [lb,ub], method = 'trf', maxfev = 1500, full_output = True)
    popt = check_param_order(popt)
    RSS = np.sum(info_popt['fvec']**2)

    return popt, RSS

#### Parallelized Function

def generate_all_estimates(i_param_combo):
    #Generates a comprehensive matrix of all parameter estimates for all param combinations, 
    #noise realizations, SNR values, and lambdas of interest
    TI_value, nr = target_iterator[i_param_combo]

    feature_df = pd.DataFrame(columns = ["TI","NR","TD_params","OD_params", "TD_RSS", "OD_RSS"])

    feature_df["TI"] = [TI_value]
    feature_df["NR"] = [nr]

    #Generate signal array from temp values
    true_signal = S_biX_6p(TE_DATA, *true_params, TI_val = TI_value)
    noised_signal = add_noise(true_signal, SNR_value)

    param_6p, RSS_6p = estP_oneCurve(S_biX_6p, TI_value, noised_signal)
    param_4p, RSS_4p = estP_oneCurve(S_biX_4p, TI_value, noised_signal)


    feature_df['TD_params'] = [param_6p]
    feature_df['TD_RSS'] = [RSS_6p]
    feature_df['OD_params'] = [param_4p]
    feature_df['OD_RSS'] = [RSS_4p]

    return feature_df



#### Looping through Iterations of the brain - applying parallel processing to improve the speed
if __name__ == '__main__':
    freeze_support()

    print("Finished Assignments...")  

    lis = []

    with mp.Pool(processes = num_cpus_avail) as pool:

        with tqdm(total = len(target_iterator)) as pbar:
            for estimates_dataframe in pool.imap_unordered(generate_all_estimates, range(len(target_iterator))):
            
                lis.append(estimates_dataframe)

                pbar.update()

        pool.close()
        pool.join()
    

    print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    df = pd.concat(lis, ignore_index= True)

    df.to_pickle(data_folder + f'/' + data_tag +'.pkl')     

############## Save General Code Code ################

hprParams = {
    "SNR_value": SNR_value,           #third iterator
    "true_params": true_params,
    "TI_DATA": TI_DATA,
    "nTE": n_TE,
    "dTE": TE_step,
    "var_reps": var_reps,
}

f = open(f'{data_folder}/hprParameter_{data_head}_SNR{SNR_value}_iter{var_reps}_{add_tag}{day}{month}{year}.pkl','wb')
pickle.dump(hprParams,f)
f.close()