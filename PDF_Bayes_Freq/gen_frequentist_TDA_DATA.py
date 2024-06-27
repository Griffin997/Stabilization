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
randStart = False                  #Initial guess for parameter values in random locations
bounded = True

############# Global Params ###############

######All Fixed parameters for code
#Parameters held constant
T11 = 600
T12 = 1200
c1 = 0.5
c2 = 0.5 
T21 = 45
T22 = 100

true_params = np.array([T11, T12, c1, c2, T21, T22])

#Building the TE array - this should be a uniform array
n_TE = 64
TE_step = 8

TE_DATA = np.linspace(TE_step, TE_step*n_TE, n_TE) #ms units


TI_DATA = sorted(list(range(208, 1000, 16)))#[200, 300, 350, 400, 416, 450, 500, 550, 600, 650, 700, 750, 800, 832, 900, 1000]#sorted(list(range(208, 1000, 16)))#

TI1star = np.log(2)*T11
TI2star = np.log(2)*T12

#SNR Values to Evaluate
SNR_value = 1000

#Number of noisy realizations
var_reps = 100000

#Number of multi starts
multi_starts = 1

#Number of tasks to execute
target_iterator = [(a,b) for a in TI_DATA for b in range(var_reps//2)]

#### Important for Naming
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

num_cpus_avail = np.min([len(target_iterator),40])
data_path = "PDF_Bayes_Freq/TDA_freq_DATA"
add_tag = ""
data_head = "manyTI"
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

def flip_order(popt):
    p_flip = np.zeros(len(popt))
    for pi in range(np.size(popt)//2):
        p_flip[2*pi] = popt[2*pi+1]
        p_flip[2*pi+1] = popt[2*pi]
    return p_flip

#### Fitting helping functions

def get_func_bounds(func):
    f_name = func.__name__
    if f_name.find("6p") > -1:
        lower_bound = (0, 0, 0, 0, 0, 0)
        upper_bound = (2000, 2000, 1, 1, 300, 300)
    elif f_name.find("4p") > -1:
        lower_bound = (-1, -1, 0, 0)
        upper_bound = (1, 1, 300, 300)
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
    if randStart:
        lbs, ubs = get_func_bounds(func)
        p0 = np.zeros(len(lbs))
        for i in range(len(lbs)):
            p0[i] = np.random.uniform(lbs[i], ubs[i])

    else:
        if f_name.find("6p") > -1:
            p0 = true_params
        else:
            p0 = [d_value(TI,c1,T11), d_value(TI,c2,T12), T21, T22]
   
    return p0


def estP_oneCurve(func, TI_val, noisey_data):

    f_name = func.__name__
    init_p = set_p0(func, TI = TI_val)
    init_p_inv = flip_order(init_p)
    lb, ub = get_func_bounds(func)

    if bounded:
        if f_name.find("6p") > -1:
            popt_one, _, info_popt_one, _, _ = curve_fit(functools.partial(func, TI_val = TI_val), TE_DATA, noisey_data, p0 = init_p, bounds = [lb,ub], method = 'trf', maxfev = 4000, full_output = True)
            popt_two, _, info_popt_two, _, _ = curve_fit(functools.partial(func, TI_val = TI_val), TE_DATA, noisey_data, p0 = init_p_inv, bounds = [lb,ub], method = 'trf', maxfev = 4000, full_output = True)
        
        else:
            popt_one, _, info_popt_one, _, _ = curve_fit(func, TE_DATA, noisey_data, p0 = init_p, bounds = [lb,ub], method = 'trf', maxfev = 4000, full_output = True)
            popt_two, _, info_popt_two, _, _ = curve_fit(func, TE_DATA, noisey_data, p0 = init_p_inv, bounds = [lb,ub], method = 'trf', maxfev = 4000, full_output = True)
    else:
        if f_name.find("6p") > -1:
            popt_one, _, info_popt_one, _, _ = curve_fit(functools.partial(func, TI_val = TI_val), TE_DATA, noisey_data, p0 = init_p, maxfev = 4000, full_output = True)
            popt_two, _, info_popt_two, _, _ = curve_fit(functools.partial(func, TI_val = TI_val), TE_DATA, noisey_data, p0 = init_p_inv, maxfev = 4000, full_output = True)
        
        else:
            popt_one, _, info_popt_one, _, _ = curve_fit(func, TE_DATA, noisey_data, p0 = init_p, maxfev = 4000, full_output = True)
            popt_two, _, info_popt_two, _, _ = curve_fit(func, TE_DATA, noisey_data, p0 = init_p_inv, maxfev = 4000, full_output = True)

    
    RSS_one = np.sum(info_popt_one['fvec']**2)
    RSS_two = np.sum(info_popt_two['fvec']**2)

    return popt_one, RSS_one, popt_two, RSS_two

#### Parallelized Function

def generate_all_estimates(i_param_combo):
    #Generates a comprehensive matrix of all parameter estimates for all param combinations, 
    #noise realizations, SNR values, and lambdas of interest
    TI_value, nr = target_iterator[i_param_combo]

    feature_df = pd.DataFrame(columns = ["TI","NR","params1","RSS1", "params2","RSS2"])

    feature_df["TI"] = [TI_value]
    feature_df["NR"] = [nr]

    #Generate signal array from temp values
    true_signal = S_biX_6p(TE_DATA, *true_params, TI_val = TI_value)

    params_found = False
    while not params_found:
        try:
            noised_signal = add_noise(true_signal, SNR_value)

            RSS1_best = np.inf
            RSS2_best = np.inf
            for iMS in range(multi_starts):
                param1_temp, RSS1_temp, param2_temp, RSS2_temp = estP_oneCurve(S_biX_4p, TI_value, noised_signal)
            
                if RSS1_temp < RSS1_best:
                    RSS1_best = RSS1_best
                    param1_best = param1_temp

                if RSS2_temp < RSS2_best:
                    RSS2_best = RSS2_best
                    param2_best = param2_temp

            params_found = True
        except:
            params_found = False


    feature_df['params1'] = [param1_best]
    feature_df['RSS1'] = [RSS1_best]
    feature_df['params2'] = [param2_best]
    feature_df['RSS2'] = [RSS2_best]

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
    "rand_start": randStart,
    "multi_start": multi_starts,
    "bounded": bounded
}

f = open(f'{data_folder}/hprParameter_{data_head}_SNR{SNR_value}_iter{var_reps}_{add_tag}{day}{month}{year}.pkl','wb')
pickle.dump(hprParams,f)
f.close()