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

cwd = os.getcwd()
experiment_folder = "DualGel_Experiments"

####### Options #######
randStart = False                  #Initial guess for parameter values in random locations
gen_stand_ref = True

run_number = 82

file_oi = f"real_phased_run{run_number}" #imag_unphased_dataset    real_unphased_dataset   real_phased_dataset

raw = scipy.io.loadmat(f'{cwd}/{experiment_folder}/{file_oi}.mat')
raw_data = raw['real_phased_dataset']

############# Global Params ###############

######All Fixed parameters for code
#Parameters held constant
T21 = 30.8 #+- 0.2 ms
T11 = 91.5 #+- 2.3 ms
c1 = 0.5
T22 = 40.7 #+- 0.2 ms
T12 = 353.3 #+- 12.6 ms
c2 = 1 - c1

true_params = np.array([T11, T12, c1, c2, T21, T22])

#Null Point Values
TI1star = np.log(2)*T11
TI2star = np.log(2)*T12

## TI index
TI_STANDARD = (np.logspace(0,np.log10(2000),12)+.5)//1
index_TI1star = np.argmin((TI_STANDARD - TI1star)**2)
assert(index_TI1star == 6)
index_TI2star = np.argmin((TI_STANDARD - TI2star)**2)
assert(index_TI2star == 8)

with open(f'{cwd}/{experiment_folder}/dualGel_TI.txt') as f:
    TI = f.readlines()
TI_DATA = [int(sub.replace("\n", "")) for sub in TI]

assert(TI_DATA[0] < TI_DATA[-1])
TE_DATA = np.arange(1,2048.1,1)*0.4 #ms

TI_STANDARD_indices = [np.where(iTI == TI_DATA)[0][0] for iTI in TI_STANDARD]

if gen_stand_ref:
    TI1g_indices = [TI_STANDARD_indices[index_TI1star]]
    TI2g_indices = [TI_STANDARD_indices[index_TI2star]]
else:
    #These are the windows that we are going to be analyzing
    ### For TI1star we have 15% boundaries with sampling at every 1 ms - we can use all 21 points
    TI1g_indices = np.arange(TI_STANDARD_indices[5]+1,TI_STANDARD_indices[7],1)
    ### For TI2star we have 15% boundaries with sampling at every 1 ms
    ### This results in 75 points - we use every third point for 25 points total
    TI2g_indices = np.arange(TI_STANDARD_indices[7]+1,TI_STANDARD_indices[9],3)

#This block identifies the moX indices with a one and all biX indices with a zero
Exp_STANDARD = np.zeros(len(TI_STANDARD))
Exp_NP = np.zeros(len(TI_STANDARD))
Exp_NP[index_TI1star] = 1
Exp_NP[index_TI2star] = 1

#This is where we label indices as moX or biX based on the previous block
curve_options = ["BiX", "MoX"] 
Exp_label_NP = [curve_options[int(elem)] for elem in Exp_NP]
Exp_label_STANDARD = [curve_options[int(elem)] for elem in Exp_STANDARD]

#Loading in the data
sp = 2      #spacing of TE in terms of indices
ext = 256   #number of TE points

#separate by spacing
raw_data = raw_data[sp-1::sp,:,:]
TE_DATA = TE_DATA[sp-1::sp]
#limit number of points studied - removes unneeded tail - don't want to overfit to the baseline
raw_data = raw_data[:ext,:,:]
TE_DATA = TE_DATA[:ext]

#make sure all the data is here as we expect - this is important for indexing later
assert(len(TI_DATA) == raw_data.shape[1])

#number of noise realizations
var_reps = raw_data.shape[-1]

if randStart:
    multi_starts_obj = 2
else:
    multi_starts_obj = 1 

target_iterator = [(a,b) for a in TI1g_indices for b in TI2g_indices]

#Builds a string of parameters to use in the titles
ParamTitle_6p = [r'$T_{11}$', r'$T_{12}$', r'$c_1$', r'$c_2$', r'$T_{21}$', r'$T_{22}$']
round_Params = [round(num, 2) for num in true_params]
pList = ', '.join(str(x) for x in round_Params)
pTitleList = ', '.join(x for x in ParamTitle_6p) #Builds the list of parametes used to generate the original data

#### Important for Naming
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

num_cpus_avail = np.min([len(target_iterator), 15, mp.cpu_count()])

if gen_stand_ref:
    front_label = 'SR_'
else:
    front_label = ''

data_path = f"{experiment_folder}/DG_Analysis_DATA"
add_tag = ""
data_tag = (f"{front_label}reassignExp_run{run_number}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)

############# Functions ##############

#### Signaling Functions

def S_biX_6p(TE, T11, T12, c1, c2, T21, T22, TI = 0):
    exp1 = c1*(1-2*np.exp(-TI/T11))*np.exp(-TE/T21)
    exp2 = c2*(1-2*np.exp(-TI/T12))*np.exp(-TE/T22)
    return exp1 + exp2

#The ravel structure necessary for the curve_fit algorithm
def S_biX_6p_ravel(T_dat, T11, T12, c1, c2, T21, T22):
    TE, TI = T_dat
    exp1 = c1*(1-2*np.exp(-TI/T11))*np.exp(-TE/T21)
    exp2 = c2*(1-2*np.exp(-TI/T12))*np.exp(-TE/T22)
    return exp1 + exp2

def calculate_RSS_TI(func, popt, TI_val, data):
    est_curve = func(TE_DATA, *popt, TI = TI_val)
    RSS = np.sum((est_curve - data)**2)
    
    return RSS

def S_moX_3p(TE, T1, c, T2, TI = 0):
    return c*(1-2*np.exp(-TI/T1))*np.exp(-TE/T2)

#The one dimensional models are used to evaluate if a curve is more likely monoX or biX
def S_biX_4p(TE, d1, d2, T21, T22):
    exp1 = d1*np.exp(-TE/T21)
    exp2 = d2*np.exp(-TE/T22)
    return exp1 + exp2

def S_moX_2p(TE, d, T2):
    return d*np.exp(-TE/T2)

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
    if f_name == "S_biX_6p":
        lower_bound = (1, 1, 0, 0, 1, 1)
        upper_bound = (500, 500, 1, 1, 150, 150)
    elif f_name == "S_moX_3p":
        lower_bound = (1, 0, 1)
        upper_bound = (500, 1, 150)
    elif f_name == "S_biX_4p":
        lower_bound = (-1, -1, 1, 1)
        upper_bound = (1, 1, 150, 150)
    elif f_name == "S_moX_2p":
        lower_bound = (-1, 1)
        upper_bound = (1, 150)
    else:
        raise Exception("Not a valid function: " + f_name)

    return lower_bound, upper_bound

def get_param_list(func):
    f_name = func.__name__
    if f_name.find("S_biX_6p") > -1:
        params = ("T21","T22","c1","c2","T21","T22")
    elif f_name.find("S_moX_3p") > -1:
        params = ("T21","c","T2")
    else:
        raise Exception("Not a valid function: " + f_name)

    return params

def set_p0(func, random = True):
    # true_params = [T11, T12, c1, c2, T21, T22]
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
        # p0 = np.zeros(len(ParamTitle_6p))
            
    return p0

def check_param_order(popt):
    #Reshaping of array to ensure that the parameter pairs all end up in the appropriate place - ensures that T22 > T21
    if (popt[-1] < popt[-2]): #We want by convention to make sure that T21 is <= T22
        for pi in range(np.size(popt)//2):
            p_hold = popt[2*pi]
            popt[2*pi] = popt[2*pi+1]
            popt[2*pi+1] = p_hold
    return popt

def calculate_RSS_TI(func, popt, TI_val, data):
    est_curve = func(TE_DATA, *popt, TI = TI_val)
    RSS = np.sum((est_curve - data)**2)
    
    return RSS

def calculate_RSS(func, popt, data):
    est_curve = func(TE_DATA, *popt)
    RSS = np.sum((est_curve - data)**2)
    
    return RSS

def det_normFactor(noisy_curve):

    lb = (0, 0, 0, 0)
    ub = (np.inf, np.inf, 150, 150)

    d_init = 0.5*noisy_curve[0]

    popt, _, _, _, _ = curve_fit(S_biX_4p, TE_DATA, noisy_curve, p0 = (d_init, d_init, 30, 50), bounds = [lb,ub], method = 'trf', maxfev = 1500, full_output=True)

    return popt[0]+popt[1]

def d_value(TI,c,T1):
    return c*(1-2*np.exp(-TI/T1))

def estP_oneCurve(func, noisey_data):

    init_p = set_p0(func, random = True)
    lb, ub = get_func_bounds(func)

    popt, _ = curve_fit(func, TE_DATA, noisey_data, p0 = init_p, bounds = [lb,ub], method = 'trf', maxfev = 1500)
    popt = check_param_order(popt)
    RSS = calculate_RSS(func, popt, noisey_data)

    return popt, RSS


def bounds_condensed(lb, ub):
    lb, ub = get_func_bounds(S_biX_6p)
    bnd_cat = [lb,ub]
    bnd_cat = np.array(bnd_cat)
    bnd_cat = np.transpose(bnd_cat)
    bnds = bnd_cat.tolist()
    return bnds
    

#### Objective Function

def list_objective_func(param_est, data_2d, TI_array, X_list):
    assert(data_2d.shape[0] == len(TI_array))
    assert(len(X_list) == len(TI_array))

    curve_RSS = 0

    X_truth = [elem == "BiX" for elem in X_list]

    for iter in range(len(X_truth)):
        if X_truth[iter]:
            RSS_add = calculate_RSS_TI(S_biX_6p, param_est, TI_array[iter], data_2d[iter,:])
        else:
            if data_2d[iter,0] < 0:
                #first null point -> that means that only the long parameters with the two are used
                RSS_add = calculate_RSS_TI(S_moX_3p, [param_est[-5], param_est[-3], param_est[-1]], TI_array[iter], data_2d[iter,:])
            else:
                #second null point -> that means that only the short parameters with the two are used
                RSS_add = calculate_RSS_TI(S_moX_3p, [param_est[-6], param_est[-4], param_est[-2]], TI_array[iter], data_2d[iter,:])

        curve_RSS += RSS_add

    return curve_RSS

def RSS_obj_func(popt, data, TI_val, func):
    est_curve = func(TE_DATA, *popt, TI = TI_val)
    RSS = np.sum((est_curve - data)**2)
    
    return RSS

#### Metric

def calc_MSE(paramStore, true_params):
    varMat = np.var(paramStore, axis=0)
    biMat = np.mean(paramStore, axis = 0) - true_params  #E[p_hat] - p_true
    MSEMat = varMat + biMat**2
    return MSEMat, varMat, biMat


#### Parallelized Function

def preEstimate_parameters(TE_DATA, TI_DATA, noised_data, lb, ub):

    mTE, mTI = np.meshgrid(TE_DATA, TI_DATA)
    vecT = np.vstack((mTE.ravel(), mTI.ravel())) #flattens the data points

    cF_fval = np.inf

    no_opt_found = 0

    for ms_iter in range(multi_starts_obj):
        init_p = set_p0(S_biX_6p, random = randStart)

        try:
            vecS = noised_data.ravel()
            popt_temp, _ = curve_fit(S_biX_6p_ravel, vecT, vecS, p0 = init_p, bounds = [lb, ub], method = 'trf', maxfev = 5000)
            RSS_cF_array = []
            for iter in range(noised_data.shape[0]):
                RSS_cF_array.append(calculate_RSS_TI(S_biX_6p, popt_temp, TI_DATA[iter], noised_data[iter,:]))
            RSS_cF_temp = np.sum(RSS_cF_array)
            if RSS_cF_temp < cF_fval:
                popt = popt_temp
                cF_fval = RSS_cF_temp
        except:
            no_opt_found+=1

    #This is the failsafe to ensure that some parameters are found
    if no_opt_found == multi_starts_obj:
        print("Overtime")
        while no_opt_found > 0:
            init_p = set_p0(S_biX_6p, random = randStart)
            try:
                vecS = noised_data.ravel()
                popt_temp, _ = curve_fit(S_biX_6p_ravel, vecT, vecS, p0 = init_p, bounds = [lb, ub], method = 'trf', maxfev = 5000)
                RSS_cF_array = []
                for iter in range(noised_data.shape[0]):
                    RSS_cF_array.append(calculate_RSS_TI(S_biX_6p, popt_temp, TI_DATA[iter], noised_data[iter,:]))
                RSS_cF_temp = np.sum(RSS_cF_array)
                if RSS_cF_temp < cF_fval:
                    popt = popt_temp
                    cF_fval = RSS_cF_temp
                no_opt_found = 0
            except:
                no_opt_found = 1

    return check_param_order(popt)

def estimate_parameters(popt, TI_DATA, noised_data, lb, ub, list_curve_X, list_curve_cvn):

    bnds = bounds_condensed(lb, ub)

    res_COFFEE = minimize(list_objective_func, popt, args = (noised_data, TI_DATA, list_curve_X), method = 'Nelder-Mead', bounds = bnds, options = {'maxiter': 4000, 'disp': False})
    res_cvn = minimize(list_objective_func, popt, args = (noised_data, TI_DATA, list_curve_cvn), method = 'Nelder-Mead', bounds = bnds, options = {'maxiter': 4000, 'disp': False})

    #Pulling relevant AIC objective function parts
    param_est_COFFEE = check_param_order(res_COFFEE.x)
    #Pulling relevant cvn objective function parts
    param_est_cvn = check_param_order(res_cvn.x)

    return param_est_COFFEE, param_est_cvn

def generate_all_estimates(i_param_combo, passed_data):
    #Generates a comprehensive matrix of all parameter estimates for all param combinations, 
    #noise realizations, SNR values, and lambdas of interest
    TI1ind, TI2ind = target_iterator[i_param_combo]
    TI_NP = np.copy(TI_STANDARD)
    temp_indices = np.copy(TI_STANDARD_indices)
    temp_indices[index_TI1star] = TI1ind
    temp_indices[index_TI2star] = TI2ind
    TI_NP[index_TI1star] = TI_DATA[TI1ind]
    TI_NP[index_TI2star] = TI_DATA[TI2ind]

    feature_df = pd.DataFrame(columns = ["TI1*g","TI2*g","TI_DATA", "MSE", "var", "bias", "pEst_cvn", "pEst_AIC", "pEst_cf"])

    feature_df["TI1*g"] = [TI_DATA[TI1ind]]
    feature_df["TI2*g"] = [TI_DATA[TI2ind]]
    feature_df["TI_DATA"] = [TI_NP]

    signal_array = np.zeros([len(TE_DATA), len(TI_NP), var_reps])
    #Generate signal array from temp values
    for iTI in range(len(TI_NP)):
        signal_array[:,iTI,:] = passed_data[:,int(temp_indices[iTI]),:]

    #Temp Parameter matrices
    param_est_COFFEE = np.zeros((var_reps, len(true_params)))
    param_est_cvn = np.zeros((var_reps, len(true_params)))
    param_est_cF = np.zeros((var_reps, len(true_params)))

    lb, ub = get_func_bounds(S_biX_6p)

    MSE_mat = np.zeros((3, len(true_params)))
    var_mat = np.zeros((3, len(true_params)))
    bias_mat = np.zeros((3, len(true_params)))

    for nr in range(var_reps):    #Loop through all noise realizations
        noised_data = signal_array[:,:,nr]

        param_est_cF[nr,:] = preEstimate_parameters(TE_DATA, TI_NP, np.transpose(noised_data), lb, ub)
        param_est_COFFEE[nr,:], param_est_cvn[nr,:] = estimate_parameters(param_est_cF[nr,:], TI_NP, np.transpose(noised_data), lb, ub, Exp_label_NP, Exp_label_STANDARD)

    MSE_mat[0,:], var_mat[0,:], bias_mat[0,:] = calc_MSE(param_est_COFFEE, true_params)
    MSE_mat[1,:], var_mat[1,:], bias_mat[1,:] = calc_MSE(param_est_cvn, true_params) 
    MSE_mat[2,:], var_mat[2,:], bias_mat[2,:] = calc_MSE(param_est_cF, true_params)

    feature_df['MSE'] = [MSE_mat]
    feature_df['var'] = [var_mat]
    feature_df['bias'] = [bias_mat]
    feature_df['pEst_AIC'] = [param_est_COFFEE]
    feature_df['pEst_cf'] = [param_est_cF]
    feature_df['pEst_cvn'] = [param_est_cvn]

    return feature_df

norm_factor = det_normFactor(np.mean(raw_data[:,-1,:], axis = -1))
normed_data = raw_data/norm_factor
# print(f"All data was normalized by the normalization factor value of {norm_factor: 0.2f}")

#### Looping through Iterations of the brain - applying parallel processing to improve the speed
if __name__ == '__main__':
    freeze_support()

    print("Finished Assignments...")

    ##### Set number of CPUs that we take advantage of
    print(f"Using {num_cpus_avail} cpu")

    lis = []

    with mp.Pool(processes = num_cpus_avail) as pool:

        with tqdm(total = len(target_iterator)) as pbar:
            for estimates_dataframe in pool.imap_unordered(functools.partial(generate_all_estimates, passed_data = normed_data), range(len(target_iterator))):
            
                lis.append(estimates_dataframe)

                pbar.update()

        pool.close()
        pool.join()
    

    print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    df = pd.concat(lis, ignore_index= True)

    df.to_pickle(data_folder + f'/' + data_tag +'.pkl')     

############## Save General Code Code ################

hprParams = {
    "TI1g_indices": TI1g_indices,         #first iterator
    "TI2g_indices": TI2g_indices,         #second iterator
    "true_params": true_params,
    "TI_STANDARD": TI_STANDARD,
    "spacing": sp,
    "num_TE": ext,
    'multi_start': multi_starts_obj,
    'run_number': run_number,
    'norm_factor': norm_factor,
    'n_noise_realizations': var_reps
}

f = open(f'{data_folder}/{front_label}hprParameter_run{run_number}_{day}{month}{year}.pkl','wb')
pickle.dump(hprParams,f)
f.close()

