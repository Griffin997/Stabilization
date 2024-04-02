from Metropolis import run_metro
import numpy as np

def run_metro_2(TI, n_iters, SNR: float = 10000, verbose: bool = True):
    data1 = run_metro(TI, int(n_iters/2), verbose = verbose, flip = False, SNR = SNR)
    data2 = run_metro(TI, int(n_iters/2), verbose = verbose, flip = True, SNR = SNR)
    return np.concatenate((data1, data2))

def normalize(data, max = None, min = None):
    #normalize data to make it between 0 and 1
    if max == None:
        max = np.max(data)
    if min == None:
        min = np.min(data)
    return (data - min)/(max - min), min, max

def determine_bin(values, bin_size):
    # returns the center of the bin
    return tuple((int(value//bin_size)+0.5)*bin_size for value in values)

def bin_points(data: list, bin_size: float):
    # points is a dictionary mapping a tuple (the bin center)
    # to the amount of points in that bin
    points = {}
    for i in range(0, len(data)):
        point = [data[:,j][i] for j in range(0, 4)]
        bin = determine_bin(point, bin_size)
        if bin in points:
            points[bin] += 1
        else:
            points[bin] = 1
    return points