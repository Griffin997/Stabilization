import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from Metropolis import run_metro
from Helpers import run_metro_2, normalize, bin_points
from tqdm import trange
from tqdm.notebook import trange as trange_nb
from scipy import optimize
import os

class Data:
    def __init__(self, TIs: list, n_iters: int, sample_size: int, SNR: float, nullpts: list = [416, 832], thresh: float = 1, bin_size: float = 0.01, normalize_each_TI: bool = True, flip: bool = True):
        # Data class includes methods for generating, saving, loading, and everything else
        # necessary to calculate the average critical radius
        # These methods can be run sequentially to using generate_all
        # ** The TIs parameter assumes that the TIs have constant deltaTI
        # for naming purposes but variable deltaTI is allowed **

        # Initializes the data object for a given SNR, TI range, number of iterations, sample size, and null points
        self.TIs = TIs
        self.TItitle = f"{min(TIs)}-{max(TIs)},{round(TIs[1]-TIs[0], 5)}"
        self.n_iters = n_iters
        self.sample_size = sample_size
        self.SNR = SNR
        self.nullpts = nullpts
        self.thresh = thresh
        self.bin_size = bin_size
        self.normalize_each_TI = normalize_each_TI
        self.flip = flip

        self.data = []
        self.binned = []
        self.threshed = []
        self.ripped = []
        self.acr_mean = []
        self.acr_std = []
        self.fit = None
        self.minTI = None
        # maps null points to dTIs
        self.dTI = {}

    def generate_data(self):
        # Generates data from the Metropolis algorithm
        # Doesn't normalize
        for i in trange(0, len(self.TIs)):
            self.data.append([])
            for j in range(0, self.sample_size):
                if self.flip:
                    self.data[i].append(run_metro_2(self.TIs[i], self.n_iters, verbose = False, SNR = self.SNR))
                else:
                    self.data[i].append(run_metro(self.TIs[i], self.n_iters, verbose = False, SNR = self.SNR))
        
        min_params = [min([min([min(self.data[i][j][:,k]) for j in range(0, self.sample_size)]) for i in range(0, len(self.TIs))]) for k in range(0, 4)]
        max_params = [max([max([max(self.data[i][j][:,k]) for j in range(0, self.sample_size)]) for i in range(0, len(self.TIs))]) for k in range(0, 4)]

        self.mins = np.zeros((len(self.TIs), self.sample_size, 4))
        self.maxs = np.zeros((len(self.TIs), self.sample_size, 4))

        # normalizes the data
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i])):
                if self.normalize_each_TI:
                    for k in range(0, 4):
                        self.data[i][j][:,k], self.mins[i][j][k], self.maxs[i][j][k] = normalize(self.data[i][j][:,k])
                else:
                    for k in range(0, 4):
                        self.data[i][j][:,k], self.mins[i][j][k], self.maxs[i][j][k] = normalize(self.data[i][j][:,k], min_params[k], max_params[k])

        self.data = np.array(self.data)


    def save_data(self, filename: str = None):
        if filename == None:
            filename = "MH_DATA\\data(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        np.save(filename, self.data)

    def load_data(self, filename: str = None):
        if filename == None:
            filename = "data(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"        
        self.data = np.load(filename, allow_pickle = True)

    def bin_data(self, verbose: bool = False):
        # normalizes and puts the data into bins of size bin_size
        self.binned = []
        if verbose:
            for i in trange(0, len(self.data)):
                self.binned.append([])
                for j in range(0, len(self.data[i])):
                    sample = self.data[i][j]
                    # normalizes each parameter
                    pts = bin_points(sample, self.bin_size)
                    self.binned[i].append(pts)
        else:
            for i in range(0, len(self.data)):
                self.binned.append([])
                for j in range(0, len(self.data[i])):
                    sample = self.data[i][j]
                    pts = bin_points(sample, self.bin_size)
                    self.binned[i].append(pts)

        self.binned = np.array(self.binned)

    def thresh_bins(self, verbose: bool = False):
        self.threshed = []
        if verbose:
            for i in trange(len(self.binned)):
                self.threshed.append([])
                for j in range(len(self.binned[i])):
                    # thresholds the binned data
                    threshed = [x for x in self.binned[i][j] if self.binned[i][j][x] >= self.thresh]
                    self.threshed[i].append(threshed)
        else:
            for i in range(len(self.binned)):
                self.threshed.append([])
                for j in range(len(self.binned[i])):
                    threshed = [x for x in self.binned[i][j] if self.binned[i][j][x] >= self.thresh]
                    self.threshed[i].append(threshed)

    def rip_threshed(self, verbose: bool = False):
        self.ripped = []
        if verbose:
            for i in trange(len(self.threshed)):
                self.ripped.append([])
                for j in range(len(self.threshed[i])):
                    # applies ripser to the threshed data and gets the critical radius
                    # ["dgms"][0][-2][1] gets the persistence data from Ripser, selects H_0,
                    # the penultimate bar, and the endpoint
                    try:
                        self.ripped[i].append(ripser(np.array(self.threshed[i][j]), maxdim = 0)["dgms"][0][-2][1])
                    except:
                        # less than 2 points
                        continue    
        else:
            for i in range(len(self.threshed)):
                self.ripped.append([])
                for j in range(len(self.threshed[i])):
                    try:
                        self.ripped[i].append(ripser(np.array(self.threshed[i][j]), maxdim = 0)["dgms"][0][-2][1])
                    except:
                        continue

        self.ripped = np.array(self.ripped)

    def acr_ripped(self, verbose: bool = False):
        self.acr_mean = []
        self.acr_std = []
        if verbose:
            for i in trange(len(self.ripped)):
                # calculates the mean and standard deviation of the critical radii
                self.acr_mean.append(np.mean(self.ripped[i]))
                self.acr_std.append(2*np.std(self.ripped[i])/np.sqrt(self.sample_size))
        else:
            for i in range(len(self.ripped)):
                self.acr_mean.append(np.mean(self.ripped[i]))
                self.acr_std.append(2*np.std(self.ripped[i])/np.sqrt(self.sample_size))

        self.acr_mean = np.array(self.acr_mean)
        self.acr_std = np.array(self.acr_std)

    def save_acr(self, filename: str = None):
        if filename == None:
            filename = "acr(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        np.save(filename, [self.acr_mean, self.acr_std])

    def load_acr(self, filename: str = None):
        if filename == None:
            filename = "acr(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        acr = np.load(filename, allow_pickle = True)
        self.acr_mean = acr[0]
        self.acr_std = acr[1]

    def plot_acr(self, error_bars: bool = True, null_color: str = "limegreen", connected: bool = True, polyfit: int = None, save: bool = False, filename: str = None):
        '''
        Plots the average critical radius across all TIs
        See fit_poly for details about fitting

        Parameters:
            error_bars: whether to plot error bars
            polyfit: If specified, will fit a polynomial of degree polyfit
                     to the data and plot the fit
            save: If True, will save the plot to a file
                  Otherwise, will display the plot
            filename: If save is True, will save the plot to this file
        '''
        
        title = f"Iterations: {self.n_iters}, Sample size: {self.sample_size}, Bin size: {self.bin_size}, Threshold: {self.thresh}\nSNR: {self.SNR}"
        plt.xlabel("TI")
        plt.ylabel("Average critical radius")
        marker = '-o' if connected else 'o'

        if error_bars:
            # connect points with line
            plt.errorbar(self.TIs, self.acr_mean, yerr = self.acr_std, fmt = marker, zorder = 1)
        else: 
            plt.plot(self.TIs, self.acr_mean, marker, zorder = 1)
        null = None
        for i in self.nullpts:
            if i in self.TIs:
                null = i
                plt.errorbar(i, self.acr_mean[self.TIs.index(i)], yerr = self.acr_std[self.TIs.index(i)], color = null_color, marker = 'o', zorder = 2)
        xlim = plt.xlim()
        ylim = plt.ylim()
        # plots the polynomial fit
        if polyfit != None:
            f = self.fit_poly(polyfit, null)
            fit = self.fit
            x = np.linspace(min(self.TIs), max(self.TIs), 1000)
            y = f(x)
            plt.plot(x, y, 'r')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.plot(self.minTI, f(self.minTI), 'ro', zorder = 3)
            title += f", Fit degrees: {polyfit}, Minimum: {self.minTI:.2f}, $\delta$ = {416 - self.minTI:.2f}"
        plt.title(title)
        if save:
            if filename == None:
                filename = ""
                if polyfit != None:
                    filename += "fit(" + str(polyfit) + ");"
                filename += "acr(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".png"
            plt.savefig(filename)
            plt.show()
            plt.close()
        else:
            plt.show()

    def fit_poly(self, degrees: int, null: float = None):
        '''
        Fits a polynomial to the data and sets the minimum point to self.minTI

        Parameters:
            degrees: How many degrees of the polynomial to fit
                     (2 for quadratic, 4 for quartic, etc.)
            null: The null point in the TI range
                  If specified self.dTI's value for that null poin
                  will be set to the difference between the null
                  point and the minimum SNR
        
        Returns:
            The polynomial function
        '''
        fit = np.polyfit(self.TIs, self.acr_mean, degrees)
        def f(x):
            sum = 0
            for i in range(degrees + 1):
                sum += fit[i]*x**(degrees - i)
            return sum
        self.fit = fit
        # finds the SNR where the fit is minimized
        self.minTI = optimize.fminbound(f, self.TIs[0], self.TIs[-1])
        if null != None:
            self.dTI[null] = null - self.minTI
        return f

    def generate_all(self, save: bool = False, verbose: bool = False):
        '''
        Runs everything sequentially

        Parameters:
            bin_size: The bin size used for binning
            thresh: The threshold used after binning
            save: Whether or not to save the data to a file
            verbose: Whether or not to print progress
        '''
        self.generate_data()
        if save:
            self.save_data()
        self.bin_data()
        self.thresh_bins()
        self.rip_threshed()
        self.acr_ripped()
        if save:
            self.save_acr()

# example usage
# TIs = list(range(366, 466, 2))
# #TIs = list(np.arange(405*10, 425.1*10, 1)/10)
# n_iters = 1000
# sample_size = 200
# os.chdir("test")
# #os.chdir("Data/1000(405-425)")
# SNRs = list(range(1000, 50250, 250))
# #SNRs = list(range(4000, 20000, 500)) + list([22500, 27500, 32500, 37500, 42500, 47500]) + list([20000, 25000, 30000, 35000, 40000, 45000, 50000])
# dTIs = []
# fit_degrees = 4
# for SNR in SNRs:
#     data = Data(TIs, n_iters, sample_size, SNR)
#     try:
#         data.load_acr()
#     except:
#         print(SNR)
#         data.generate_all()
#         data.load_acr()
#     data.fit_poly(fit_degrees, 416)
#     #data.plot_acr(False, fit_degrees, True)
#     dTIs.append(data.dTI[416])
# log = False
# if log:
#     plt.scatter(np.log(SNRs), np.log(dTIs), 10)
#     plt.xlabel("log(SNR)")
#     plt.ylabel("log($\delta$)")
# else:
#     plt.scatter(SNRs, dTIs, 10)
#     plt.xlabel("SNR")
#     plt.ylabel("$\delta$")
# plt.title(f"Iterations = {n_iters}, Sample Size = {sample_size}, Bin Size = 0.01, Threshold = 1\nFit Degrees = {fit_degrees}")
# plt.show()
