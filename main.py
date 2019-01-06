"""
 may better approach to construct func symbolic with sympy and compute jacobian
 import sympy as sym

 def Jacobian(v_str, f_list):
     vars = sym.symbols(v_str)
     f = sym.sympify(f_list)
     J = sym.zeros(len(f),len(vars))
     for i, fi in enumerate(f):
         for j, s in enumerate(vars):
             J[i,j] = sym.diff(fi, s)
     return J

 Jacobian('u1 u2', ['2*u1 + 3*u2','2*u1 - 3*u2'])
 $ returns: Matrix([[2,  3],[2, -3]])
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.optimize import least_squares
from joblib import Parallel, delayed

path_to_file = 'hT__Heterostructure_5keV_BGO.txt'
counts = []
with open(path_to_file, 'r') as myfile:
    csv_reader = csv.reader(
        myfile,
        delimiter=' ',
        skipinitialspace=True,
        escapechar='#',
        quoting=csv.QUOTE_NONNUMERIC)
    for number, line in enumerate(csv_reader):
        counts = np.append(counts, line[0])

# range_end = 6001
norm = 15 * 10**3
background = np.mean(counts[5000:5900])
posC0 = 399
sigma = 10
# changed (kevin)
X_train = np.linspace(1, 2000, 2000)
Y_train = counts[1:2000 + 1]


# Here we need array with 2 entries
def my4expmodel(tau, intens, range_end):
    array = []
    # range end is now an integer
    for afk in range(len(range_end)):
        if afk < posC0:
            value = 0.
            value = value + background
        else:
            value = intens[0] / (tau[0]) * np.exp(-(afk - posC0) / (tau[0] * 4)) \
                    + intens[1] / (tau[1]) * np.exp(-(afk - posC0) / (tau[1] * 4)) \
                    + intens[2] / (tau[2]) * np.exp(-(afk - posC0) / (tau[2] * 4)) \
                    + intens[3] / (tau[3]) * np.exp(-(afk - posC0) / (tau[3] * 4))
            value = value * norm
            value = value + background
        array = np.append(array, value)
    return array


def conv_nopoisson_4exp(tau, intens, range_end):
    gauss_1D_kernel_250ps = Gaussian1DKernel(sigma)
    values4expmodel = my4expmodel(tau, intens, range_end)
    astropy_conv = convolve(values4expmodel, gauss_1D_kernel_250ps)
    return astropy_conv


################################################################################
################################################################################


def to_fit(params):
    return conv_nopoisson_4exp(
        range_end=X_train, tau=params[0:4], intens=params[4:]) - Y_train


def construct_fit_start_params():
    tau_start = [140., 20., 0.6, 0.125] * np.random.rand(4, )
    int_start = [20.0, 15.0, 15.0, 25.0] * np.random.rand(4, )
    return np.hstack((tau_start, int_start))


def para_fit(i):
    fit = least_squares(
        to_fit,
        x0=construct_fit_start_params(),
        method='dogbox',
        loss='soft_l1',
        max_nfev=100,
        xtol=1e-14,
        gtol=1e-14,
        ftol=1e-14,
        jac='3-point',
        bounds=(0, 150),
        verbose=0)

    if (fit['x'].shape[0] > 0):
        np.savetxt('results/params' + str(i) + '.txt',
                   np.atleast_1d(fit['optimality']))
        np.savetxt('results/optimality' + str(i) + '.txt', fit['x'])


# 1000  random fits to maxiter 100
Parallel(n_jobs=-1)(delayed(para_fit)(i) for i in range(1000))

################################################################################
################################################################################

# x = fit['x']

# plt.clf()
# plt.semilogy(X_train,
#             conv_nopoisson_4exp(range_end=X_train,
#                         tau=x[0:4],
#                         intens=x[4:]),
#              label='fit')

# plt.semilogy(X_train,
#             conv_nopoisson_4exp(range_end=X_train,
#                         tau=construct_fit_start_params()[0:4],
#                         intens=construct_fit_start_params()[4:]),
#              label='start')

# plt.semilogy(X_train, Y_train, label='real data')
# plt.xlabel('250ps/bin')
# plt.ylabel('counts')
# plt.legend(loc='best')
# plt.title('kev fit')
# plt.savefig('fit.pdf')
