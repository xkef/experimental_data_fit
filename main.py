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
from kaufmann import *

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
X_train = np.linspace(1, 5000, 5000)
Y_train = counts[1:5000 + 1]


def Jacobian(x, *params):
    # wolfram alpha:
    # jacobian (a_0, a_1*exp(-b_1*x), a_2*exp(-b_2*x),a_3*exp(-b_3*x),a_4*exp(-b_4*x)) with respect to (a_0, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4)
    a = params[0:5]
    b = params[5:]
    J = np.array([
         [
             1, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
          0, np.exp(-x * b[1]), 0, 0, 0, -np.exp(-x * b[1]) * x * a[1], 0, 0, 0
         ],
         [
          0, 0, np.exp(-x * b[2]), 0, 0, 0, -np.exp(-x * b[2]) * x * a[2], 0, 0
         ],
         [
          0, 0, 0, np.exp(-x * b[3]), 0, 0, 0, -np.exp(-x * b[3]) * x * a[3], 0
         ],
         [
          0, 0, 0, 0, np.exp(-x * b[4]), 0, 0, 0, -np.exp(-x * b[4]) * x * a[4]
         ]
                 ])
    return J


def main_fit(X_train, Y_train, a, b):
    X_train = X_train[397:2000]
    Y_train = Y_train[397:2000] / Y_train[397:].max()

    def fitter(params):
        return exp_decay(
            range_end=X_train, a=params[0:5], b=params[5:]) - Y_train

    fit = least_squares(
        fitter,
        x0=np.hstack((a, b)),
        method='dogbox',
        loss='soft_l1',
        #jac=Jacobian,
        #bounds=(0, 150),
        verbose=2)
    plt.clf()
    plt.semilogy(X_train, exp_decay(range_end=X_train, a=a, b=b), label='fit')
    plt.semilogy(X_train, Y_train, label='real data')
    plt.xlabel('250ps/bin')
    plt.ylabel('counts')
    plt.legend(loc='best')
    plt.title('random fit')
    plt.savefig('plots/fit_kaufmann_start_params.pdf')
    print(fit)
    return fit

def exp_decay(a, b, range_end):
    array = []
    # range end is now an integer
    for x in range(len(range_end)):
        value = abs(a[0]) + abs(a[1]) * np.exp(-abs(b[0])*x)   \
                     + abs(a[2]) * np.exp(-abs(b[1])*x)   \
                     + abs(a[3]) * np.exp(-abs(b[2])*x)   \
                     + abs(a[4]) * np.exp(-abs(b[3])*x)
        array = np.append(array, value)

    gauss_1D_kernel_250ps = Gaussian1DKernel(sigma)
    astropy_conv = convolve(array, gauss_1D_kernel_250ps)
    return astropy_conv / astropy_conv.max()


# Parallel Grid Search for Start Values
def para_fit(i):
    # init for parallel grid search
    a, b = Kaufmann2003Solve(int(4), X_train, Y_train)
    a*=np.random.rand(5,)
    b*=np.random.rand(4,)
    fit = main_fit(X_train, Y_train, a, b)
    if (fit['x'].shape[0] > 0):
        np.savetxt('results/params' + str(i) + '.txt',
                   np.atleast_1d(fit['optimality']))
        np.savetxt('results/optimality' + str(i) + '.txt', fit['x'])


def run_parallel():
    # 1000  random fits to maxiter 100
    Parallel(n_jobs=-1)(delayed(para_fit)(i) for i in range(50))

################################################################################
################################################################################

if __name__ == '__main__':
    run_parallel()
    #kaufmann_inits(X_train, Y_train)
