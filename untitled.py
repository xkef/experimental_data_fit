import sympy as sym
import matplotlib.pyplot as plt
import numpy as np


 def Jacobian(v_str, f_list):
     vars = sym.symbols(v_str)
     f = sym.sympify(f_list)
     J = sym.zeros(len(f),len(vars))
     for i, fi in enumerate(f):
         for j, s in enumerate(vars):
             J[i,j] = sym.diff(fi, s)
     return J

 Jacobian('u1 u2', ['2*u1 + 3*u2','2*u1 - 3*u2'])
 # returns: Matrix([[2,  3],[2, -3]])

x = symbols('x')
i = IndexedBase('i')
t = IndexedBase('t')
Sum(intens[n] / (tau[n]) * exp(-(x - 1) / (tau[n] * 4)), (n, 0, 3)).doit()
