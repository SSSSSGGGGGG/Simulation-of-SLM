# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import sympy as sp
import scipy as sc

k = 4  # period
delta = 6.4E-6
n = np.linspace(-50, 50, 100)
a = 0.1  # fill factor
# a=np.arange(0,1.1,0.1 )

phi= sp.symbols('phi')
# accu=[]
# for a in np.arange(0, 1.1, 0.1):
#     f0=a**2*sp.cos((phi/2)*sp.pi)**2
#     f1=a**2*sp.sinc((a/2)*sp.pi)**2*sp.sin((phi/2)*sp.pi)**2
#     equation=sp.Eq(f1, f0)
#     solutions = sp.solve(equation, phi)
#     accu.append(solutions)
# print("Solutions for phi:", accu)

f0=a**2*sp.cos((phi/2)*sp.pi)**2
f1=a**2*sp.sinc((a/2)*sp.pi)**2*sp.sin((phi/2)*sp.pi)**2
equation= f0 - f1
solutions = sp.solve(equation, phi)
print("Solutions for phi:", solutions)
