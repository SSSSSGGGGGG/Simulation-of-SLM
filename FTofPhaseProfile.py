# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import scipy as sy
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# AL, AR, B, M, x,p=sp.symbols("Al AR B M x p")
x=np.linspace(-0.5, 0.5,640)  # linear
x_=np.linspace(0, 0.5,320)    #nonlinear
p=1
B=1
M=1.007
A=1000
AL=10
AR=1

# Phase profile for the linear
f_linear=2*x/p
# define nonlinear f_1, f is normalized, which affects phase profile a lot
f_1=np.arctan(B*(f_linear/2))
max_index_1 = np.argmax(f_1)
max_value_1 = f_1[max_index_1]
f_n=f_1/max_value_1
# equation=M*sp.atan(AL*sp.cos(f) )
# print(f"Equation: {equation}")

PhaseProfile=np.arctan(AL*np.cos(f_linear*np.pi))/(np.pi)
max_index = np.argmax(PhaseProfile)
max_value = PhaseProfile[max_index]

# Phase profile for the positive/right
f_=np.arctan(B*(x_/p))
max_index_ = np.argmax(f_)
max_value_ = f_[max_index_]
f_L=f_/max_value_

PhaseProfile_L=np.arctan(AL*np.cos(f_L*np.pi))/(np.pi)
max_index_L = np.argmax(PhaseProfile_L)
max_value_L = PhaseProfile_L[max_index_L]
PhaseProfile_L=PhaseProfile_L/max_value_L

# Phase profile for the negtive/left            
f_r=np.arctan(B*(-x_/p))
min_index_r = np.argmin(f_r)
min_value_r = f_r[min_index_r]
f_R=f_/min_value_r

PhaseProfile_R=np.arctan(AR*np.cos(f_R*np.pi))/(np.pi)
max_index_R = np.argmax(PhaseProfile_R)
max_value_R = PhaseProfile_R[max_index_R]
PhaseProfile_R=PhaseProfile_R/max_value_R

# combine L and R together
# PhaseProfile_n=[0] * (len(x_)+1)
# PhaseProfile_n[-1:] = PhaseProfile_L[::-1]
PhaseProfile_n=np.concatenate((PhaseProfile_L[::-1], PhaseProfile_R))

plt.figure(1)
plt.plot(x,f_linear,label="linear")
plt.plot(x,f_n,label="nonlinear")
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.25))
plt.legend(loc="best")
plt.minorticks_on()
plt.grid(True, linestyle='--')

# plt.figure(2)
# plt.plot(x, PhaseProfile / max_value,label="linear")
# plt.plot(x_,PhaseProfile_R,-x_,PhaseProfile_L,label="nonlinear_")
# plt.legend(loc="best")
# ax_n = plt.gca()
# ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
# plt.minorticks_on()
# plt.grid(True, linestyle='--')
# plt.show()

plt.figure(3)
plt.plot(x, PhaseProfile / max_value,label="linear")
plt.plot(x,PhaseProfile_n,label="nonlinear")
plt.legend(loc="best")
ax_n = plt.gca()
ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
plt.minorticks_on()
plt.grid(True, linestyle='--')
plt.show()