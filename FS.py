# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# p=64

# AL, AR, B, M, x,p=sp.symbols("Al AR B M x p")
x=np.linspace(-0.5, 0.5,640)  # linear
# x_=np.linspace(0, 0.5,320)    #nonlinear
m=0
p=1
B=1
M=1.007
A=1000
AL=1000
AR=1

# Phase profile for the linear
f_linear=2*x/p
# define nonlinear f_1, f is normalized, which affects phase profile a lot
f_1=np.arctan(B*(f_linear/2))
max_index_1 = np.argmax(f_1)
max_value_1 = f_1[max_index_1]
f_n=f_1/max_value_1

PhaseProfile=np.arctan(AL*np.cos(f_linear*np.pi))/(np.pi)
max_index = np.argmax(PhaseProfile)
max_value = PhaseProfile[max_index]
PhaseProfile=PhaseProfile/max_value
g=np.exp(1j *PhaseProfile*np.pi)

plt.figure(3)
plt.plot(x, PhaseProfile,label="linear")
plt.legend(loc="best")
ax_n = plt.gca()
ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
plt.minorticks_on()
plt.grid(True, linestyle='--')
plt.show()

cm=[]
for i in range(len(x)):
    # print(x[i],g[i])
    im_expr =  g[i] * np.exp(-1j * m * x[i] * np.pi / p)*(1/640)
    cm.append(im_expr)

cm_c=np.conjugate(cm)
Im=cm*cm_c/(2*len(x))**2
total = np.sum(Im)

plt.figure(1)
plt.plot(Im,label="linear")
# ax = plt.gca()
# ax.xaxis.set_major_locator(MultipleLocator(0.25))
# plt.legend(loc="best")
# plt.minorticks_on()
# plt.grid(True, linestyle='--')