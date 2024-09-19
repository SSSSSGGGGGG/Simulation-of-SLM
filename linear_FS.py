# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator



# AL, AR, B, M, x,p=sp.symbols("Al AR B M x p")
x=np.linspace(-0.5, 0.5,3600)  # linear
# x_=np.linspace(0, 0.5,320)    #nonlinear
m=0
p=1
B=1
M=1.007
A=1000
a=0.9
gray=0
gray_M=M*gray/255

# Phase profile for the linear
f_linear=2*x/p
# define nonlinear f_1, f is normalized, which affects phase profile a lot
f_1=np.arctan(B*(f_linear/2))
max_index_1 = np.argmax(f_1)
max_value_1 = f_1[max_index_1]
f_n=f_1/max_value_1

PhaseProfile=np.arctan(A*np.cos(f_linear*np.pi))/(np.pi)
max_index = np.argmax(PhaseProfile)
max_value = PhaseProfile[max_index]
PhaseProfile=PhaseProfile/max_value
PhaseProfile_M=PhaseProfile*M
phaseProfile_C=PhaseProfile_M*gray_M

pixelation=[0] * len(x)
for i in range(len(x)):
    if x[i]<len(x)/2 and x[i]>a/2:
        pixelation[i]=0
    elif x[i]>-len(x)/2 and x[i]<-a/2:
        pixelation[i]=0
    else:
        pixelation[i]=1
        
        
        
        
        
        
ex=m*np.pi*f_linear            
phase=pixelation*np.exp(1j*(phaseProfile_C*np.pi))*np.exp(1j*ex)
cm=np.sum(phase)
Im=cm*np.conjugate(cm)/len(x)**2
print(Im)
# plt.figure(1)
# plt.plot(x, ex,label="exp")
# plt.plot(x, phaseProfile_C,label="with gray")
# plt.plot(x, pixelation,label="fill factor")
# plt.legend(loc="best")
# ax_n = plt.gca()
# ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
# plt.minorticks_on()
# plt.grid(True, linestyle='--')
# plt.show()



