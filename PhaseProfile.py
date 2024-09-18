# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import scipy as sy
import sympy as sp
import matplotlib.pyplot as plt

# AL, AR, B, M, x,p=sp.symbols("Al AR B M x p")
x=np.linspace(-0.5, 0.5,640)
p=1
B=5

f_linear=2*x/p
# define nonlinear f_1, f is normalized, which affects phase profile a lot
f_1=np.arctan(B*(f_linear/2))
max_index_1 = np.argmax(f_1)
max_value_1 = f_1[max_index_1]
f_n=f_1/max_value_1
# equation=M*sp.atan(AL*sp.cos(f) )
# print(f"Equation: {equation}")

M=1.007
AL=1000
AR=1
PhaseProfile=np.arctan(AL*np.cos(f_linear*np.pi))/(np.pi)
max_index = np.argmax(PhaseProfile)
max_value = PhaseProfile[max_index]


def phase_non(AL,AR,x,f_n):
    j=0
    y=[]
    for j in range(len(x)):
        print(j)
        PhaseProfile_nl=[]
        PhaseProfile_nr=[]
        if x[j]<0:
            
            PhaseProfile_L=np.arctan(AL*np.cos(f_n[j]*np.pi))/(np.pi)
            PhaseProfile_nl.append(PhaseProfile_L)
            max_index_L = np.argmax(PhaseProfile_nl)
            max_value_L = PhaseProfile_nl[max_index_L]
            PhaseProfile_nl=PhaseProfile_nl/max_value_L
            
            
        else:

            PhaseProfile_R=np.arctan(AR*np.cos(f_n[j]*np.pi))/(np.pi)
            PhaseProfile_nr.append(PhaseProfile_R)
            max_index_R = np.argmax(PhaseProfile_nr)
            max_value_R = PhaseProfile_nr[max_index_R]
            PhaseProfile_nr=PhaseProfile_nr/max_value_R
            
        y.append(j)        
      
    return y,x[0]

plt.figure(1)
plt.plot(x,f_linear,label="linear")
plt.plot(x,f_n,label="nonlinear")
plt.legend(loc="best")
plt.minorticks_on()

L,R=phase_non(AL,AR,x,f_n)

# phase_N=phase_non(AL,AR,x)
# plt.figure(2)
# plt.plot(x, PhaseProfile / max_value,label="linear")
# plt.plot(x,phase_N[0],label="nonlinear")
# plt.legend(loc="best")
# plt.minorticks_on()
# plt.show()