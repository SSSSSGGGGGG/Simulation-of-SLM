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
m=[0,1,2]
p=1
B=1
M=1.007
A=1000
a=1
gray=[0,8,16,24,32,40,48,56,64,72,80,88,96,
      104,112,120,128,136,144,152,160,168,
      176,184,192,200,208,216,224,232,240,
      248,255]


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


pixelation=[0] * len(x)
for i in range(len(x)):
    if x[i]<len(x)/2 and x[i]>a/2:
        pixelation[i]=0
    elif x[i]>-len(x)/2 and x[i]<-a/2:
        pixelation[i]=0
    else:
        pixelation[i]=1
        
mc=[]        
for k in range(len(m)):        
    cc=[]       
    for j in range(len(gray)):        
        gray_M=M*gray[j]/255        
        phaseProfile_C=PhaseProfile_M*gray_M        
        ex=m[k]*np.pi*f_linear            
        phase=pixelation*np.exp(1j*(phaseProfile_C*np.pi))*np.exp(1j*ex)
        cm=np.sum(phase)
        cc.append(cm)
    mc.append(cc)
Im=mc*np.conjugate(mc)/len(x)**2    
        

print(Im)

plt.figure(2)
plt.plot(gray,Im[0],label="I_0th")
plt.plot(gray,Im[1],label="I_1th")
plt.plot(gray,Im[2],label="I_2th")
plt.legend(loc="best")
ax_n = plt.gca()
# ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
plt.minorticks_on()
plt.grid(True, linestyle='--')
plt.show()

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



