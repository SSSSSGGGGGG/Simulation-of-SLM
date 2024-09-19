# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:08:29 2024

@author: gaosh
"""
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sympy as sp


# plt.grid(True)

k=4  # period
delta=6.4E-6
N_tri=[0,1,2] # the resulting order
n=np.arange(-50,50,1)


FF=0.99999999999 # fill factor
# phi=np.linspace(0,2,100)
phi_r=np.linspace(0, 2,200)

# print(f"the period is {k}, the order {N} is calculated, m={m} is for grating order, and n={n} is for SLM.")
# print(f"the spatial frequences are {u}")
I_tri=[[]]
for j, N in enumerate(N_tri):
    m=N-k*n
    u=(m+k*n)/(k*delta)
    I=[]
    for i, phi in enumerate(phi_r):
        a_sinc=FF*sp.sinc((FF/k)*(m+k*n)*sp.pi)
        cos=sp.cos(0.5*m*sp.pi)*sp.cos(0.5*phi*sp.pi)
        sin=sp.sin(0.5*m*sp.pi)*sp.sin(0.5*phi*sp.pi)
        diff=cos-sin
        
        G=a_sinc*diff*sp.sinc(0.5*m*sp.pi)*pow(-1.0,n)
        G_sum=sum(G)
        Intensity=G_sum**2
        I.append(Intensity)
    I_tri.append(I)

plt.plot(phi_r,I_tri[1],label="0th")
plt.plot(phi_r,I_tri[2],label="1st")
plt.plot(phi_r,I_tri[3],label="2nd")
plt.ylim(0, 1)
plt.xlim(0, 2)
plt.legend()
plt.yticks(np.arange(0, 1.2, 0.2))
plt.minorticks_on()
print(f"maximum of I_0 is{np.max(I_tri[1])} and the maximum of I_1 is{np.max(I_tri[2])}")
