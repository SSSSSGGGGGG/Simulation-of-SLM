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
from scipy.optimize import minimize

def model(gray,AL,AR,B,M):
    x_=np.linspace(0, 0.5,320)    #nonlinear
    
    # M=1.0889580590571197
    # AL=7.200633666791641
    # AR=7.200633666791641
    # B=4.5555246333088015e-05
    p=1
    
    a=0.964
    
    m=[0,1,-1]
    # gray=np.arange(0,256,1)
    
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
    
    PhaseProfile_n=np.concatenate((PhaseProfile_L[::-1], PhaseProfile_R))
    
    # Loop for pixelation
    PhaseProfile_M=PhaseProfile_n*M
    x=np.linspace(-0.5, 0.5,640)
    pixelation=[0] * len(x)
    for i in range(len(x)):
        if x[i]<len(x)/2 and x[i]>a/2:
            pixelation[i]=0
        elif x[i]>-len(x)/2 and x[i]<-a/2:
            pixelation[i]=0
        else:
            pixelation[i]=1
            
    # loop for different order
    mc=[]        
    for k in range(len(m)):        
        cc=[]       
        for j in range(len(gray)):        
            gray_M=M*gray[j]/255        
            phaseProfile_C=PhaseProfile_M*gray_M       
            ex=m[k]*np.pi*2*x/p            
            phase=pixelation*np.exp(1j*phaseProfile_C*np.pi)*np.exp(1j*ex)
            cm=np.sum(phase)
            cc.append(cm)
        mc.append(cc)
    Im=mc*np.conjugate(mc)/len(x)**2 
    return Im.real



gray_y = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
y_data = np.array([0.93, 0.898, 0.765, 0.643, 0.466, 0.292, 0.138, 0.0038, 0.011, 0.051, 0.133, 0.275,
                   0.431, 0.592, 0.73, 0.831, 0.876])

# I_diff_or=model(gray_y,1000,1000,0.1,1)

def RMSE(params,gray_y, y_data):
    AL,AR,B,M = params
    y_pred = model(gray_y, AL,AR,B,M)
    # weights = 1 / (1 + (y_data - y_pred) ** 2)
    rmse_opt = np.sqrt(np.mean((y_data - y_pred[0]) ** 2))
    return  rmse_opt       #☻np.sum(weights * (y_data - y_pred) ** 2)

initial_guess = [100, 100, 0.2248,0.992]
bounds = [(0, 1000), (0, 1000), (0, 2),(0, 1.5)]

result = minimize(RMSE, initial_guess, method='SLSQP',args=(gray_y, y_data), bounds=bounds)
AL_opt, AR_opt, B_opt,M_opt = result.x
# Minimization result for RMSE
minimized_rmse = result.fun
print(f" Parameters: B={B_opt}, M={M_opt}, AL={AL_opt},AL={AL_opt},RMSE = {minimized_rmse:.4f}")

gray=np.arange(0,256,1)
I_fitted = model(gray, AL_opt, AR_opt, B_opt,M_opt)
plt.figure()
plt.scatter(gray_y, y_data, label='Experimental Data', color='blue')
plt.plot(gray, I_fitted[0], label=f'Optimized Fit ', color='green') #(RMSE = {rmse_opt:.4f})
plt.legend()
ax_n = plt.gca()
ax_n.yaxis.set_minor_locator(MultipleLocator(0.02))
plt.legend(loc="best")
plt.grid(True, linestyle='--')
plt.ylim([0,1])
plt.xlim([0,255])
plt.show()
plt.xlabel('Gray Level')
plt.ylabel('Intensity')

plt.show()

# plt.figure(2)
# plt.plot(gray,Im[0],label="I_0th")
# plt.plot(gray,Im[1],label="I_1th")
# plt.plot(gray,Im[2],label="I_2th")

# plt.plot(gray_y,y_data ,label="Exp I_0th",linestyle='dotted',marker="o")
# # plt.plot(gray,Im[1],label="I_1th")
# plt.legend(loc="best")
# plt.grid(True, linestyle='--')
# plt.show()

# plt.figure(4)
# plt.plot(x, PhaseProfile / max_value,label="linear")
# plt.plot(x_,PhaseProfile_R,-x_,PhaseProfile_L,label="nonlinear_")
# plt.legend(loc="best")
# ax_n = plt.gca()
# ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
# plt.minorticks_on()
# plt.grid(True, linestyle='--')
# plt.show()

# plt.figure(3)
# plt.plot(x, PhaseProfile / max_value,label="linear")
# plt.plot(x,PhaseProfile_n,label="nonlinear")
# plt.plot(x,phaseProfile_C,label="nonlinearM+g")
# plt.legend(loc="best")
# ax_n = plt.gca()
# ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
# plt.minorticks_on()
# plt.grid(True, linestyle='--')
# plt.show()