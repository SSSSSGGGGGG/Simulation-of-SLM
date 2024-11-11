# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
"""
In this section, we will use our model to fit the experimental results. The model is the nonlinear phase profile funtion in terms of AL, AR, B, and M.
The normalized intensities of diffraction orders will be calculated based on this model. In addition, we will use the "minimize" algrithm to obtain a minimal RMSE between the experimental and model results.
Once the RMSE is small enough, we are able to fit the experimental results well. 
"""
# We define a model function, which can be called with several necessary arguments : an array of gray level, roundness AL/AR, parameters B and M. 
# Then the function will return the normalized intensities of diffraction orders.
def model(gray,AL,AR,B,M,order):
    p=1  
    a=0.964    
    m=[order] #0,-+1
    x_=np.linspace(0, 0.5,320)    
    
    f_=np.arctan(B*(x_/p))
    max_index_ = np.argmax(f_)
    max_value_ = f_[max_index_]
    f_L=f_/max_value_
    
    PhaseProfile_L=np.arctan(AL*np.cos(f_L*np.pi))/(np.pi)
    max_index_L = np.argmax(PhaseProfile_L)
    max_value_L = PhaseProfile_L[max_index_L]
    max_value_L = max(max_value_L, 1e-10)  # Prevent divide by zero
    PhaseProfile_L=PhaseProfile_L/max_value_L    
                
    f_r=np.arctan(B*(-x_/p))
    min_index_r = np.argmin(f_r)
    min_value_r = f_r[min_index_r]
    f_R=f_/min_value_r
    
    
    
    PhaseProfile_R=np.arctan(AR*np.cos(f_R*np.pi))/(np.pi)
    max_index_R = np.argmax(PhaseProfile_R)
    max_value_R = PhaseProfile_R[max_index_R]
    max_value_R = max(max_value_R, 1e-10)  # Prevent divide by zero
    PhaseProfile_R=PhaseProfile_R/max_value_R
    
    PhaseProfile_n=np.concatenate((PhaseProfile_L[::-1], PhaseProfile_R))    
    
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

gray_y = np.array([0,8,16,24,32,40,48,56,64,72,80,88,96
,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216
,224,232,240,248,255])    # Experimental gray level
y_data = np.array([0.930,0.926,0.921,0.913,0.896,0.875,0.845,0.815
,0.772,0.729,0.683,0.623,0.567,0.499,0.448,0.384,0.341,0.294,0.230
,0.192,0.158,0.132,0.102,0.090,0.081,0.085,0.094,0.102,0.128,0.154
,0.188,0.222,0.243])
y_data_1 = np.array([0.000,0.001,0.004,0.009,0.017,0.027,0.041
,0.053,0.073,0.090,0.110,0.135,0.156,0.188,0.203,0.233,0.248,0.263
,0.288,0.296,0.309,0.309,0.316,0.311,0.311,0.297,0.280,0.270,0.247
,0.224,0.195,0.169,0.157])
y_data_N1 = np.array([0.000,0.001,0.004,0.009,0.017,0.027,0.041
,0.054,0.073,0.089,0.110,0.134,0.157,0.186,0.206,0.230,0.243,0.260
,0.276,0.286,0.291,0.294,0.291,0.286,0.273,0.258,0.242,0.218,0.196
,0.173,0.146,0.124,0.105])
# We define a RMSE function, which can be called with several necessary arguments : an group of initialized parameters AL/AR, B, and M, the experimental gray levels and corresponding normaolized diffraction order´s intensities. 
# Then the function will return the optimal RMSE value between the model´s and the experimental normalized intensities of diffraction orders.
def RMSE(params,gray_y, y_data,y_data_1, y_data_N1):
    AL,AR,B,M = params
    y_pred_0 = model(gray_y, AL,AR,B,M,0)
    rmse_opt_0 = np.sqrt(np.mean((y_data - y_pred_0[0]) ** 2))
    
    y_pred_1 = model(gray_y, AL,AR,B,M,1)
    rmse_opt_1 = np.sqrt(np.mean((y_data_1 - y_pred_1[0]) ** 2))
    
    y_pred_N1 = model(gray_y, AL,AR,B,M,-1)
    rmse_opt_N1 = np.sqrt(np.mean((y_data_N1 - y_pred_N1[0]) ** 2))
    
    rmse_opt = 1*rmse_opt_0 + 1*rmse_opt_1 + 1*rmse_opt_N1
    return  rmse_opt       

# Function to update the initial guess dynamically based on previous results
def update_initial_guess(prev_results, tolerance=1e-5):
    if prev_results is not None:
        # Apply logic to update initial guess dynamically, e.g., by perturbing parameters slightly
        AL_opt, AR_opt, B_opt, M_opt = prev_results
        # Slight perturbation of optimal values
        initial_guess = [
            AL_opt + np.random.uniform(-0.5, 0.5),
            AR_opt + np.random.uniform(-0.5, 0.5),
            B_opt + np.random.uniform(-0.1, 0.1),
            M_opt + np.random.uniform(-0.05, 0.05)
        ]
    else:
        # Default initial guess if no previous results
        initial_guess = [4.067, 3.0394, 3.1, 0.8803]
    return initial_guess

# Initial guess and bounds
initial_guess = [4.067, 3.0394, 3.1, 0.8803]  # Initialize AL, AR, B, and M.
bounds = [(0, 6), (0, 5), (0, 3.1), (0, 0.9)]  # The bounds for AL, AR, B, and M.

# Fitting loop with L-BFGS-B method
tolerance = 1e-5  # Threshold for RMSE improvement
max_iterations = 20  # Maximum number of iterations
prev_results = None  # To store previous iteration results

for iteration in range(max_iterations):
    # Perform minimization with the current initial guess
    result = minimize(RMSE, initial_guess, method='L-BFGS-B', args=(gray_y, y_data, y_data_1, y_data_N1), bounds=bounds)
    
    AL_opt, AR_opt, B_opt, M_opt = result.x
    minimized_rmse = result.fun

    print(f"Iteration {iteration+1}: AL={AL_opt:.4f}, AR={AR_opt:.4f}, B={B_opt:.4f}, M={M_opt:.4f}, RMSE={minimized_rmse:.4f}")
    
    # Calculate individual RMSE values for logging
    rmse_0 = np.sqrt(np.mean((y_data - model(gray_y, AL_opt, AR_opt, B_opt, M_opt, 0)[0]) ** 2))
    rmse_1 = np.sqrt(np.mean((y_data_1 - model(gray_y, AL_opt, AR_opt, B_opt, M_opt, 1)[0]) ** 2))
    rmse_N1 = np.sqrt(np.mean((y_data_N1 - model(gray_y, AL_opt, AR_opt, B_opt, M_opt, -1)[0]) ** 2))
    print(f"RMSE values - 0th: {rmse_0:.4f}, 1st: {rmse_1:.4f}, -1st: {rmse_N1:.4f}")
    
    # Stop if all individual RMSEs meet tolerance criteria
    if all(abs(rmse) < tolerance for rmse in [rmse_0, rmse_1, rmse_N1]):
        print("Converged to minimal RMSEs for all orders")
        break

    # Update previous results and initial guess for the next iteration
    prev_results = [AL_opt, AR_opt, B_opt, M_opt]
    initial_guess = update_initial_guess(prev_results)
# Now we can call model funtion by inputing the optiaml AL, AR, B, and M to see how close our fitting is from the experimental results.
gray=np.arange(0,256,1)
print(AL_opt, AR_opt, B_opt,M_opt)
I_fitted = model(gray, AL_opt, AR_opt, B_opt,M_opt,0)
I_fitted_1 = model(gray, AL_opt, AR_opt, B_opt,M_opt,1)
I_fitted_N1 = model(gray, AL_opt, AR_opt, B_opt,M_opt,-1)

plt.figure()
plt.scatter(gray_y, y_data, label='Experimental Data m=0th', color='blue')
plt.plot(gray, I_fitted[0], label=f'Optimized Fit m=0th', color='green') #(RMSE = {rmse_opt:.4f})

plt.scatter(gray_y, y_data_1, label='Experimental Data m=1st', color='orange')
plt.plot(gray, I_fitted_1[0], label=f'Optimized Fit m=1st', color='pink')

plt.scatter(gray_y, y_data_N1, label='Experimental Data m=-1st', color='purple')
plt.plot(gray, I_fitted_N1[0], label=f'Optimized Fit m=-1st', color='red')
plt.legend()
ax_n = plt.gca()
ax_n.yaxis.set_minor_locator(MultipleLocator(0.02))
plt.legend(loc="best")
plt.grid(True, linestyle='--')
plt.ylim([0,1])
plt.xlim([0,255])
plt.show()
plt.xlabel('Gray Level')
plt.ylabel('Normlized Intensity')
plt.show()
