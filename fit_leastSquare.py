# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:23:34 2024

@author: gaosh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def linearModel(gray, B, M, A):
    # Generate x domain for the model, scaled based on the input gray length
    x = np.linspace(-0.5, 0.5, len(gray))
    x_=np.linspace(0, 0.5,320)
    # Phase profile for the linear
    p=1
    f_linear = 2 * x/p
    f_1 = np.arctan(B * (f_linear / 2))
    max_value_1 = np.max(f_1)
    f_n = f_1 / max_value_1  # Normalized

    f_=np.arctan(B*(x_/p))
    max_index_ = np.argmax(f_)
    max_value_ = f_[max_index_]
    f_L=f_/max_value_
    AL=A
    PhaseProfile_L=np.arctan(AL*np.cos(f_L*np.pi))/(np.pi)
    max_index_L = np.argmax(PhaseProfile_L)
    max_value_L = PhaseProfile_L[max_index_L]
    PhaseProfile_L=PhaseProfile_L/max_value_L

    # Phase profile for the negtive/left            
    f_r=np.arctan(B*(-x_/p))
    min_index_r = np.argmin(f_r)
    min_value_r = f_r[min_index_r]
    f_R=f_/min_value_r
    AR=A
    PhaseProfile_R=np.arctan(AR*np.cos(f_R*np.pi))/(np.pi)
    max_index_R = np.argmax(PhaseProfile_R)
    max_value_R = PhaseProfile_R[max_index_R]
    PhaseProfile_R=PhaseProfile_R/max_value_R
    PhaseProfile_n=np.concatenate((PhaseProfile_L[::-1], PhaseProfile_R))
    # Smooth pixelation process: smooth transition instead of binary
    a=0.964
    # pixelation=[0] * len(x)
    # for i in range(len(x)):
    #     if x[i]<len(x)/2 and x[i]>a/2:
    #         pixelation[i]=0
    #     elif x[i]>-len(x)/2 and x[i]<-a/2:
    #         pixelation[i]=0
    #     else:
    #         pixelation[i]=1  # Smooth transition around 0.25

    # Calculate intensity profile
    Im = []
    for g in gray:
        gray_M = M * g / 255
        phaseProfile_C = PhaseProfile_n * gray_M
        phase = np.exp(1j * (phaseProfile_C * np.pi))#pixelation * 
        cm = np.abs(np.sum(phase))  # Sum the phases and take the magnitude
        Im.append(cm)
    
    Im = np.array(Im)
    
    # Normalize output to ensure it's between 0 and 1
    Im = (a**2)*(Im / np.max(Im) ) # Normalize to the maximum value
    # Im = np.clip(Im, 0, 1)  # Ensure all values are between 0 and 1
    
    return Im.real  # Only return the real part for fitting purposes

def calculate_rmse(y_true, y_pred):
    """Calculate the Root Mean Square Error (RMSE) between true and predicted values."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Test data
gray = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
y_data = [0.93, 0.898, 0.765, 0.643, 0.466, 0.292, 0.138, 0.0038, 0.011, 0.051, 0.133, 0.275,
          0.431, 0.592, 0.73, 0.831, 0.876]

# Define the objective function for least_squares, returning the residuals (difference between model and data)
def residuals(params, gray, y_data):
    B, M, A = params
    y_pred = linearModel(gray, B, M, A)
    return y_data - y_pred

# Initial guess for the parameters
initial_guess = [0.1, 1.001, 999]  # B, M, A

# Set parameter bounds (all positive)
bounds = ([0, 0, 0], [2, 2, 1000])

# Run least_squares to minimize the residuals
result = least_squares(residuals, initial_guess, args=(gray, y_data), bounds=bounds)

# Extract optimized parameters
B_opt, M_opt, A_opt = result.x
y_fitted_opt = linearModel(gray, B_opt, M_opt, A_opt)

# Calculate RMSE between the optimized fit and y_data
rmse_opt = calculate_rmse(y_data, y_fitted_opt)

print(f"Optimized parameters after least_squares:\nB = {B_opt}\nM = {M_opt}\nA = {A_opt}")
print(f"Optimized RMSE = {rmse_opt}")

# Plot the results
plt.figure(1)
plt.scatter(gray, y_data, label='Experimental Data', color='blue')
plt.plot(gray, y_fitted_opt, label=f'Optimized Fit (RMSE = {rmse_opt:.4f})', color='green')
plt.legend()
plt.xlabel('Gray Level')
plt.ylabel('Intensity')
plt.grid(True, linestyle='--')
plt.show()
