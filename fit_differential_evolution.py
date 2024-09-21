# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:24:53 2024

@author: gaosh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, differential_evolution

def linearModel(gray, B, M, A):
    # Generate x domain for the model, scaled based on the input gray length
    x = np.linspace(-0.5, 0.5, len(gray))
    
    # Phase profile for the linear
    f_linear = 2 * x
    f_1 = np.arctan(B * (f_linear / 2))
    max_value_1 = np.max(f_1)
    f_n = f_1 / max_value_1  # Normalized

    # Phase profile with the nonlinearity
    PhaseProfile = np.arctan(A * np.cos(f_linear * np.pi)) / np.pi
    max_value = np.max(PhaseProfile)
    PhaseProfile = PhaseProfile / max_value
    PhaseProfile_M = PhaseProfile * M

    # Smooth pixelation process: smooth transition instead of binary
    pixelation = 1 / (1 + np.exp(-10 * (np.abs(x) - 0.25)))  # Smooth transition around 0.25

    # Calculate intensity profile
    Im = []
    for g in gray:
        gray_M = M * g / 255
        phaseProfile_C = PhaseProfile_M * gray_M
        phase = pixelation * np.exp(1j * (phaseProfile_C * np.pi))
        cm = np.abs(np.sum(phase))  # Sum the phases and take the magnitude
        Im.append(cm)
    
    Im = np.array(Im)
    
    # Normalize output to ensure it's between 0 and 1
    Im = Im / np.max(Im)  # Normalize to the maximum value
    Im = np.clip(Im, 0, 1)  # Ensure all values are between 0 and 1
    
    return Im.real  # Only return the real part for fitting purposes

def calculate_rmse(y_true, y_pred):
    """Calculate the Root Mean Square Error (RMSE) between true and predicted values."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Test data
gray = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
y_data = [0.93, 0.898, 0.765, 0.643, 0.466, 0.292, 0.138, 0.0038, 0.011, 0.051, 0.133, 0.275,
          0.431, 0.592, 0.73, 0.831, 0.876]

# Objective function for differential_evolution
def objective_function(params, gray, y_data):
    B, M, A = params
    y_pred = linearModel(gray, B, M, A)
    # Return the sum of squared residuals
    return np.sum((y_data - y_pred) ** 2)

# Initial guess for the parameters
initial_guess = [9.8, 8.12, 32]  # B, M, A

# Set parameter bounds (all positive)
bounds = [(0, 10), (0, 10), (0, 5000)]  # Bounds for B, M, A

# Run differential_evolution
result = differential_evolution(objective_function, bounds=bounds, args=(gray, y_data))

# Extract optimized parameters
B_opt, M_opt, A_opt = result.x
y_fitted_opt = linearModel(gray, B_opt, M_opt, A_opt)

# Calculate RMSE between the optimized fit and y_data
rmse_opt = calculate_rmse(y_data, y_fitted_opt)

print(f"Optimized parameters after differential_evolution:\nB = {B_opt}\nM = {M_opt}\nA = {A_opt}")
print(f"Optimized RMSE = {rmse_opt}")

# Use the result of differential_evolution to refine using least_squares
# Note: Bounds for least_squares must be formatted as ([lower bounds], [upper bounds])
bounds_ls = (np.array([0, 0, 0]), np.array([10, 10, 5000]))  # Example bounds

result_ls = least_squares(objective_function, result.x, args=(gray, y_data), bounds=bounds_ls)

# Extract refined optimized parameters
B_refined, M_refined, A_refined = result_ls.x
y_fitted_refined = linearModel(gray, B_refined, M_refined, A_refined)

# Calculate RMSE for the refined fit
rmse_refined = calculate_rmse(y_data, y_fitted_refined)

print(f"Refined parameters after least_squares:\nB = {B_refined}\nM = {M_refined}\nA = {A_refined}")
print(f"Refined RMSE = {rmse_refined}")

# Plot the results
plt.figure(1)
plt.scatter(gray, y_data, label='Experimental Data', color='blue')
plt.plot(gray, y_fitted_opt, label=f'Differential Evolution Fit (RMSE = {rmse_opt:.4f})', color='green')
plt.plot(gray, y_fitted_refined, label=f'Refined Fit (RMSE = {rmse_refined:.4f})', color='red', linestyle='--')
plt.legend()
plt.xlabel('Gray Level')
plt.ylabel('Intensity')
plt.grid(True, linestyle='--')
plt.show()
