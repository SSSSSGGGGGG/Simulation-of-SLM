import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

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

# Initial guess for the parameters
initial_guess = [0.1, 2, 10]  # B, M, A

# Add bounds to ensure positive values of B, M, and A
bounds = (0, [np.inf, np.inf, np.inf])

# Step 1: Perform the curve fitting with bounds using curve_fit
popt, pcov = curve_fit(linearModel, gray, y_data, p0=initial_guess, bounds=bounds)

# Get the fitted values from the model
y_fitted = linearModel(gray, *popt)

# Calculate RMSE between the fitted curve and the actual y_data
rmse = calculate_rmse(y_data, y_fitted)
print(f"Initial fit parameters using curve_fit:\nB = {popt[0]}\nM = {popt[1]}\nA = {popt[2]}")
print(f"Initial RMSE = {rmse}")

# Step 2: Minimize RMSE using scipy's minimize
def objective(params):
    # Objective function to minimize RMSE
    B, M, A = params
    y_pred = linearModel(gray, B, M, A)
    return calculate_rmse(y_data, y_pred)

# Set parameter bounds (all positive)
bounds = [(0, np.inf), (0, np.inf), (0, np.inf)]

# Run optimization to minimize RMSE
result = minimize(objective, popt, bounds=bounds)

# Extract optimized parameters
B_opt, M_opt, A_opt = result.x
y_fitted_opt = linearModel(gray, B_opt, M_opt, A_opt)
rmse_opt = calculate_rmse(y_data, y_fitted_opt)

print(f"Optimized parameters after minimizing RMSE:\nB = {B_opt}\nM = {M_opt}\nA = {A_opt}")
print(f"Optimized RMSE = {rmse_opt}")

# Plot the results
plt.figure(1)
plt.scatter(gray, y_data, label='Experimental Data', color='blue')
plt.plot(gray, y_fitted, label=f'Initial Fit (RMSE = {rmse:.4f})', color='red')
plt.plot(gray, y_fitted_opt, label=f'Optimized Fit (RMSE = {rmse_opt:.4f})', color='green')
plt.legend()
plt.xlabel('Gray Level')
plt.ylabel('Intensity')
plt.grid(True, linestyle='--')
plt.show()
