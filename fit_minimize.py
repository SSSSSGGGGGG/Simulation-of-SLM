import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def linearModel(gray, B, M, A):
    x = np.linspace(-0.5, 0.5, len(gray))
    x_=np.linspace(0, 0.5,len(gray))
    p=1
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
    a=0.964
    Im = []
    for g in gray:
        gray_M = M * g / 255
        phaseProfile_C = PhaseProfile_n * gray_M
        phase = np.exp(1j * (phaseProfile_C * np.pi))
        cm = np.abs(np.sum(phase))
        Im.append(cm)

    Im = np.array(Im)
    Im = (a**2)*(Im / np.max(Im))
    return Im.real

def objective_function(params, gray, y_data):
    B, M, A = params
    y_pred = linearModel(gray, B, M, A)
    weights = 1 / (1 + (y_data - y_pred) ** 2)
    return np.sum(weights * (y_data - y_pred) ** 2)

# Test data
gray = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
y_data = np.array([0.93, 0.898, 0.765, 0.643, 0.466, 0.292, 0.138, 0.0038, 0.011, 0.051, 0.133, 0.275,
                   0.431, 0.592, 0.73, 0.831, 0.876])

# Optimization loop to ensure RMSE is below the threshold
target_rmse = 0.01
rmse_opt = 0.1
initial_guess = [0.1, 1, 1000]
bounds = [(0, 10), (0, 1.2), (0, 1000)]

# Store optimized parameters for future use
optimized_params_history = []

# Set maximum iterations
max_iterations = 100
iteration = 0

while rmse_opt > target_rmse and iteration < max_iterations:
    result = minimize(objective_function, initial_guess, method='SLSQP',args=(gray, y_data), bounds=bounds)
    B_opt, M_opt, A_opt = result.x
    y_fitted_opt = linearModel(gray, B_opt, M_opt, A_opt)
    rmse_opt = np.sqrt(np.mean((y_data - y_fitted_opt) ** 2))

    print(f"Iteration {iteration + 1}: RMSE = {rmse_opt:.6f}, Parameters: B={B_opt}, M={M_opt}, A={A_opt}")

    # Store the optimized parameters
    optimized_params_history.append((B_opt, M_opt, A_opt))

    if rmse_opt > target_rmse:
        initial_guess = [B_opt * 0.95, M_opt * 0.95, A_opt * 0.95]  # Update initial guess
    iteration += 1

if rmse_opt <= target_rmse:
    print(f"Converged to target RMSE: {rmse_opt:.6f}")
else:
    print("Reached maximum iterations without converging.")

print(f"Optimized parameters:\nB = {B_opt}\nM = {M_opt}\nA = {A_opt}")
print(f"Optimized RMSE = {rmse_opt:.6f}")

# Plot the results
plt.figure()
plt.scatter(gray, y_data, label='Experimental Data', color='blue')

# Generate a smooth fitted curve
smooth_gray = np.linspace(0, 255, 10000)
y_fitted_smooth = linearModel(smooth_gray, B_opt, M_opt, A_opt)

plt.plot(smooth_gray, y_fitted_smooth, label=f'Optimized Fit (RMSE = {rmse_opt:.4f})', color='green')
plt.legend()
plt.xlabel('Gray Level')
plt.ylabel('Intensity')
plt.grid(True, linestyle='--')
plt.show()

# Optionally: Save optimized parameters to a file for future runs
# np.savetxt('optimized_params.txt', optimized_params_history)
