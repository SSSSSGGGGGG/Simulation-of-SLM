# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:46:53 2024

@author: gaosh
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import symbols, simplify
import matplotlib as mpl


mpl.rcParams['font.size'] = 15
phi= sp.symbols('phi')
# Define range for 'a'
a_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.96,1.0]  # Adjust range as needed

# Initialize an empty list to store solutions
solutions = []
solutions2 = []

# Iterate over 'a' values
for a in a_values:
    # Calculate f0 and f1 for the current 'a'
    f0 = a**2 * sp.cos((phi / 2) * sp.pi)**2
    f1 = a**2 * sp.sinc((a / 2) * sp.pi)**2 * sp.sin((phi / 2) * sp.pi)**2

    # Solve equation and apply modulo operation
    equation = sp.Eq(f1, f0)
    solution = sp.solve(equation, phi)
    # print(solution,"...")
    # Append the first solution (if exists) to the list
    if solution:
        solutions.append(solution[0] % 2)
        solutions2.append(solution[1] % 2)
    else:
        solutions.append(None)  # Append None if no solution found
        solutions2.append(None)
# Plot 'a' vs 'phi' for each solution
new_sol1=np.array(solutions)
new_sol1[10]=np.array(solutions2)[10]
new_sol2=np.array(solutions2)
new_sol2[10]=np.array(solutions)[10]
# print(new_sol1,"...",new_sol2)
plt.figure(figsize=(10, 6))
plt.plot(a_values, new_sol1, label='Solution1')
plt.plot(a_values, new_sol2, label='Solution2')
plt.xlabel('a')
plt.ylabel('phi (mod 2)')
plt.title('Solutions mod 2 for Different Values of a')
plt.legend()
plt.xlim([0.1,1])
plt.ylim([0.5,1.5])
plt.yticks(ticks=np.arange(0.4,1.7,0.1))
plt.grid(True)
plt.show()

sol_f0 = []
sol_f1 = []
sol2_f0 = []
sol2_f1 = []
sum1_tri = []
sum2_tri = []
sum1_tri_a=[]
sum2_tri_a=[]
# Iterate over the values of 'a'
for j,i in enumerate(a_values) :
    phi1 = new_sol1.astype(float)
    phi2 = new_sol2.astype(float)
    f0_inverse=i**2 * np.cos((phi1[j] / 2) * np.pi)**2
    f1_inverse = i**2 * np.sinc((i / 2) )**2 * np.sin((phi1[j] / 2) * np.pi)**2
    f0_inverse2=i**2 * np.cos((phi2[j] / 2) * np.pi)**2
    f1_inverse2 = i**2 * np.sinc((i / 2) )**2 * np.sin((phi2[j] / 2) * np.pi)**2
    print(f"var1={f0_inverse-f1_inverse},var2={f0_inverse2-f1_inverse2}")
    sum1=f0_inverse+2*f1_inverse
    sum2=f0_inverse2+2*f1_inverse2
    sum1_tri.append(sum1)
    sum2_tri.append(sum2)
    sum1_tri_a.append(sum1/(i**2))
    # print(sum1/(i**2))
    sum2_tri_a.append(sum2/(i**2))
    sol_f0.append(f0_inverse)  # Append None if no solution found
    sol_f1.append(f1_inverse)
    sol2_f0.append(f0_inverse2)  # Append None if no solution found
    sol2_f1.append(f0_inverse2)
    
# Print the dictionaries (optional)
# print(f"I1_tri0={sol_f0},I1_tri1={sol_f1},I2_tri0={sol2_f0},I2_tri1={sol2_f1}")
# print(f"I1_tri_tot={sum1_tri},I2_tri_tot={sum2_tri}")
plt.figure(figsize=(10, 6))
plt.plot(a_values, sum1_tri, label='diffraction efficiency for sol1',marker="*")
plt.plot(a_values, sum2_tri, label='diffraction efficiency for sol2')
plt.xlabel('a')
plt.ylabel('different efficiency')
plt.title('DE of triplicator for Different Values of a')
plt.legend()
plt.xlim([0.1,1])
plt.minorticks_on()
plt.yticks(ticks=np.arange(0,1.1,0.1))
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(a_values, sum1_tri_a, label='diffraction efficiency for sol1',marker="*")
plt.plot(a_values, sum2_tri_a, label='diffraction efficiency for sol2')
plt.xlabel('a')
plt.ylabel('diffraction efficiency')
plt.title('DE of triplicator for Different Values of a is divided by the square of a')
plt.legend()
plt.xlim([0.1,1])
plt.minorticks_on()
plt.yticks(ticks=np.arange(0.8,1.5,0.1))
plt.grid(True)
plt.show()