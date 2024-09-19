# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""
import sympy as sp
# Define the symbol
n = sp.Symbol('n')

# Define the terms of the series
def term(n):
    if n % 4 == 0 or n % 4 == 1:
        return 1 / (2*n + 1)
    else:
        return -1 / (2*n + 1)

# Define the sum of the series as n goes to infinity
series_sum = sum(term(n) for n in range(10))  # Summing up to a large value, e.g., 10000

# Print the result
print("The sum of the alternating series as n approaches infinity is:", series_sum*(2/sp.pi))

# Print the evaluated sum
print("The sum of the alternating series is:", series_sum)