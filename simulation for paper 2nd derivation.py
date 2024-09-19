# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:08:29 2024

@author: gaosh
"""
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from scipy.stats import norm
import sympy as sp
from sympy import latex

from sympy import init_printing

# Define symbolic variables
N,delta,u,a, phi = sp.symbols('N delta u a phi')
# N=m+kn

# Define the constraint equation
n = sp.Symbol('n', integer=True)
k = sp.Symbol('k', integer=True) 
m = sp.Symbol('m', integer=True) # Define n as an integer
constraint_eq = sp.Eq(n + k * m, sp.Integer(0))

# Define the expression to be summed
expr = (-1)**n * a * (-sp.sin(sp.pi * m / 2) * sp.sin(sp.pi * phi / 2) +
                        sp.cos(sp.pi * m / 2) * sp.cos(sp.pi * phi / 2)) * sp.sinc(m/2) * sp.sinc(sp.pi*a * (k * n + m) / k)*sp.DiracDelta(u-(N/k/delta))

# Compute the summation for each solution of n

result_sub=expr.subs({k:16,N:2,m:2-2*n}) #!!!! change N and m at the same time.
result = sp.Sum(sp.Sum(result_sub, (n, -sp.oo, sp.oo)),(m, -sp.oo, sp.oo)) #(n, -sp.oo, sp.oo)),(m, -sp.oo, sp.oo)
# result=sum(result_sub for n in range(-100, 101) for m in range(-100, 101))
#final_result = sum(result_sub.subs({n: i}) for i in range(-50, 50))# numerical evaluate

G=sp.simplify(result)
print("G=")
# sp.pprint(sp.simplify(result_sub))
sp.pprint(G)
# sp.pprint(result_sub.doit().simplify())
# sp.pprint(result_sub.evalf())
print("I=")

sp.pprint(G**2)

expr_latex=sp.latex(result_sub**2)
# print(expr_latex)
combined1 = r'$\sum_{-\infty \leq n \leq \infty, -\infty \leq m \leq \infty}'
G_latex =combined1 +" "+ expr_latex +"$"

#G_latex=r'$\sum_{-\infty \leq n \leq \infty, -\infty \leq m \leq \infty}\ a *\cos{\left(\frac{\pi \phi}{2} \right)} \delta\left(u\right) \operatorname{sinc}{\left(n \right)}$'
plt.text(-0.15, 0.5,G_latex, fontsize=15, ha='left') #r'$%s$' % expr_latex
plt.axis('off')  # Turn off axes
plt.savefig('expression_image.jpg')
plt.show()




# print(f"the period is {k}, the order {N} is calculated, m={m} is for grating order, and n={n} is for SLM.")
# print(f"the spatial frequences are {u}")

# a_sinc=a*sp.sinc((a/k)*(m+k*n))
# cos=sp.cos(0.5*m*sp.pi)*sp.cos(0.5*phi*sp.pi)
# sin=sp.sin(0.5*m*sp.pi)*sp.sin(0.5*phi*sp.pi)
# diff=cos-sin

# G=a_sinc*diff*sp.sinc(0.5*m)*pow(-1.0,n)

# print(G)
# G_sum=sp.Sum(G)
# Intensity=G_sum**2


