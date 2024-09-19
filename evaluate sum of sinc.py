# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import sympy as sp

n = sp.Symbol('n', integer=True)
# a = sp.Symbol('a', integer=True)
expr_sinc=sp.sinc((2*n-1/2)*sp.pi)**2

evaluate_sinc=sp.Sum(expr_sinc, (n, -1000, 1000))
print(f"The value of summation of {expr_sinc}")
sp.pprint(evaluate_sinc.doit())

#.doit()