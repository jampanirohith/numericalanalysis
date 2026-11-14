import numpy as np
import math

def f(x):
    return np.exp(x)*np.sin(2*x) + pow(x,3)*np.log(x+1)

x = 0.5
exact_val = 1.5620180519521627

D2 = {}
for k in range(1, 12):
    h = 2.0**(-k)
    D2[k] = (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


print(f"{'k':<3} | {'h':<9} | {'D2 Error':<10} | {'R1 Error':<10} | {'R2 Error':<10}")
print("-" * 55)

for k in range(1, 10):
    h = 2.0**(-k)
    
    d2_h = D2[k]
    d2_half = D2[k + 1]
    d2_quarter = D2[k + 2]
    
    r1_h = (4 * d2_half - d2_h) / 3
    r1_half = (4 * d2_quarter - d2_half) / 3
    
    r2_h = (16 * r1_half - r1_h) / 15
    
    err_d2 = abs(d2_h - exact_val)
    err_r1 = abs(r1_h - exact_val)
    err_r2 = abs(r2_h - exact_val)
    
    print(f"{k:<3} | {h:<9.6f} | {err_d2:<10.2e} | {err_r1:<10.2e} | {err_r2:<10.2e}")

