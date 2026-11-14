import numpy as np

def f(x):
    return np.sin(x)**2-2*x*np.sin(x) + 1
def F(x):
    return 1.5 * x - 0.25 * np.sin(2 * x) + 2 * x * np.cos(x) - 2 * np.sin(x)

a = 0.75
b = 1.3

exact = F(b) - F(a)

def trapezodial(h1,a, b):
    return h1/2*(f(a)+f(b))

def simpson(h2,a,b):
    return h2/3*(f(a)+f(b)+4*f((a+b)/2))

h1 = (b-a)
h2 = (b-a)/2



trapezod = trapezodial(h1,a,b)
err_trap = abs(exact - trapezod)

simp = simpson(h2,a,b)
err_simp = abs(exact - simp)

print("Exact Value:", exact)
print("Trapezoidal:", trapezod, "Error:",err_trap)
print("Simpson's 1/3:",  simp, "Error:", err_simp)
