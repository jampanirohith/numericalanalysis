import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return np.exp(x)/(1+(x**2))

def centraldifference(x,h):
    return (f(x+h)-f(x-h))/(2*h)

def df(x):
    return (np.exp(x) * (1 + x**2) - np.exp(x) * (2 * x)) / ((1 + x**2)**2)


x = 1
p = df(1)
h = []

dh = []
dh_error = []

dh2 = []

R = []
R_error = []

for k in range (0,10):
    h.append(2**(-1*(k+1)))

for i in range (0,10):
    dh.append(centraldifference(x, h[i]))
    dh_error.append(np.abs(dh[i]-p))

    dh2.append(centraldifference(x, h[i]/2))
    R.append((4*dh2[i]-dh[i])/3)
    R_error.append(np.abs(R[i]- p))

plt.loglog(h, dh_error)
plt.loglog(h, R_error)
plt.show()