import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return np.log((x**2)+1) + np.sin(x**2)

def centraldifference(x,h):
    return (f(x+h)-f(x-h))/(2*h)

def df(x):
    return (2*x)/((x**2)+1) + 2*x*np.cos(x**2)

x = 0.8
cd = []
cd_error = []

p = df(0.8)

for i in range (0,10):
    cd.append(centraldifference(x, 10**(-i)))
    cd_error.append(np.abs(cd[i]-p))

 