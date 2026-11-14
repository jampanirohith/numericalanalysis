import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return (np.exp(x))*(np.cos(x))

def forwarddifference(x,h):
    return (f(x+h)-f(x))/h

def backwarddifference(x,h):
    return (f(x)-f(x-h))/h

def centraldifference(x,h):
    return (f(x+h)-f(x-h))/(2*h)
 

x = 1

p = f(1)

h = []
fd = []
bd = []
cd = []
fd_error = []
bd_error = []
cd_error = []

for k in range (0,10):
    h.append(1/(2**(k+1)))

for i in range (0,10):
    fd.append(forwarddifference(x,h[i]))
    bd.append(backwarddifference(x,h[i]))
    cd.append(centraldifference(x,h[i]))

    fd_error.append(np.abs(fd[i]-p))
    bd_error.append(np.abs(bd[i]-p))
    cd_error.append(np.abs(cd[i]-p))

print('Forward Difference | Backward Difference | Central Difference\n')

for i in range(0,10):
    print(fd[i],' | ', bd[i], " | ", cd[i])

plt.loglog(h, fd_error)
plt.loglog(h, bd_error)
plt.loglog(h, cd_error)
plt.show()