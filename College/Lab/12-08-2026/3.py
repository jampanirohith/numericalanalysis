import matplotlib.pyplot as plt
import numpy as np

x = [0, 1, 1.5, 2, 3]
y = [2, 8, 12, 10, 20]

def lagrange(alpha):
    n = len(x)
    p = 0
    
    for i in range(n):
        l_i = 1.0
        for j in range(n):
            if j != i:
                l_i *= (alpha - x[j])/(x[i] - x[j])

        p += y[i] * l_i

    return p

a = np.linspace(0, 3, 100)
b = lagrange(a)

plt.plot(a, b)
plt.show()

