import math
import numpy as np

def fact(n):
    if n == 1:
        return 1
    elif n > 1 :
        return n*fact(n-1)

i = (((2*pow(2,0.5))/9801)*(1103))
j = i + (((2*pow(2,0.5))/9801)*(fact(4)*(1103+26390*4))/pow(396, 4))


print("For N=0, value of pi is", 1/i)
print("For N=1, value of pi is", 1/j)
print("Python's stored value of pi is", np.pi)