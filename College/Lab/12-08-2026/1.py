import math

def f(x):
    return (math.exp(x)-1)/x

print("Values from 10^-1 to 10^15:\n")
for i in range(0,16):
    x = 10.0 ** -i
    print(f(x))

print("\nValues of 10^16 to 10^-20")
for i in range(15,21):
    x = 10.0 ** -i
    print(f(x))

