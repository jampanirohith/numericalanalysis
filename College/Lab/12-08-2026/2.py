x = [-2, 0, -1, 1, 2]
y = [4, 1, -1, 1, 9]

def lagrannge(alpha):
    n = len(x)
    p = 0
    
    for i in range(n):
        l_i = 1.0
        for j in range(n):
            if j != i:
                l_i *= (alpha - x[j])/(x[i] - x[j])

        p += y[i] * l_i

    return p


alpha = float(input("Enter aplha: "))

print(lagrannge(alpha))
