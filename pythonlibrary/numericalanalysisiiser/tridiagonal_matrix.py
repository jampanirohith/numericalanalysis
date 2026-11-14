import numpy as np


def tridiagonal_matrix(n, diag=4.0, off=-1.0):
    """n x n matrix with diag on the main diagonal and off on the two neighbouring diagonals."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = diag
        if i > 0:
            A[i, i - 1] = off
        if i < n - 1:
            A[i, i + 1] = off
    return A
