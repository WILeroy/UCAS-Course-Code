import numpy as np

def Elimination(A, pivot_row, pivot_col):
    pivot = A[pivot_row, pivot_col]
    for i in range(A.shape[0]):
        if i == pivot_row:
            continue
        else:
            A[i, :] -= (A[i, pivot_col] / pivot) * A[pivot_row, :]
    A[pivot_row, :] /= pivot
    return A