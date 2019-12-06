import numpy as np
import pandas as pd
import sys

def Inverse(A):
    """ 使用G-J方法进行消元求逆矩阵.
    """
    dim = A.shape[0]
    AI = np.hstack([A, np.identity(dim)])
    
    for row in range(dim):
        """ 选择主元, 并将主元变为1.
        """
        flag = 0
        if AI[row, row] == 0:
            for row_ in range(row+1, dim):
                if AI[row_, row] != 0:
                    flag = 1
                    t = AI[row, :].copy()
                    AI[row, :] = AI[row_, :].copy()
                    AI[row_, :] = t.copy()
        else:
            flag = 1
        if flag == 0:
            print('求解中断, 矩阵:\n{}\n不可逆.'.format(A))
            exit(1)

        pivot = AI[row, row]
        AI[row, :] = AI[row, :] / pivot

        for row_ in range(dim):
            """ 对当前列进行消元.
            """
            if row_ == row:
                continue
            cur = AI[row_, row]
            AI[row_, :] = AI[row_, :] - cur * AI[row, :]

    return AI[:, dim:2*dim]

def LU(A):
    """ 使用部分主元法求解PA=LU.
    args:
        A: 需要进行分解的矩阵.
    return:
        P, L, U: PA=LU.
    """
    dim  = A.shape[0]
    P = np.arange(dim).reshape(dim, 1)
    AP = np.concatenate((A, P), axis=1)

    for i in range(dim):
        max_idx = i
        pivot = AP[i, i]
        for row in range(i+1, dim):
            if abs(AP[row, i]) > abs(pivot):
                max_idx = row
                pivot = AP[row, i]

        if pivot == 0:
            print('求解中断, 矩阵:\n{}\n不可分解.'.format(A))
            exit(1)

        t = AP[i, :].copy()
        AP[i, :] = AP[max_idx, :].copy()
        AP[max_idx, :] = t.copy()

        for row in range(i+1, dim):
            AP[row, i] = AP[row, i] / AP[i, i]
            AP[row, i+1:dim] -= AP[i, i+1:dim] * AP[row, i]

    P = np.zeros_like(A)
    for row in range(dim):
        P[row, int(AP[row, dim])] = 1
    
    L = np.identity(dim)
    U = np.zeros_like(A)
    for row in range(dim):
        for col in range(row, dim):
            U[row, col] = AP[row, col]
    
    for col in range(dim):
        for row in range(col+1, dim):
            L[row, col] = AP[row, col]

    P = np.round(P, 6)
    L = np.round(L, 6)
    U = np.round(U, 6)
    return P, L, U

def GS(A):
    """使用经典Gram-Schmidt对矩阵A做QR分解.
    args:
        A: 需要进行分解的矩阵.
    return:
        Q, R: A=QR, 其中Q, R中的小数保留小数点后6位.
    """
    Q = np.zeros((A.shape[0], A.shape[0]))
    R = np.zeros_like(A)

    for k in range(A.shape[1]):
        Q[:, k] = A[:, k]
        for row in range(k):
            Q[:, k] -= R[row, k] * Q[:, row]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] = Q[:, k] / R[k, k]
        for col in range(k+1, A.shape[1]):
            R[k, col] = np.dot(Q[:, k], A[:, col])
    
    Q = np.round(Q, 6)
    R = np.round(R, 6)
    return Q, R

def Householder(A):
    """使用Householder约减对矩阵A做QR分解.
    args:
        A: 需要进行分解的矩阵.
    return:
        Q, R: A=QR, 其中Q, R中的小数保留小数点后6位.
    """
    dim = A.shape[1]
    Q = np.identity(A.shape[0])
    for k in range(dim):
        H = np.identity(A.shape[0])
        e1 = np.zeros((A.shape[0]-k))
        e1[0] = 1
        u = A[k:, k] - np.linalg.norm(A[k:, k]) * e1
        if np.dot(u, u) != 0:
            H[k:, k:] -= 2 * np.matmul(u.reshape(A.shape[0]-k, 1), u.reshape(1, A.shape[0]-k)) / np.dot(u, u)
        A = np.matmul(H, A)
        Q = np.matmul(H, Q)

    Q = np.round(Q, 6)
    A = np.round(A, 6)
    return np.transpose(Q), A
    
def Givens(A):
    """使用经典Givens约减对矩阵A做QR分解.
    args:
        A: 需要进行分解的矩阵.
    return:
        Q, R: A=QR, 其中Q, R中的小数保留小数点后6位.
    """
    Q = np.identity(A.shape[0])
    for col in range(A.shape[1]):
        for row in range(col+1, A.shape[0]):
            G = np.identity(A.shape[0])
            t = np.sqrt(A[col, col]**2+A[row, col]**2)
            G[col, col] = A[col, col] / t
            G[col, row] = A[row, col] / t
            G[row, col] = -A[row, col] / t
            G[row, row] = A[col, col] / t

            Q = np.matmul(G, Q)
            A = np.matmul(G, A)

    Q = np.round(Q, 6)
    A = np.round(A, 6)
    return np.transpose(Q), A

if __name__ == '__main__':
    input_data = pd.read_csv('input.csv')
    A = np.array(input_data, dtype=np.float32)
    
    mode = list(sys.argv)[1]
    if mode == 'GS':
        Q, R = GS(A)
        print('GS A=QR\n{}\n=\n{}\n{}'.format(A, Q, R))
    elif mode == 'Householder':
        Q, R = Householder(A)
        print('householder A=QR\n{}\n=\n{}\n{}'.format(A, Q, R))
    elif mode == 'Givens':
        Q, R = Givens(A)
        print('Givens A=QR\n{}\n=\n{}\n{}'.format(A, Q, R))
    elif mode == 'LU':
        P, L, U = LU(A)
        print('LU PA=LU\n{}\n{}\n=\n{}\n{}'.format(P, A, L, U))
    else:
        print('args {} error\n'.format(mode))