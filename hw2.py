import numpy as np

def LU_decomposition(A_init):
    A = A_init
    eps = 10 ** -10
    for i in range(0, n):
        maxA = 0
        imax = i

        for k in range(i, n):
            absA = abs(A_init[k][i])
            if absA > maxA:
                maxA = absA
                imax = k

        if maxA < eps:
            return 0

        if imax != i:
            j = A[i]
            A[i] = A[imax]
            A[imax] = j

            ptr = A_init[i]
            A[i] = A[imax]
            A[imax] = ptr

        for j in range(i+1, n):
            A_init[j][i] /= A_init[i][i]

            for k in range(i+1, n):
                A_init[j][k] -= A_init[j][i] * A_init[i][k]

    return A


def A_determinant(A_init):
    A_LU = LU_decomposition(A_init)
    # det(A) = det(L) * det(U)  si  det(L) = 1  =>  det(A) = det(U)
    # det(U) = produsul elementelor de pe diagonala principala
    det = 1
    for i in range(0, len(A_LU)):
        det *= A_LU[i][i]
    return det


def direct_substitution(A_init, b):
    y = [0] * len(A_init)

    for i in range(len(A_init)):
        y[i] = (b[i] - sum([A_init[i][j] * y[j] for j in range(i)]))
    return y


def inverse_substitution(A_init, b):
    x = [0] * len(A_init)
    for i in range(n-1, -1, -1):
        for i in range(len(A_init)):
            x[i] = (b[i] - sum([A_init[i][j] * x[j] for j in range(i+1, n)])) / A_init[i][i]
    return x


def matrix_inverse(A, n):
    A_inverse = np.zeros((n, n))
    for i in range(n):
        Ei = np.zeros(n)
        Ei[i] = 1

        Ly = direct_substitution(A, Ei.T)
        Ux = inverse_substitution(A, Ly)

        A_inverse[:, i] = Ux
    return A_inverse


A_init = [[2.5, 2.0, 2.0],
     [5.0, 6.0, 5.0],
     [5.0, 6.0, 6.5]]
b = [2, 2, 2]
n = len(A_init)

print("Initial A:")
for line in A_init:
    print(line)
print('################################################################')

print("LU decomposition:")
A_LU = LU_decomposition(A_init)
for line in A_LU:
    print(line)
print('################################################################')

A_init = [[2.5, 2.0, 2.0],
     [5.0, 6.0, 5.0],
     [5.0, 6.0, 6.5]]
print("Determinant:")
print(A_determinant(A_init))
print('################################################################')

print("Direct Substitution:")
Y = direct_substitution(A_LU, b)
print(Y)
print("Inverse Substitution:")
X = inverse_substitution(A_LU, Y)
print(X)
print('################################################################')

A_init = [[2.5, 2.0, 2.0],
     [5.0, 6.0, 5.0],
     [5.0, 6.0, 6.5]]
A_original = np.array(A_init)
X_original = np.array(X)
print(A_original)
Mul = np.dot(A_original, X_original)
print(Mul)
print("A_init * X_LU - b_init:")
norm = Mul - b
print(sum(i**2 for i in norm))
print('################################################################')

print("X_LU - X_lib:")
X_lib = np.linalg.solve(A_original, b)
print(sum(i**2 for i in (X_original-X_lib)))
print('################################################################')

A_init = [[2.5, 2.0, 2.0],
     [5.0, 6.0, 5.0],
     [5.0, 6.0, 6.5]]
print("X_LU - A_inverse_lib * b_init:")
A_inverse_lib = np.linalg.inv(A_init)
A_inverse_lib_minus_b = A_inverse_lib - np.array(b)
X_LU_minus_A_inverse_lib_minus_b = X_original - A_inverse_lib_minus_b
print(sum(i**2 for i in X_LU_minus_A_inverse_lib_minus_b))
print('################################################################')

A_init = [[2.5, 2.0, 2.0],
     [5.0, 6.0, 5.0],
     [5.0, 6.0, 6.5]]
print("Matrix inverse:")
print(matrix_inverse(A_LU, n))
print(A_inverse_lib)
print('################################################################')

print("A_inverse_LU - A_inverse_lib:")
A_inverse_LU = np.array(matrix_inverse(A_LU, n))
A_inverse_LU_minus_A_inverse_lib = A_inverse_LU - A_inverse_lib
print(sum(i**2 for i in A_inverse_LU_minus_A_inverse_lib))