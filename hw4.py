from tema3 import get_matrix
import numpy as np

def get_free_vector(file):
    with open(file, 'r') as f:
        n = int(f.readline().strip())
        matrix = [] * n
        for line in range(n):
            matrix.append(float(f.readline().strip()))

    return matrix


def check_diagonal(matrix):
    count = 0
    n = 54321
    eps = 10 ** -10
    for i, line in enumerate(matrix):
        found = False
        for nr in line:
            if nr[1] == i:
                found = True
                if abs(nr[0]) > eps:
                    count += 1
        if not found:
            return False
    if count == n:
        return True
    return False


def formula3(A, b, x):
    delta_x = 0
    for i, line in enumerate(A):
        s1 = 0
        s2 = 0
        a_diag_element = None
        for element in line:
            j = element[1]  # column
            value = element[0]  # value
            if i == j:
                a_diag_element = element[0]
            elif j < i:
                s1 += value * x[j]
            else:
                s2 += value * x[j]
        x_i = (b[i] - s1 - s2) / a_diag_element
        delta = abs(x_i - x[i]) ** 2
        delta_x += delta
        x[i] = x_i
    return delta_x

def gauss_seidel(A, b):
    x_GS = np.zeros(len(b))
    k = 0
    eps = 10 ** -10

    while True:
        k += 1
        delta_x = formula3(A, b, x_GS)
        print(delta_x)
        if k >= 10000 or delta_x < eps or delta_x > (10 ** 8):
            break
    if delta_x < eps:
        return x_GS
    else:
        return -1


def product(A, x):
    b = np.zeros(len(A))
    for i, line in enumerate(A):
        total = 0
        for elem in line:
            j = elem[1]
            value = elem[0]
            total += value * x[j]
        b[i] = total

    return b


def norm(AxGS, b):
    res = np.zeros(len(AxGS))
    for i in range(len(AxGS)):
        res[i] = AxGS[i] - b[i]
    return res


A = get_matrix("a_5.txt")
# print(A)
b = get_free_vector("b_5.txt")
# print(b)

# print(check_diagonal(A))

# x_GS = gauss_seidel(A, b)
# print(x_GS)
# if x_GS != -1:
#     AxGS = product(A, x_GS)
#     # print(AxGS)
#
#     print(norm(AxGS, b))
# else:
#     print("divergenta")
