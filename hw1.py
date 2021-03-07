import numpy as np
import time
import random


def ex1(m):
    u = 10 ** -m
    if 1 + u == 1:
        return True
    return False


def NUM(v):
    res = int("".join(str(x) for x in v), 2)
    return res


def four_russians(A, B, n):
    m = int(np.floor(np.log(n)))
    p = int(np.ceil(n / m))
    C = np.zeros((n, n))

    suma_B = np.zeros((int(2 ** m), n))
    for i in range(p):
        for j in range(1, 2 ** m):
            k = int(np.floor(np.log2(j))) # j = 2^k
            Bi = B[i*m:(i+1)*m] # linia i din B
            suma_B[j] = np.logical_or(suma_B[j - int(2 ** k)], Bi[k])
        Ci = np.zeros((n, n))
        for r in range(n):
            Ai = A[:, i*m:(i+1)*m] # coloana i din A
            Ci[r] = suma_B[NUM(Ai[r])]
        C = np.logical_or(C, Ci).astype(np.int)
    return C.astype(np.int)


if __name__ == '__main__':
    # ex 1
    for i in range(20):
        u = 10 ** (-i)
        if 1 + u == 1:
            print(u)
            break
    print('################################################################')

    # ex 2
    for i in range(1000):
        x = random.random()
        y = random.random()
        z = random.random()
        left = x + y + z
        right = y + z + x
        left_mul = x * y * z
        right_mul = y * z * x

        if left != right or left_mul != right_mul:
            print(f'x = {x}, y = {y}, z = {z}')
            print(f'steps = {i}')
            break
    print('################################################################')

    # ex 3
    n = 100
    A = np.random.randint(2, size=(n, n))
    B = np.random.randint(2, size=(n, n))

    print('Four russians:')
    start_f = time.time()
    print(four_russians(A, B, n))
    finish_f = time.time()
    print(finish_f - start_f)

    print('Numpy:')
    start_n = time.time()
    print(np.dot(A, B).astype(np.bool).astype(np.int))
    finish_n = time.time()
    print(finish_n - start_n)