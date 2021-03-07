import numpy as np
import random
from tema3 import get_matrix, get_matrix_transpose
from tema4 import product


def generate_matrix(n):
    matrix = [[] for _ in range(n)]
    epsilon = 10 ** -9

    for i in range(n):
        val = random.uniform(-50.0, 50.0)
        matrix[i].append([val, i])
        for j in range(i + 1, n):
            val = random.uniform(-50.0, 50.0)
            insert = random.randint(0, 15)
            if insert == 15 and (val > epsilon or val < -epsilon):
                matrix[i].append([val, j])
                matrix[j].append([val, i])

    f = open("generated_matrix.txt", "w")
    f.write(str(n) + '\n')
    for i, line in enumerate(matrix):
        for element in line:
            f.write(f'{str(element[0])}, {str(i)}, {str(element[1])}' + '\n')

    return matrix


def power_method(matrix):
    n = len(matrix)
    epsilon = 10 ** -9
    v = np.random.random(n)
    v *= 1 / np.linalg.norm(v)

    w = product(matrix, v)
    lambda0 = np.inner(w, v)
    k = 0
    while True:
        v = w * (1 / np.linalg.norm(w))
        w = product(matrix, v)
        lambda0 = np.inner(w, v)
        k += 1

        if (np.linalg.norm(w - lambda0 * v) <= n * epsilon) or k > 1000000:
            break
    if k > 1000000:
        print('k > kmax')
        return None, None
    else:
        # lambda0 = aproximare a unei valori proprii de modul maxim a matricei
        # v = aproximare a unui vector propriu asociat
        return lambda0, v


if __name__ == '__main__':
    # 1
    generated_matrix = generate_matrix(500)
    print(generated_matrix)

    # 2
    A = get_matrix('generated_matrix.txt')
    AT = get_matrix_transpose('generated_matrix.txt')
    if A == AT:
        print('A = AT')

    print('################################################################')
    lambda0, v = power_method(A)
    print(f'lambda = {lambda0} \n v = {v}')

    print('################################################################')
    a_500 = get_matrix('a_500.txt')
    a_500_lambda0, a_500_v = power_method(a_500)
    print(f'lambda = {a_500_lambda0} \n v = {a_500_v}')
    print('################################################################')

    # 3
    epsilon = 10 ** -9
    regular_matrix = np.random.random((501, 500)).astype(np.float)
    b = np.random.random((501, 1)).astype(np.float)

    u, s, vh = np.linalg.svd(regular_matrix)
    print(f'singular values = {s}')

    rank_svd = s[s > epsilon]
    print(f'rank (SVD method) = {len(rank_svd)}')

    rank = np.linalg.matrix_rank(regular_matrix)
    print(f'rank = {rank}')

    min_s = min(rank_svd)
    max_s = max(rank_svd)
    cond_svd = max_s / min_s
    print(f'condition number (SVD method) = {cond_svd}')

    cond = np.linalg.cond(regular_matrix)
    print(f'condition number = {cond}')

    si = np.zeros((500, 501))
    for i in range(rank):
        si[i, i] = 1 / rank_svd[i]
    mp_svd = np.dot(np.dot(vh.T, si), u.T)
    print(f'Moore-Penrose (SVD method) = {mp_svd}')

    mp = np.linalg.pinv(regular_matrix)
    print(f'Moore-Penrose = {mp}')

    x = np.dot(mp, b)
    norm = np.linalg.norm(b - np.dot(regular_matrix, x))
    print(f'x = {x}')
    print(f'norm = {norm}')

    matrixT = regular_matrix.T
    matrixJ = np.dot((np.linalg.inv(np.dot(matrixT, regular_matrix))), matrixT)
    norm2 = np.linalg.norm(mp - matrixJ)
    print(f'norm2 = {norm2}')