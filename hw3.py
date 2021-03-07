import numpy as np


def get_tuple(line_str):
    info = line_str[:-1].split(', ')
    return float(info[0]), int(info[1]), int(info[2])


def get_tuple_transpose(line_str):
    info = line_str[:-1].split(', ')
    return float(info[0]), int(info[2]), int(info[1])


def get_matrix(file):
    matrix = None
    with open(file, 'r') as f:
        n = int(f.readline())
        matrix = [[] for _ in range(n)]
        for line in f:
            nr, line2, column = get_tuple(line)
            matrix[line2].append([nr, column])

    return matrix


def get_matrix_transpose(file):
    matrix = None
    with open(file, 'r') as f:
        n = int(f.readline())
        matrix = [[] for _ in range(n)]
        for line in f:
            nr, line2, column = get_tuple_transpose(line)
            matrix[line2].append([nr, column])

    return matrix


A = get_matrix("a.txt")
B = get_matrix("b.txt")
B_transpose = get_matrix_transpose("b.txt")

def A_plus_B(matrix1, matrix2):
    n = max(len(matrix2), len(matrix1))
    concatenated_lists = [[] for _ in range(n)]
    addition_matrix = [[] for _ in range(n)]
    # concatenate matrix1 and matrix2
    for line in range(n):
        concatenated_lists[line] += matrix1[line] + matrix2[line]
    # sort the concatenated lists by column
    for line in concatenated_lists:
        line.sort(key=lambda x: x[1])
    # check if a column appears in the list twice and add the two values
    for line in concatenated_lists:
        for i in range(len(line)):
            if line[i][1] == line[i + 1][1]:
                sum = line[i][0] + line[i + 1][0]
                line.append([sum, line[i + 1][1]])
                line[i][0] = 0
                line[i + 1][0] = 0
    for line in concatenated_lists:
        for elem in line:
            if elem[0] == 0:
                line.remove(elem)
                line.remove(elem)

    return concatenated_lists


def multiply(line, column):
    sum = 0
    for i in range(len(line)):
        for j in range(len(column)):
            if line[i][1] == column[j][1]:
                sum += line[i][0] * column[j][0]
    return sum if sum != 0 else None


def A_mul_B(matrix1, matrix2):
    result = [[] for _ in range(len(matrix2))]
    for line1 in range(len(matrix1)):
        for line2 in range(len(matrix2)):
            element = multiply(matrix1[line1], matrix2[line2])
            if element is not None:
                result[line1].append([element, line1, line2])

    return result


# print(A)
# print(B)
# print(A_plus_B(A, B))
# print(A_mul_B(A, B_transpose))
