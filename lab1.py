import numpy as np
import sys

INVALID_INPUT = 'Invalid input. Please try again.'
INVALID_FILE_DATA = 'Invalid file data.'
LEFT_BOUND = 0
RIGHT_BOUND = 100

def get_file_by_user():
    while True:
        try:
            filename = input('Enter the filename or press enter to use the keyboard: ')
            if filename == '': return None
            file = open(filename, 'r')
            return file
        except: print(INVALID_INPUT)

def get_matrix_size_by_user():
    while True:
        try: 
            n = int(input('Enter the matrix size: '))
            return n
        except: print(INVALID_INPUT)

def is_random_generated():
    while True:
        try:
            is_random = input('Do you want to generate a matrix with random coefficients? [y/n]: ')
            if (is_random == 'y' or is_random == 'n'):
                return True if is_random == 'y' else False
        except: print(INVALID_INPUT)

def get_matrix_by_user(n):
    print('Enter the matrix: ')
    i = 0
    arr = []
    while i < n:
        try:
            line = list(map(float, input().split()))
            if len(line) == n:
                i += 1
                arr.append(line)
            else:
                print("Invalid length of line. Please try again.")
        except: print(INVALID_INPUT)
    return np.array(arr)

def get_values_by_user(n):
    while True:
        try:
            print('Enter the values: ')
            values = np.array(list(map(float, input().split())))
            if len(values) != n:
                print("Invalid length of values. Please try again.")
                continue
            return values
        except: print(INVALID_INPUT)

def get_accuracy_by_user():
    while True:
        try:
            accuracy = float(input('Enter the accuracy: '))
            return accuracy
        except: print(INVALID_INPUT)

def read_user_input():
    n = get_matrix_size_by_user()
    is_random = is_random_generated()
    matrix = None
    values = None
    if is_random:
        matrix = np.random.rand(n, n) * 100
        values = np.random.rand(n) * 100
        for i in range(n):
            matrix[i][i] = sum(matrix[i]) + 1
    else:
        matrix = get_matrix_by_user(n)
        values = get_values_by_user(n)
    accuracy = get_accuracy_by_user()
    return matrix, values, accuracy

def read_file_input(file):
    try:
        n = int(file.readline())
        matrix = np.array([list(map(float, file.readline().split())) for _ in range(n)])
        values = np.array(list(map(float, file.readline().split())))
        accuracy = float(file.readline())
        return matrix, values, accuracy
    except: 
        print(INVALID_FILE_DATA)
        return None, None, None

def read_data(file):
    if file is None: return read_user_input()
    return read_file_input(file)

def check_diagonal_dominance(matrix):
    n = len(matrix)
    for i in range(n):
        row_sum = sum(abs(matrix[i, j]) for j in range(n)) - abs(matrix[i, i])
        if abs(matrix[i, i]) < row_sum:
            return False
    return True

def swap_rows(matrix, b, i, j):
    matrix[[i, j]] = matrix[[j, i]]
    b[i], b[j] = b[j], b[i]

def make_diagonal_dominant(matrix, values):
    n = len(matrix)
    for i in range(n):
        if abs(matrix[i, i]) < sum(abs(matrix[i, j]) for j in range(n)) - abs(matrix[i, i]):
            for j in range(i + 1, n):
                if abs(matrix[j, i]) > abs(matrix[i, i]):
                    swap_rows(matrix, values, i, j)
                    break
    if not check_diagonal_dominance(matrix):
        print('Matrix cannot be made diagonally dominant.')
        sys.exit()

def gauss_seidel(matrix, b, accuracy):
    n = len(matrix)
    x = np.zeros_like(b)
    iterations = 0
    while True:
        x_new = np.copy(x)
        for i in range(n):
            sigma = sum(matrix[i, j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / matrix[i, i]
        errors = abs(x_new - x)
        iterations += 1
        if max(errors) < accuracy:
            return x_new, iterations, errors
        x = x_new

def matrix_norm(matrix):
    return np.linalg.norm(matrix, ord=2)

file = get_file_by_user()
matrix, values, accuracy = read_data(file)
if matrix is None:
    sys.exit()

print(f"Initial Matrix:\n{matrix}")
print(f"Values: {values}")
print(f"Accuracy: {accuracy}")

make_diagonal_dominant(matrix, values)
print(f"Diagonal Dominant Matrix:\n{matrix}")

solution, iterations, errors = gauss_seidel(matrix, values, accuracy)
print(f"Solution: {solution}")
print(f"Number of iterations: {iterations}")
print(f"Errors: {errors}")
norm = matrix_norm(matrix)
print(f"Matrix norm: {norm}")
