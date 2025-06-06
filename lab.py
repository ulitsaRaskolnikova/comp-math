from functools import reduce
from sys import stdin, stdout
import os
import math
from matplotlib import pyplot as plt
import numpy as np

generative_functions = [
    ["sin(x)", lambda x: math.sin(x)],
    ["e^x", lambda x: math.e ** x],
    ["3x^3 - 2x^2 + 4", lambda x: 3 * x ** 3 - 2 * x ** 2 + 4]
]


def generate_points(searched_x):
    while True:
        print("Выберите функцию")
        for i, _ in enumerate(generative_functions):
            print(f"{i + 1}. {_[0]}")
        inp = int(input("Ваш выбор: "))
        if 0 >= inp or inp > len(generative_functions):
            print("Нет такой опции")
            continue
        actual_func = generative_functions[inp - 1][1]
        a = int(input("Введите нижнию границу отрезка: "))
        b = int(input("Введите верхнюю границу отрезка: "))
        if b <= a:
            print("Верхняя граница должна быть больше нижней")
            continue
        n = int(input("Введите количество точек: "))
        print(f"f(x)={actual_func(searched_x)}")
        h = (b - a) / (n - 1)
        x, y = [], []
        for i in range(n):
            x.append(a + i * h)
            y.append(actual_func(a + i * h))
        return x, y


def print_deltas(deltas, output_file):
    for i in range(len(deltas)):
        for _ in deltas[i]:
            print(f"{_:.4f}", end="\t", file=output_file)
        print(f"Δy_{i}", file=output_file)


def lagrange_mnogochlen(xs, ys):
    n = len(xs)
    return lambda x: sum(
        [ys[i] * reduce(lambda a, b: a * b, [(x - xs[j]) / (xs[i] - xs[j]) if i != j else 1 for j in range(n)], 1) for i
         in range(n)])


def is_equidistant(xs):
    h = xs[1] - xs[0]
    for i in range(2, len(xs)):
        if xs[i] - xs[i - 1] != h:
            return False
    return True


def newton_divided_differences(xs, ys, output_file):
    n = len(xs)
    coeffs = [ys[0]] + [0] * (n - 1)
    table = [[0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]
    def find_func(a, b):
        poryadok = b - a
        if poryadok == 0:
            return table[a][0]
        first = find_func(a + 1, b)
        second = find_func(a, b - 1)
        res = (first - second) / (xs[b] - xs[a])
        table[a][poryadok] = res
        if a == 0:
            coeffs[poryadok] = res
        return res

    find_func(0, n - 1)


    print("Таблица разделённых разностей:", file = output_file)
    print("x_i\tf[x_i]\tf[x_i,x_j]\tf[x_i,x_j,x_k]\t...", file = output_file)
    for i in range(n):
        print(f"{xs[i]:.2f}", end="\t", file = output_file)
        for j in range(n - i):
            print(f"{table[i][j]:.8f}", end="\t", file = output_file)
        print(file = output_file)

    return lambda x: coeffs[0] + sum(
        coeffs[k] * reduce(lambda a, b: a * b, [x - xs[j] for j in range(k)], 1)
        for k in range(1, n)
    )



def newton_finite_differences(xs, ys, differences):
    def func(x):
        h = xs[1] - xs[0]
        mid = (xs[-1] + xs[0]) / 2
        if x <= mid:
            t = (x - xs[0]) / h
            return differences[0][0] + sum(
                [differences[i][0] * reduce(lambda a, b: a * b, [t - j for j in range(i)]) / math.factorial(i) for i in
                 range(1, len(differences))])
        else:
            t = (x - xs[-1]) / h
            return differences[0][-1] + sum(
                [differences[i][-1] * reduce(lambda a, b: a * b, [t + j for j in range(i)]) / math.factorial(i) for i in
                 range(1, len(differences))])

    return func

def run(x, y, searched_x, output_file):
    deltas = [y]
    n = len(x)

    func_lagrange = lagrange_mnogochlen(x, y)
    print(f"Интерполяция лагрнажа: {func_lagrange(searched_x)}", file = output_file)
    draw_plot(x, y, searched_x, func_lagrange, "Лагранж")

    func_newton_divided = newton_divided_differences(x, y, output_file)
    print(f"Интерполяция Ньютона (разд): {func_newton_divided(searched_x)}", file = output_file)
    draw_plot(x, y, searched_x, func_newton_divided, "Ньютон (разд)")
    if is_equidistant(x):
        print("Таблица конечных разностей:", file = output_file)
        for i in range(1, n):
            deltas.append([])
            for j in range(n - i):
                deltas[i].append(deltas[i - 1][j + 1] - deltas[i - 1][j])
        print_deltas(deltas, output_file)
        func_newton_finite = newton_finite_differences(x, y, deltas)
        method = ""
        if searched_x <= (x[0] + x[-1]) / 2:
            method = "1-ая формула"
            print(f"Интерполяция Ньютона (1-ая формула): {func_newton_finite(searched_x)}", file = output_file)
        else:
            method = "2-ая формула"
            print(f"Интерполяция Ньютона (2-ая формула): {func_newton_finite(searched_x)}", file = output_file)
        draw_plot(x, y, searched_x, func_newton_finite, f"Ньютон {method}")


def draw_plot(xs, ys, searched_x, func, name):
    plt.clf()
    func_x = np.linspace(xs[0], xs[-1], 400)
    func_y = []
    for x in func_x:
        func_y.append(func(x))
    plt.plot(func_x, func_y, 'r')
    plt.scatter(xs, ys, color='b')
    plt.scatter([searched_x], [func(searched_x)], color='g')
    plt.title(name)
    plt.grid(True)
    plt.show()


def main():
    while True:
        try:
            X = []
            Y = []
            searched_x = float(input("Введите X для которого будем искать значение: "))
            print("Как вы хотите задать числа?")
            print("1. С помощью ввода в консоль")
            print("2. Сгенерировать с помощью функции")
            inp = int(input("Ваш выбор: "))
            pairDots = []
            if inp == 1:
                inp_filename = input("Введите файл для считывания: ")
                if inp_filename == "":
                    print("Вводите точки x и y через пробел (f когда захотите завершить ввод).")
                    file = stdin
                else:
                    if not os.path.isfile(inp_filename):
                        raise FileNotFoundError(f"Файл {inp_filename} не найден.")
                    file = open(inp_filename, "r")
                for line in file:
                    if line.strip() == "":
                        continue
                    if "f" in line:
                        break
                    numbers = line.strip().split()
                    if len(numbers) != 2:
                        raise ValueError(
                            f"Неправильный ввод. Каждая строка должна содержать пару чисел (x, y).")
                    pairDots.append([float(numbers[0]), float(numbers[1])])
            elif inp == 2:
                X, Y = generate_points(searched_x)
            else:
                print("Нет такого варианта")
                continue

            pairDots.sort()
            for dots in pairDots:
                X.append(dots[0])
                Y.append(dots[1])
            out_filename = input("Введите файл для вывода: ")
            if out_filename == "":
                out = stdout
            else:
                out = open(out_filename, "w")

            run(X, Y, searched_x, out)
            out.close()
            break
        except FileNotFoundError as e:
            print("Файл не найден")
        except ValueError as e:
            print("Вводите числа")
        except Exception as e:
            print(f"Ошибка: {e}. Попробуйте снова.")


if __name__ == "__main__":
    main()