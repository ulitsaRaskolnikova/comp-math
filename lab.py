import math
import os
from sys import stdin, stdout

import numpy as np
import matplotlib.pyplot as plt


def linear_approximation(X, Y):
    sx = sum(X)
    sxx = sum(x * x for x in X)
    sy = sum(Y)
    sxy = sum(x * y for x, y in zip(X, Y))
    A = np.matrix([[len(X), sx], [sx, sxx]])
    B = np.matrix([[sy], [sxy]])
    res = np.linalg.solve(A, B)
    return res[0, 0], res[1, 0]


def square_approximation(X, Y):
    sx = sum(X)
    sxx = sum(x ** 2 for x in X)
    sxxx = sum(x ** 3 for x in X)
    sxxxx = sum(x ** 4 for x in X)
    sy = sum(Y)
    sxy = sum(x * y for x, y in zip(X, Y))
    sxxy = sum(x ** 2 * y for x, y in zip(X, Y))
    A = np.matrix([[len(X), sx, sxx], [sx, sxx, sxxx], [sxx, sxxx, sxxxx]])
    B = np.matrix([[sy], [sxy], [sxxy]])
    res = np.linalg.solve(A, B)
    return res[0, 0], res[1, 0], res[2, 0]


def cube_approximation(X, Y):
    sx = sum(X)
    sxx = sum(x ** 2 for x in X)
    sxxx = sum(x ** 3 for x in X)
    sxxxx = sum(x ** 4 for x in X)
    sxxxxx = sum(x ** 5 for x in X)
    sxxxxxx = sum(x ** 6 for x in X)
    sy = sum(Y)
    sxy = sum(x * y for x, y in zip(X, Y))
    sxxy = sum(x ** 2 * y for x, y in zip(X, Y))
    sxxxy = sum(x ** 3 * y for x, y in zip(X, Y))
    A = np.matrix([
        [len(X), sx, sxx, sxxx],
        [sx, sxx, sxxx, sxxxx],
        [sxx, sxxx, sxxxx, sxxxxx],
        [sxxx, sxxxx, sxxxxx, sxxxxxx]
    ])
    B = np.matrix([[sy], [sxy], [sxxy], [sxxxy]])
    res = np.linalg.solve(A, B)
    return res[0, 0], res[1, 0], res[2, 0], res[3, 0]


def exponential_approximation(X, Y):
    if any(y <= 0 for y in Y):
        raise ValueError("Для показательной аппроксимации значения Y должны быть положительными.")
    log_Y = [math.log(y) for y in Y]
    A, b = linear_approximation(X, log_Y)
    return math.exp(A), b


def logarithmic_approximation(X, Y):
    if any(x <= 0 for x in X):
        raise ValueError("Для логарифмической аппроксимации значения X должны быть положительными.")
    log_X = [math.log(x) for x in X]
    return linear_approximation(log_X, Y)


def power_approximation(X, Y):
    if any(x <= 0 for x in X) or any(y <= 0 for y in Y):
        raise ValueError("Для степенной аппроксимации X и Y должны быть положительными.")
    log_X = [math.log(x) for x in X]
    log_Y = [math.log(y) for y in Y]
    A, b = linear_approximation(log_X, log_Y)
    return math.exp(A), b


def get_linear_approximation(X, Y):
    a, b = linear_approximation(X, Y)
    return lambda x: a + b * x


def get_square_approximation(X, Y):
    a, b, c = square_approximation(X, Y)
    return lambda x: a + b * x + c * x ** 2


def get_cube_approximation(X, Y):
    a, b, c, d = cube_approximation(X, Y)
    return lambda x: a + b * x + c * x ** 2 + d * x ** 3


def get_exponential_approximation(X, Y):
    a, b = exponential_approximation(X, Y)
    return lambda x: a * math.exp(b * x)


def get_logarithmic_approximation(X, Y):
    a, b = logarithmic_approximation(X, Y)
    return lambda x: a * math.log(x) + b


def get_power_approximation(X, Y):
    a, b = power_approximation(X, Y)
    return lambda x: a * x ** b


def calculate_correlation(x_values, y_values):
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator_x = sum((x - mean_x) ** 2 for x in x_values)
    denominator_y = sum((y - mean_y) ** 2 for y in y_values)
    return numerator / math.sqrt(denominator_x * denominator_y)


def calculate_determination(y_true, y_pred):
    mean_pred = sum(y_pred) / len(y_pred)
    ss_res = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    ss_tot = sum((y_t - mean_pred) ** 2 for y_t in y_true)
    return 1 - ss_res / ss_tot


def print_approximation_quality(R, output_file):
    if R >= 0.95:
        print("Высокая аппроксимация", file=output_file)
    elif R >= 0.75:
        print("Удовлетворительная аппроксимация", file=output_file)
    elif R >= 0.5:
        print("Слабая аппроксимация", file=output_file)
    else:
        print("Недостаточная аппроксимация", file=output_file)


def calculate_standard_deviation(y_pred, y_true):
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_pred))


APPROXIMATION_METHODS = [
    {
        'name': "линейная",
        'function': linear_approximation,
        'getter': get_linear_approximation,
        'formula': "{} + ({})x"
    },
    {
        'name': "квадратичная",
        'function': square_approximation,
        'getter': get_square_approximation,
        'formula': "{} + ({})x + ({})x^2"
    },
    {
        'name': "кубическая",
        'function': cube_approximation,
        'getter': get_cube_approximation,
        'formula': "{} + ({})x + ({})x^2 + ({})x^3"
    },
    {
        'name': "показательная",
        'function': exponential_approximation,
        'getter': get_exponential_approximation,
        'formula': "{}e^({})x"
    },
    {
        'name': "степенная",
        'function': power_approximation,
        'getter': get_power_approximation,
        'formula': "{}x^{}"
    },
    {
        'name': "логарифмическая",
        'function': logarithmic_approximation,
        'getter': get_logarithmic_approximation,
        'formula': "{}ln(x) + ({})"
    }
]


def run_approximations(x_values, y_values, output_file):
    min_x, max_x = min(x_values), max(x_values)
    best_sigma = math.inf
    best_method = ""

    plt.scatter(x_values, y_values, label="Исходные точки")

    for method in APPROXIMATION_METHODS:
        try:
            coefficients = method['function'](x_values, y_values)
            approx_func = method['getter'](x_values, y_values)

            predicted = [approx_func(x) for x in x_values]
            errors = [p - y for p, y in zip(predicted, y_values)]

            sigma = calculate_standard_deviation(predicted, y_values)
            R2 = calculate_determination(y_values, predicted)

            print(f"\n{method['name']} аппроксимация:", file=output_file)
            print(f"Формула: phi(x) = {method['formula'].format(*coefficients)}", file=output_file)

            print_table(output_file, "     X:     ", x_values, ".3f")
            print_table(output_file, "     Y:     ", y_values, ".3f")
            print_table(output_file, "    φ(X):   ", predicted, ".3f")
            print_table(output_file, "(φ(X) - y)²:", errors, ".3f")

            if method['name'] == "линейная":
                corr = calculate_correlation(x_values, y_values)
                print(f"Коэффициент корреляции: {corr:.3f}", file=output_file)

            print(f"Коэффициент детерминации R²: {R2:.3f}", file=output_file)
            print_approximation_quality(R2, output_file)
            print(f"Среднеквадратичное отклонение: {sigma:.3f}", file=output_file)

            if sigma < best_sigma:
                best_sigma = sigma
                best_method = method['name']

            x_plot = np.linspace(min_x, max_x, 400)
            y_plot = np.ravel([approx_func(x) for x in x_plot])
            plt.plot(x_plot, y_plot, label=method['name'])

        except Exception as e:
            print(f"Ошибка в {method['name']} аппроксимации: {str(e)}", file=output_file)

    print(f"\nЛучший метод: {best_method}", file=output_file)
    print(f"Минимальное стандартное отклонение: {best_sigma:.3f}", file=output_file)

    plt.legend()
    plt.grid(True)
    plt.show()


def print_table(file, header, values, format_spec):
    print(header, end="\t", file=file)
    for val in values:
        print(f"{val:{format_spec}}", end="\t", file=file)
    print(file=file)


def main():
    while True:
        try:
            x_data = []
            y_data = []

            input_source = input("Введите файл для считывания: ")

            if not input_source:
                print("Вводите пары чисел x y, каждую пару с новой строчки:")
                input_file = stdin
            else:
                if not os.path.isfile(input_source):
                    raise FileNotFoundError(f"Файл {input_source} не найден.")
                input_file = open(input_source, "r")

            for line in input_file:
                if len(x_data) >= 9:
                    break
                if line.strip() == "":
                    continue

                try:
                    x, y = map(float, line.strip().split())
                    x_data.append(x)
                    y_data.append(y)
                except ValueError:
                    raise ValueError(f"Ошибка в строке '{line.strip()}'. Ожидались два числа.")

            output_dest = input("Введите файл для вывода: ")
            output_file = stdout if not output_dest else open(output_dest, "w")

            run_approximations(x_data, y_data, output_file)

            if input_source:
                input_file.close()
            if output_dest:
                output_file.close()

            break

        except FileNotFoundError as e:
            print(f"Ошибка: {str(e)}")
        except ValueError as e:
            print(f"Ошибка данных: {str(e)}")
        except Exception as e:
            print(f"Неожиданная ошибка: {str(e)}")
        finally:
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
