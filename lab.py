import math
from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate

funcs = [
    ["y'= y/3 + 2x", lambda x, y: y / 3 + 2 * x,
     lambda x, x0, y0: ((y0 + 6 * x0 + 18) / math.exp(x0 / 3)) * math.exp(x / 3) - 6 * x - 18],
    ["y'= x + y", lambda x, y: x + y, lambda x, x0, y0: ((y0 + x0 + 1) / math.exp(x0)) * math.exp(x) - x - 1],
    ["y'= 2y + cos(x)", lambda x, y: 2 * y + math.cos(x),
     lambda x, x0, y0: ((y0 + 2 * math.cos(x0) / 5 - math.sin(x0) / 5) / (math.exp(2 * x0))) * math.exp(
         2 * x) + math.sin(x) / 5 - 2 * math.cos(x) / 5],
]


def modified_euler_method(func, x0, xn, y0, h):
    xs = [x0]
    ys = [y0]
    x = x0
    while x < xn and not math.isclose(x, xn):
        new_y = ys[-1] + h / 2 * (func(x, ys[-1]) + func(x + h, ys[-1] + h * func(x, ys[-1])))
        ys.append(new_y)
        x += h
        xs.append(x)
    return xs, ys

def runge_kutta_4(func, x0, xn, y0, h):
    xs = [x0]
    ys = [y0]
    x = x0
    while x < xn and not math.isclose(x, xn):
        k1 = h * func(x, ys[-1])
        k2 = h * func(x + h / 2, ys[-1] + k1 / 2)
        k3 = h * func(x + h / 2, ys[-1] + k2 / 2)
        k4 = h * func(x + h, ys[-1] + k3)

        ys.append(ys[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        x += h
        xs.append(x)
    return xs, ys


def adams_method(func, x0, xn, y0, h, tol=1e-6):
    xs_rk, ys_rk = runge_kutta_4(func, x0, x0 + 3 * h, y0, h)
    xs = xs_rk[:4]
    ys = ys_rk[:4]

    x = x0 + 3 * h
    while x < xn and not math.isclose(x, xn):
        # Предиктор
        f_im3 = func(xs[-3], ys[-3])
        f_im2 = func(xs[-2], ys[-2])
        f_im1 = func(xs[-1], ys[-1])
        y_pred = ys[-1] + h / 24 * (55 * f_im1 - 59 * f_im2 + 37 * f_im3 - 9 * func(xs[-4], ys[-4]))

        # Корректор
        y_corr = y_pred
        while True:
            f_i = func(x + h, y_corr)
            y_new = ys[-1] + h / 24 * (9 * f_i + 19 * f_im1 - 5 * f_im2 + f_im3)

            if abs(y_new - y_corr) < tol:
                break
            y_corr = y_new

        ys.append(y_new)
        x += h
        xs.append(x)

    return xs, ys



def draw_plot(xs, ys, func, name, func_name):
    plt.clf()
    func_x = np.linspace(xs[0], xs[-1], 400)
    func_y = []
    for x in func_x:
        func_y.append(func(x, xs[0], ys[0]))
    plt.plot(func_x, func_y, 'r', label=func_name)
    plt.scatter(xs, ys, color='b')
    plt.title(name)
    plt.grid(True)
    plt.show()


methods = [
    ["Метод Рунге-Кутты 4-го порядка", runge_kutta_4, 4, 1],
    ["Модифицированный метод Эйлера", modified_euler_method, 2, 2],
    ["Метод Адамса", adams_method, None, 5]
]


def run(uravn, x0, xn, y0, normal_h, eps):
    func_name = uravn[0]
    func = uravn[1]
    original_func = uravn[2]
    for method in methods:
        method_name = method[0]
        method_func = method[1]
        method_accuracy = method[2]
        method_min_points = method[3]
        print(method_name)
        h = normal_h
        if (xn - x0) / h + 1 < method_min_points:
            print("Слишком большой шаг для такого отрезка")
            print("Минимальное число точек на отрезке для данного метода - ", method_min_points)
            h = (xn - x0) / (method_min_points - 1)
            print("Автоматически назначенный начальный шаг: ", h)
        if method_accuracy is not None:
            while True:
                xs_h, ys_h = method_func(func, x0, xn, y0, h)
                xs_h05, ys_h05 = method_func(func, x0, xn, y0, h / 2)
                inaccuracy = abs(ys_h[-1] - ys_h05[-1]) / (2 ** method_accuracy - 1)

                n = len(xs_h05)
                normal_n = (xn - x0) / (h) + 1
                delta = round((n - 1) / (normal_n - 1))
                print(h, ys_h[len(ys_h) - 1])

                if inaccuracy < eps:
                    n = len(xs_h05)
                    print("Достаточная точность получилась при шаге: ", h / 2)
                    print("Число точек: ", n)
                    print("Погрешность: ", inaccuracy)

                    orig = [original_func(xs_h05[i], xs_h05[0], ys_h05[0]) for i in range(n)]
                    normal_n = (xn - x0) / (h / 2) + 1
                    delta = round((n - 1) / (normal_n - 1))
                    print(h / 2, ys_h05[len(ys_h05) - 1])
                    output_x = []
                    output_y = []
                    output_orig = []
                    for i in range(round(normal_n)):
                        output_x.append(xs_h05[i * delta])
                        output_y.append(ys_h05[i * delta])
                        output_orig.append(orig[i * delta])
                    print(tabulate(
                        list(zip(output_x, output_y, output_orig)), headers=["x", "y", "y_toch"]))
                    draw_plot(xs_h05, ys_h05, original_func, method_name, func_name)

                    break
                h /= 2
                if h < 1e-6:
                    print("Достаточно.")
                    break
        else:
            previous_last_elements = []
            while True:
                xs_h, ys_h = method_func(func, x0, xn, y0, h, tol=eps)
                inaccuracy = max([abs(original_func(xs_h[i], x0, y0) - ys_h[i]) for i in range(len(xs_h))])
                previous_last_elements.append(ys_h[-1])
                if inaccuracy <= eps:
                    n = len(xs_h)
                    print("Достаточная точность получилась при шаге: ", h)
                    print("Число точек: ", n)
                    print("Погрешность: ", inaccuracy)

                    orig = [original_func(xs_h[i], xs_h[0], ys_h[0]) for i in range(n)]
                    normal_n = (xn - x0) / normal_h + 1
                    delta = round((n - 1) / (normal_n - 1))
                    output_x = []
                    output_y = []
                    output_orig = []
                    output_eps = []
                    for i in range(round(normal_n)):
                        output_x.append(xs_h[i * delta])
                        output_y.append(ys_h[i * delta])
                        output_orig.append(orig[i * delta])
                        output_eps.append(abs(output_orig[-1] - output_y[-1]))
                    print(tabulate(
                        list(zip(output_x, output_y, output_orig, output_eps)), headers=["x", "y", "y_toch", "|eps|"]))
                    draw_plot(xs_h, ys_h, original_func, method_name, func_name)
                    break
                h /= 2
                if h < 1e-6:
                    print("")
                    break
            print(previous_last_elements)
        print("---------------------------------------------------------------------")


def main():
    print("Выберете уравнение:")
    for i in range(len(funcs)):
        print(i + 1, ". ", funcs[i][0])
    while True:
        try:
            uravn = funcs[int(input("Ваш выбор: ")) - 1]
            break
        except Exception:
            print("Невалидный выбор!")
    x0 = float(input("Введите нижнюю границу отрезка: "))
    xn = float(input("Введите верхнюю границу отрезка: "))
    y0 = float(input("Введите y0 = y(x0): "))
    h = float(input("Введите начальный шаг: "))
    eps = float(input("Введите точность: "))
    run(uravn, x0, xn, y0, h, eps)


if __name__ == "__main__":
    main()