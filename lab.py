import math


def f1(x):
    """Линейная функция"""
    return 2 * x + 1


def f1_integral(x):
    return (x ** 2) + x


def f2(x):
    """Квадратичная функция"""
    return x ** 2 + 3 * x - 2


def f2_integral(x):
    return (x ** 3) / 3 + 3 * (x ** 2) / 2 - 2*x


def f3(x):
    """Тригонометрическая функция"""
    return math.sin(x) + 0.5


def f3_integral(x):
    return -math.cos(x) + 0.5 * x


def res(f, a, b):
    return f(b) - f(a)


def rectangle_left(f, a, b, n):
    """Метод левых прямоугольников"""
    h = (b - a) / n
    return h * sum(f(a + i * h) for i in range(n))


def rectangle_right(f, a, b, n):
    """Метод правых прямоугольников"""
    h = (b - a) / n
    return h * sum(f(a + (i + 1) * h) for i in range(n))


def rectangle_mid(f, a, b, n):
    """Метод средних прямоугольников"""
    h = (b - a) / n
    return h * sum(f(a + (i + 0.5) * h) for i in range(n))


def trapezoidal(f, a, b, n):
    """Метод трапеций"""
    h = (b - a) / n
    return h * ((f(a) + f(b)) / 2 + sum(f(a + i * h) for i in range(1, n)))


def simpson(f, a, b, n):
    """Метод Симпсона"""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    sum_odd = sum(f(a + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
    sum_even = sum(f(a + 2 * i * h) for i in range(1, n // 2))
    return h / 3 * (f(a) + f(b) + 4 * sum_odd + 2 * sum_even)


def runge_rule(I_h, I_h2, k):
    """Оценка погрешности по правилу Рунге"""
    return abs(I_h - I_h2) / (2 ** k - 1)


def integrate(f, a, b, eps=1e-6, method='simpson'):
    """Интегрирование с заданной точностью"""
    methods = {
        'rect_left': rectangle_left,
        'rect_right': rectangle_right,
        'rect_mid': rectangle_mid,
        'trapezoidal': trapezoidal,
        'simpson': simpson
    }

    if method not in methods:
        raise ValueError("Неизвестный метод интегрирования")

    n = 4
    k = 2
    if method == 'rect_left' or method == 'rect_right':
        k = 1
    elif (method == 'simpson'):
        k = 4

    while True:
        I_h = methods[method](f, a, b, n)
        I_h2 = methods[method](f, a, b, 2 * n)
        print(f"n={n} I_h={I_h:.8f} I_h2={I_h2:.8f} R = {runge_rule(I_h, I_h2, k):.8f}")
        error = runge_rule(I_h, I_h2, k)

        if error < eps:
            break
        n *= 2

    return I_h2, n, error


def main():
    print("Доступные функции:")
    print("1. 2x + 1")
    print("2. x^2 + 3x - 2")
    print("3. sin(x) + 0.5")

    choice = int(input("Выберите функцию (1-3): "))
    a = float(input("Введите нижний предел интегрирования: "))
    b = float(input("Введите верхний предел интегрирования: "))
    eps = float(input("Введите точность вычисления (например, 0.0001): "))

    funcs = [f1, f2, f3]
    f = funcs[choice - 1]
    funcs_integral = [f1_integral,f2_integral,f3_integral]
    print("\nДоступные методы:")
    methods = {
        '1': 'rect_left',
        '2': 'rect_right',
        '3': 'rect_mid',
        '4': 'trapezoidal',
        '5': 'simpson'
    }
    for num, name in methods.items():
        print(f"{num}. {name}")

    method_choice = input("Выберите метод (1-5): ")
    method = methods[method_choice]

    result, n, error = integrate(f, a, b, eps, method)

    print("\nРезультат:")
    print(f"Значение интеграла: {result:.8f}")
    print(f"Точное значение: {res(funcs_integral[choice-1],a,b)}")
    print(f"Число разбиений: {n}")
    print(f"Оценка погрешности: {error:.2e}")

if __name__ == "__main__":
    main()
