import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, E
import sympy as sp
import math
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import eel
import re

eel.init('web')

######################################################
#  ИСХОДНЫЕ ЗАДАНИЯ (1-4)
######################################################

@eel.expose
def plot_function(func_str, x_range):
    try:
        x_min, x_max = map(float, x_range.split(','))
    except ValueError:
        return "error: write correct range x (example, -4,4)"

    x = sp.symbols('x')
    try:
        func_expr = sp.sympify(func_str)
        func = sp.lambdify(x, func_expr, "numpy")
    except (sp.SympifyError, ValueError):
        return "error: incorrect function"

    x_vals = np.linspace(x_min, x_max, 500)
    try:
        y_vals = func(x_vals)
    except Exception as e:
        return f"error while completing: {e}"

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {func_str}")
    plt.axhline(0, color='red', linestyle='--', label="y = 0")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Graphical Method")
    plt.legend()
    plt.grid()
    plt.savefig("web/plot.png")
    plt.close()

    return "graph was made, and try to identify the roots."

@eel.expose
def find_roots(func_str, app_guess):
    x = sp.symbols('x')
    try:
        func_expr = sp.sympify(func_str)
        func = sp.lambdify(x, func_expr, "numpy")
        app_guess = list(map(float, app_guess.split(',')))
    except (sp.SympifyError, ValueError):
        return "error: incorrect function"

    def f_root(xx):
        return func(xx)

    roots = fsolve(f_root, np.array(app_guess))
    roots = np.unique(np.round(roots, 6))
    absolute_errors = [abs(r - g) for r, g in zip(roots, app_guess)]

    return f"found roots: {roots.tolist()}, apsolute error: {absolute_errors}"

def bisection_method(f, a, b, tol):
    iterBi = 0
    b = b + 0.01  # небольшая прибавка
    if f(a) * f(b) >= 0:
        return "Invalid initial values. f(a) and f(b) must be of different signs.", iterBi

    midpoint = (a + b) / 2
    while abs(f(midpoint)) > tol:
        iterBi += 1
        if f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        midpoint = (a + b) / 2

    return midpoint, iterBi

def secant_method(f, x0, x1, tol):
    x1 = x1 + 0.01
    iterSe = 0
    while abs(f(x1)) > tol:
        iterSe += 1
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
    return x1, iterSe

@eel.expose
def evaluate_methods(func_str, x_range):
    x = sp.symbols('x')
    try:
        func_expr = sp.sympify(func_str)
        f = sp.lambdify(x, func_expr, "numpy")
        x_min, x_max = map(float, x_range.split(','))
    except (sp.SympifyError, ValueError):
        return "error: incorrect function or range"

    rootBi, iterBi = bisection_method(f, x_min, x_max, 1e-6)
    rootSe, iterSe = secant_method(f, x_min, x_max, 1e-6)

    return f"Approx root bisection: {rootBi}, Iterations: {iterBi} | Approx root secant: {rootSe}, Iterations: {iterSe}"

@eel.expose
def relaxation_method(A, b, omega, tol=1e-6, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x0 = np.zeros(len(b))
    n = len(b)
    x = x0
    iteration_count = 0
    table = []

    for _ in range(int(max_iter)):
        x_new = np.copy(x)
        for i in range(n):
            if A[i][i] == 0:
                return "error: dioganal =0."
            summation = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - summation) / A[i][i]

        iteration_count += 1
        table.append(x_new.tolist())

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break

        x = x_new

    table_filtered = [table[i] for i in range(0, len(table), 10)]

    return {
        "solution": x.tolist(),
        "table": table_filtered,
        "error": None
    }

@eel.expose
def power_method(A, v0=None, tol=1e-6, max_iter=100):
    try:
        A = np.array(A, dtype=float)
        if A.shape[0] != A.shape[1]:
            return {"error": "matrix should be quadratic."}

        if v0 is None:
            v0 = np.ones(A.shape[0], dtype=float)
        else:
            v0 = np.array(v0, dtype=float)

        if v0.shape[0] != A.shape[0]:
            return {"error": "length of v0 should be same with A."}

        v = v0 / np.linalg.norm(v0)
        iteration_count = 0
        eigenvalue_history = []
        lambda_new = None

        for _ in range(max_iter):
            w = np.dot(A, v)
            lambda_new = np.dot(w, v)
            eigenvalue_history.append(float(lambda_new))

            if np.linalg.norm(w) == 0:
                return {"error": "null vector found check A"}

            v_new = w / np.linalg.norm(w)
            iteration_count += 1

            if np.linalg.norm(v_new - v) < tol:
                # Plot convergence
                plt.figure(figsize=(8, 6))
                plt.plot(range(1, len(eigenvalue_history) + 1), eigenvalue_history, 'b-')
                plt.xlabel('Iteration')
                plt.ylabel('Eigenvalue')
                plt.title('Eigenvalue Convergence')
                plt.grid(True)
                plt.savefig('web/convergence_plot.png')
                plt.close()

                return {
                    "eigenvalue": lambda_new,
                    "eigenvector": v_new.tolist(),
                    "iterations": iteration_count,
                    "error": None
                }

            v = v_new

        # Если превысили max_iter
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(eigenvalue_history) + 1), eigenvalue_history, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Convergence')
        plt.grid(True)
        plt.savefig('web/convergence_plot.png')
        plt.close()

        return {
            "eigenvalue": lambda_new,
            "eigenvector": v_new.tolist(),
            "iterations": iteration_count,
            "error": None
        }
    except Exception as e:
        return {"error": f"error: {str(e)}"}

######################################################
#  НОВЫЕ ЗАДАНИЯ (5-8) ДЛЯ ВАРИАНТА 3
######################################################

# 5) Exponential Curve Fitting
def parse_value(val_str):
    # Replace any case variation of "np.e" with sympy's Euler constant "E"
    replaced = re.sub(r'np\.e', 'E', val_str.strip(), flags=re.IGNORECASE)
    return float(sympify(replaced))

@eel.expose
def exponential_curve_fitting(x_data_str, y_data_str):
    """
    Exponential curve fitting using SciPy's curve_fit.
    Expects x and y values as comma-separated strings.
    Example:
        x: "0,1,2,3"
        y: "1,np.e,np.e**2,np.e**3"
    """
    try:
        # Parse input values using parse_value
        x_values = [parse_value(x) for x in x_data_str.split(',')]
        y_values = [parse_value(y) for y in y_data_str.split(',')]
        
        if len(x_values) != len(y_values):
            return {"error": "Количество значений x и y не совпадает"}
        if len(x_values) < 2:
            return {"error": "Need at least 2 data points"}
        if any(y <= 0 for y in y_values):
            return {"error": "All y values must be positive for exponential fitting"}
        
        # Convert lists to NumPy arrays
        x_arr = np.array(x_values, dtype=float)
        y_arr = np.array(y_values, dtype=float)
        
        # Define the exponential model: y = a * exp(b * x)
        def exponential_func(x, a, b):
            return a * np.exp(b * x)
        
        # Fit the model to the data
        params, _ = curve_fit(exponential_func, x_arr, y_arr)
        a, b = params
        
        # Generate a smooth curve for plotting
        x_fit = np.linspace(min(x_arr), max(x_arr), 100)
        y_fit = exponential_func(x_fit, a, b)
        
        # Plot the data and the fitted curve
        plt.figure(figsize=(7, 5))
        plt.scatter(x_arr, y_arr, color='orange', label='Data Points')
        plt.plot(x_fit, y_fit, label=f'Best Fit: y = {a:.2f}e^({b:.2f}x)', color='red')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.title('Curve Fitting: Exponential Fit')
        plt.grid(True)
        plt.savefig('web/exponential_fit.png')
        plt.close()
        
        return {
            "a": round(a, 3),
            "b": round(b, 3),
            "equation": f"y = {a:.3f} * e^({b:.3f}x)",
            "plot_path": "/exponential_fit.png",
            "error": None
        }
        
    except Exception as e:
        return {"error": str(e)}

# 6) Cubic Spline Interpolation
@eel.expose
def cubic_spline_interpolation(x_data_str, y_data_str, eval_points_str):
    """
    Задание 6: Кубическая сплайновая интерполяция.
    """
    try:
        # Add input validation
        if not all([x_data_str.strip(), y_data_str.strip(), eval_points_str.strip()]):
            return {"error": "Please fill in all fields"}
            
        x_vals = np.array(list(map(float, x_data_str.split(','))))
        if len(x_vals) < 2:
            return {"error": "Need at least 2 points for interpolation"}
            
        # Check for sorted x values
        if not all(x_vals[i] <= x_vals[i+1] for i in range(len(x_vals)-1)):
            return {"error": "X values must be in ascending order"}
            
        y_vals = np.array(list(map(float, y_data_str.split(','))))

        if len(x_vals) != len(y_vals):
            return {"error": "Число точек x и y не совпадает"}

        # Создаём кубический сплайн
        cs = CubicSpline(x_vals, y_vals)

        # Точки, в которых вычисляем сплайн
        eval_points = np.array(list(map(float, eval_points_str.split(','))))
        spline_values = cs(eval_points)

        # График
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        x_dense = np.linspace(x_min, x_max, 200)
        y_dense = cs(x_dense)

        plt.figure(figsize=(7, 5))
        plt.plot(x_vals, y_vals, 'o', label='Data Points')
        plt.plot(x_dense, y_dense, label='Cubic Spline')
        plt.plot(eval_points, spline_values, 'rx', label='Eval Points')
        plt.title("Cubic Spline Interpolation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

        plt.savefig('web/cubic_spline_plot.png')
        plt.close()

        result_list = []
        for ep, val in zip(eval_points, spline_values):
            result_list.append({"x": float(ep), "y": float(val)})

        return {
            "values": result_list,
            "plot_path": "/cubic_spline_plot.png",
            "error": None
        }

    except Exception as e:
        return {"error": str(e)}

# 7) Picard's Method (для y' = x + y, y(0)=1)
@eel.expose
def picard_method_picard_approx(x0):
    """
    Задание 7: Метод Пикара для y'(x)=x + y, y(0)=1,
    найти до 4-го приближения и y(0.2).
    """
    try:
        x = sp.Symbol('x', real=True)
        # Начальное приближение: y0(x)=1
        y_approx = [sp.Integer(1)]

        # Формула: y_{n+1}(x) = 1 + \int_0^x [t + y_n(t)] dt
        t = sp.Symbol('t', real=True)
        num_steps = 4
        for _ in range(num_steps):
            current = y_approx[-1]
            expr_to_int = t + current.subs(x, t)
            next_func = 1 + sp.integrate(expr_to_int, (t, 0, x))
            y_approx.append(sp.simplify(next_func))

        final_approx = y_approx[-1]
        # Найдём значение в x=0.2
        val_02 = final_approx.subs(x, x0)
        val_02_float = float(val_02.evalf())

        approx_strs = []
        for i, func in enumerate(y_approx):
            approx_strs.append(f"y{i}(x) = {sp.simplify(func)}")

        return {
            "approximations": approx_strs,
            "y4_at_0_2": val_02_float,
            "error": None
        }

    except Exception as e:
        return {"error": str(e)}

# 8) Simpson's 1/3 Rule
@eel.expose
def simpson_one_third_rule(func_str, a_str, b_str, n_str):
    """
    Задание 8: Метод Симпсона (1/3) для численного интегрирования.
    """
    try:
        # Add better error messages
        if not func_str.strip():
            return {"error": "Please enter a function"}
            
        if not all([a_str.strip(), b_str.strip(), n_str.strip()]):
            return {"error": "Please fill in all fields"}
            
        a = float(a_str)
        b = float(b_str)
        n = int(n_str)
        if n % 2 != 0:
            return {"error": "n должно быть чётным для 1/3 правила Симпсона"}

        x_var = sp.Symbol('x', real=True)
        try:
            func_expr = sp.sympify(func_str)
        except Exception:
            return {"error": "Неверная функция"}

        f = sp.lambdify(x_var, func_expr, 'numpy')

        h = (b - a)/n
        s = 0.0
        for i in range(n+1):
            xi = a + i*h
            if i == 0 or i == n:
                coef = 1
            elif i % 2 == 1:
                coef = 4
            else:
                coef = 2
            s += coef * f(xi)

        integral_approx = (h/3)*s
        return {
            "approx": integral_approx,
            "error": None
        }

    except Exception as e:
        return {"error": str(e)}


######################################################
# Запуск приложения Eel
######################################################
def start_app():
    eel.start('index.html', size=(800, 600))

if __name__ == "__main__":
    start_app()
