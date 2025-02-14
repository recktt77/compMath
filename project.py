import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import eel

# Initialize Eel with folder 'web'
eel.init('web')

######################################################
#  INITIAL TASKS (1-4)
######################################################

@eel.expose
def plot_function(func_str, x_range):
    # Plots a given function in the specified x-range
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
    # Finds roots numerically using fsolve
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
    # Bisection method
    iterBi = 0
    b = b + 0.01  # small offset
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
    # Secant method
    x1 = x1 + 0.01
    iterSe = 0
    while abs(f(x1)) > tol:
        iterSe += 1
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
    return x1, iterSe

@eel.expose
def evaluate_methods(func_str, x_range):
    # Compares bisection and secant methods
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
    # Relaxation method (SOR) for solving linear systems
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

    # Store partial iteration results (step of 10)
    table_filtered = [table[i] for i in range(0, len(table), 10)]

    return {
        "solution": x.tolist(),
        "table": table_filtered,
        "error": None
    }

@eel.expose
def power_method(A, v0=None, tol=1e-6, max_iter=100):
    # Power method to find dominant eigenvalue and eigenvector
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
                # Plot eigenvalue convergence
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

        # If max_iter exceeded
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
#  NEW TASKS (5-8) FOR VARIANT 3
######################################################

# 5) Exponential Curve Fitting
@eel.expose
def exponential_curve_fitting(x_data_str, y_data_str):
    """
    Task 5: Exponential approximation y = a * e^(b*x).
    """
    try:
        x_values = list(map(float, x_data_str.split(',')))
        y_values = list(map(float, y_data_str.split(',')))
        if len(x_values) != len(y_values):
            return {"error": "Количество x и y не совпадает"}

        X = np.array(x_values, dtype=float)
        Y = np.array(y_values, dtype=float)

        # Check if Y > 0
        if np.any(Y <= 0):
            return {"error": "Все y должны быть > 0 для экспоненциальной аппроксимации"}

        lnY = np.log(Y)
        n = len(X)
        sum_x = np.sum(X)
        sum_lnY = np.sum(lnY)
        sum_x_lnY = np.sum(X * lnY)
        sum_x2 = np.sum(X**2)

        b = (n * sum_x_lnY - sum_x * sum_lnY) / (n * sum_x2 - sum_x**2)
        ln_a = (sum_lnY - b * sum_x) / n
        a = np.exp(ln_a)

        # Plot the fitted curve
        x_fit = np.linspace(min(X), max(X), 100)
        y_fit = a * np.exp(b * x_fit)

        plt.figure(figsize=(7, 5))
        plt.scatter(X, Y, color='blue', label='Data points')
        plt.plot(x_fit, y_fit, color='red', label=f"Fitted: y = {a:.3f} * e^({b:.3f}x)")
        plt.title("Exponential Curve Fitting")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

        plot_path = "web/exponential_fit.png"
        plt.savefig(plot_path)
        plt.close()

        return {
            "a": a,
            "b": b,
            "plot_path": plot_path,
            "error": None
        }

    except Exception as e:
        return {"error": str(e)}

# 6) Cubic Spline Interpolation
@eel.expose
def cubic_spline_interpolation(x_data_str, y_data_str, eval_points_str):
    """
    Task 6: Cubic spline interpolation.
    """
    try:
        x_vals = np.array(list(map(float, x_data_str.split(','))))
        y_vals = np.array(list(map(float, y_data_str.split(','))))

        if len(x_vals) != len(y_vals):
            return {"error": "Число точек x и y не совпадает"}

        # Build cubic spline
        cs = CubicSpline(x_vals, y_vals)

        # Evaluate at given points
        eval_points = np.array(list(map(float, eval_points_str.split(','))))
        spline_values = cs(eval_points)

        # Prepare a dense range for plotting
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

        spline_plot_path = "web/cubic_spline_plot.png"
        plt.savefig(spline_plot_path)
        plt.close()

        result_list = []
        for ep, val in zip(eval_points, spline_values):
            result_list.append({"x": float(ep), "y": float(val)})

        return {
            "values": result_list,
            "plot_path": spline_plot_path,
            "error": None
        }

    except Exception as e:
        return {"error": str(e)}

# 7) Picard’s Method
@eel.expose
def picard_method_picard_approx():
    """
    Task 7: Picard Method for y'(x)=x + y, y(0)=1.
    Compute up to 4th approximation and evaluate at x=0.2.
    """
    try:
        x = sp.Symbol('x', real=True)
        # Initial approx: y0(x)=1
        y_approx = [sp.Integer(1)]

        # Recurrence: y_{n+1}(x) = 1 + ∫[0->x] [t + y_n(t)] dt
        t = sp.Symbol('t', real=True)
        num_steps = 4
        for _ in range(num_steps):
            current = y_approx[-1]
            expr_to_int = t + current.subs(x, t)
            next_func = 1 + sp.integrate(expr_to_int, (t, 0, x))
            y_approx.append(sp.simplify(next_func))

        final_approx = y_approx[-1]
        val_02 = final_approx.subs(x, 0.2)
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

# 8) Simpson’s 1/3 Rule
@eel.expose
def simpson_one_third_rule(func_str, a_str, b_str, n_str):
    """
    Task 8: Simpson's 1/3 rule for numerical integration.
    """
    try:
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
# Start Eel application
######################################################
def start_app():
    eel.start('index.html', size=(800, 600))

if __name__ == "__main__":
    start_app()
