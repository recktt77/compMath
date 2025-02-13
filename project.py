import numpy as np
import matplotlib.pyplot as plt
import eel
import sympy as sp
import math
from scipy.optimize import fsolve

eel.init('web')

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

def find_roots(func_str, app_guess):
    x = sp.symbols('x')
    try:
        func_expr = sp.sympify(func_str)
        func = sp.lambdify(x, func_expr, "numpy")
        app_guess = list(map(float, app_guess.split(',')))
    except (sp.SympifyError, ValueError):
        return "error: incorrect function"
    
    def f_root(x):
        return func(x)
    
    roots = fsolve(f_root, np.array(app_guess)) 
    roots = np.unique(np.round(roots, 6)) 
    absolute_errors = [abs(root - guess) for root, guess in zip(roots, app_guess)]
    
    return f"found roots: {roots.tolist()}, apsolute error: {absolute_errors}"

def bisection_method(f, a, b, tol):
    iterBi = 0
    b=b+0.01
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
    x1=x1+0.01
    iterSe = 0
    while abs(f(x1)) > tol:
        iterSe += 1
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
    return x1, iterSe

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
    
    return f"Approximate root bisection: {rootBi}, Iterations: {iterBi}, Approximate root secant: {rootSe}, Iterations: {iterSe}"

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

def power_method(A, v0=np.array([1, 1, 1], dtype=float), tol=1e-6, max_iter=100):
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

        for _ in range(max_iter):
            w = np.dot(A, v)
            lambda_new = np.dot(w, v)
            eigenvalue_history.append(float(lambda_new))
            
            if np.linalg.norm(w) == 0:
                return {"error": "null vector found check A"}

            v_new = w / np.linalg.norm(w)
            iteration_count += 1

            if np.linalg.norm(v_new - v) < tol:
                # Plot convergence using matplotlib
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

        # Plot convergence if max iterations reached
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


eel.expose(plot_function)
eel.expose(find_roots)
eel.expose(evaluate_methods)
eel.expose(relaxation_method)
eel.expose(power_method)

def start_app():
    eel.start('index.html', size=(800, 600))

if __name__ == "__main__":
    start_app()
