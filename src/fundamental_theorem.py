"""
Fundamental Theorem of Calculus Demonstrations
=============================================

This module provides interactive visualizations and demonstrations of the
Fundamental Theorem of Calculus, connecting derivatives and integrals.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Callable, List, Tuple, Union
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy import integrate


class FundamentalTheorem:
    """Class for demonstrating the Fundamental Theorem of Calculus."""

    def __init__(self):
        self.x = sp.Symbol('x')
        self.t = sp.Symbol('t')

    def riemann_sum_visualization(self, func_str: str = "x**2", interval: Tuple[float, float] = (0, 2),
                                n_rectangles: int = 10) -> None:
        """
        Visualize Riemann sums approximating definite integrals.

        Args:
            func_str: Function to integrate as string
            interval: Integration interval (a, b)
            n_rectangles: Number of rectangles for approximation
        """
        func = sp.sympify(func_str)
        func_numpy = sp.lambdify(self.x, func, 'numpy')

        a, b = interval
        x_vals = np.linspace(a, b, 1000)
        y_vals = func_numpy(x_vals)

        # Calculate Riemann sum (right endpoint rule)
        dx = (b - a) / n_rectangles
        x_rects = np.linspace(a, b - dx, n_rectangles)
        y_rects = func_numpy(x_rects + dx)  # Right endpoints

        riemann_sum = np.sum(y_rects * dx)

        # Calculate exact integral
        try:
            exact_integral = float(sp.integrate(func, (self.x, a, b)))
        except:
            exact_integral = None

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Function with Riemann rectangles
        ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')

        for i, (x_rect, y_rect) in enumerate(zip(x_rects, y_rects)):
            rect = patches.Rectangle((x_rect, 0), dx, y_rect,
                                   linewidth=1, edgecolor='red', facecolor='lightblue', alpha=0.6)
            ax1.add_patch(rect)

        ax1.fill_between(x_vals, y_vals, alpha=0.3, color='green', label='Exact area')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'Riemann Sum (n={n_rectangles})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Convergence of Riemann sums
        n_values = range(1, 101)
        riemann_values = []

        for n in n_values:
            dx_temp = (b - a) / n
            x_temp = np.linspace(a, b - dx_temp, n)
            y_temp = func_numpy(x_temp + dx_temp)
            riemann_values.append(np.sum(y_temp * dx_temp))

        ax2.plot(n_values, riemann_values, 'ro-', markersize=3, label='Riemann sums')
        if exact_integral is not None:
            ax2.axhline(y=exact_integral, color='green', linestyle='--',
                       label=f'Exact = {exact_integral:.6f}')
        ax2.set_xlabel('Number of rectangles')
        ax2.set_ylabel('Approximate integral')
        ax2.set_title('Convergence to Exact Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"Function: f(x) = {func}")
        print(f"Interval: [{a}, {b}]")
        print(f"Riemann sum (n={n_rectangles}): {riemann_sum:.6f}")
        if exact_integral is not None:
            print(f"Exact integral: {exact_integral:.6f}")
            print(f"Error: {abs(riemann_sum - exact_integral):.6f}")

    def ftc_part1_demo(self, func_str: str = "2*x", lower_bound: float = 0) -> None:
        """
        Demonstrate FTC Part 1: If F(x) = ∫[a to x] f(t) dt, then F'(x) = f(x)

        Args:
            func_str: Integrand function as string
            lower_bound: Lower bound of integration
        """
        func = sp.sympify(func_str)

        # Define the accumulator function F(x) = ∫[a to x] f(t) dt
        # We'll calculate this numerically at various points

        x_vals = np.linspace(lower_bound, lower_bound + 5, 100)
        F_vals = []
        F_prime_vals = []

        func_numpy = sp.lambdify(self.t, func, 'numpy')

        # Calculate F(x) and F'(x) numerically
        for x_val in x_vals:
            # F(x) = integral from lower_bound to x_val
            if x_val == lower_bound:
                F_val = 0
            else:
                F_val, _ = integrate.quad(func_numpy, lower_bound, x_val)
            F_vals.append(F_val)

            # F'(x) should equal f(x) by FTC Part 1
            F_prime_vals.append(func_numpy(x_val))

        # Calculate numerical derivative of F(x) for comparison
        F_vals = np.array(F_vals)
        dx = x_vals[1] - x_vals[0]
        F_prime_numerical = np.gradient(F_vals, dx)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Plot original function f(t)
        t_vals = np.linspace(lower_bound, lower_bound + 5, 1000)
        f_vals = func_numpy(t_vals)
        ax1.plot(t_vals, f_vals, 'b-', linewidth=2, label=f'f(t) = {func}')
        ax1.set_xlabel('t')
        ax1.set_ylabel('f(t)')
        ax1.set_title('Original Function f(t)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accumulator function F(x)
        ax2.plot(x_vals, F_vals, 'g-', linewidth=2, label=f'F(x) = ∫[{lower_bound} to x] f(t) dt')
        ax2.set_xlabel('x')
        ax2.set_ylabel('F(x)')
        ax2.set_title('Accumulator Function F(x)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Compare F'(x) with f(x)
        ax3.plot(x_vals, F_prime_vals, 'b-', linewidth=2, label='f(x) (theoretical F\'(x))')
        ax3.plot(x_vals, F_prime_numerical, 'r--', linewidth=2, label='F\'(x) (numerical)')
        ax3.set_xlabel('x')
        ax3.set_ylabel("F'(x)")
        ax3.set_title("FTC Part 1: F'(x) = f(x)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate and display error
        error = np.mean(np.abs(np.array(F_prime_vals) - F_prime_numerical))
        print(f"FTC Part 1 Verification:")
        print(f"Function: f(t) = {func}")
        print(f"F(x) = ∫[{lower_bound} to x] f(t) dt")
        print(f"Average error between f(x) and numerical F'(x): {error:.8f}")

    def ftc_part2_demo(self, func_str: str = "x**2", interval: Tuple[float, float] = (0, 2)) -> None:
        """
        Demonstrate FTC Part 2: ∫[a to b] f(x) dx = F(b) - F(a) where F'(x) = f(x)

        Args:
            func_str: Function to integrate
            interval: Integration bounds (a, b)
        """
        func = sp.sympify(func_str)
        a, b = interval

        # Find antiderivative symbolically
        try:
            antiderivative = sp.integrate(func, self.x)
            print(f"Function: f(x) = {func}")
            print(f"Antiderivative: F(x) = {antiderivative}")

            # Calculate definite integral using FTC Part 2
            F_b = float(antiderivative.subs(self.x, b))
            F_a = float(antiderivative.subs(self.x, a))
            definite_integral = F_b - F_a

            print(f"F({b}) = {F_b:.6f}")
            print(f"F({a}) = {F_a:.6f}")
            print(f"∫[{a} to {b}] f(x) dx = F({b}) - F({a}) = {definite_integral:.6f}")

        except:
            antiderivative = None
            definite_integral = None
            print("Symbolic integration failed")

        # Numerical verification
        func_numpy = sp.lambdify(self.x, func, 'numpy')
        numerical_integral, error = integrate.quad(func_numpy, a, b)

        print(f"Numerical integration: {numerical_integral:.6f}")
        if definite_integral is not None:
            print(f"Error: {abs(definite_integral - numerical_integral):.8f}")

        # Visualization
        x_vals = np.linspace(a - 1, b + 1, 1000)
        y_vals = func_numpy(x_vals)

        # Area under curve
        x_area = np.linspace(a, b, 1000)
        y_area = func_numpy(x_area)

        plt.figure(figsize=(12, 8))

        # Plot function
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')

        # Shade area under curve
        plt.fill_between(x_area, y_area, alpha=0.3, color='lightblue',
                        label=f'Area = {numerical_integral:.4f}')

        # Mark integration bounds
        plt.axvline(x=a, color='red', linestyle='--', alpha=0.7, label=f'x = {a}')
        plt.axvline(x=b, color='red', linestyle='--', alpha=0.7, label=f'x = {b}')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'FTC Part 2: ∫[{a} to {b}] f(x) dx')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Plot antiderivative if available
        if antiderivative is not None:
            antideriv_numpy = sp.lambdify(self.x, antiderivative, 'numpy')
            F_vals = antideriv_numpy(x_vals)

            plt.figure(figsize=(12, 6))
            plt.plot(x_vals, F_vals, 'g-', linewidth=2, label=f'F(x) = {antiderivative}')
            plt.plot([a, b], [F_a, F_b], 'ro', markersize=8, label='F(a) and F(b)')
            plt.xlabel('x')
            plt.ylabel('F(x)')
            plt.title('Antiderivative Function')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    def mean_value_theorem_demo(self, func_str: str = "x**3 - 3*x**2 + 2*x",
                               interval: Tuple[float, float] = (0, 3)) -> None:
        """
        Demonstrate the Mean Value Theorem for integrals.

        Args:
            func_str: Function to analyze
            interval: Interval [a, b]
        """
        func = sp.sympify(func_str)
        func_numpy = sp.lambdify(self.x, func, 'numpy')
        a, b = interval

        # Calculate average value of function over interval
        integral_value, _ = integrate.quad(func_numpy, a, b)
        average_value = integral_value / (b - a)

        # Find point c where f(c) = average value (if it exists)
        x_vals = np.linspace(a, b, 1000)
        y_vals = func_numpy(x_vals)

        # Find closest point to average value
        diff = np.abs(y_vals - average_value)
        min_idx = np.argmin(diff)
        c = x_vals[min_idx]
        f_c = y_vals[min_idx]

        plt.figure(figsize=(12, 8))

        # Plot function
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')

        # Fill area under curve
        plt.fill_between(x_vals, y_vals, alpha=0.3, color='lightblue',
                        label=f'Area = {integral_value:.4f}')

        # Plot average value line
        plt.axhline(y=average_value, color='red', linestyle='--',
                   label=f'Average value = {average_value:.4f}')

        # Mark the point c
        plt.plot(c, f_c, 'ro', markersize=10, label=f'f({c:.3f}) ≈ {f_c:.4f}')

        # Mark interval bounds
        plt.axvline(x=a, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=b, color='gray', linestyle=':', alpha=0.5)

        # Draw rectangle representing average value
        rect = patches.Rectangle((a, 0), b-a, average_value,
                               linewidth=2, edgecolor='orange',
                               facecolor='none', linestyle='-',
                               label='Rectangle with same area')
        plt.gca().add_patch(rect)

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Mean Value Theorem for Integrals on [{a}, {b}]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Function: f(x) = {func}")
        print(f"Interval: [{a}, {b}]")
        print(f"Integral: ∫[{a} to {b}] f(x) dx = {integral_value:.6f}")
        print(f"Average value: {average_value:.6f}")
        print(f"Point c ≈ {c:.6f} where f(c) ≈ {f_c:.6f}")
        print(f"Rectangle area: {(b-a) * average_value:.6f}")

    def net_change_theorem_demo(self, rate_func_str: str = "2*x + 1",
                               interval: Tuple[float, float] = (0, 3)) -> None:
        """
        Demonstrate the Net Change Theorem: ∫[a to b] f'(x) dx = f(b) - f(a)

        Args:
            rate_func_str: Rate of change function f'(x)
            interval: Time interval [a, b]
        """
        rate_func = sp.sympify(rate_func_str)

        # Find the position function by integration
        try:
            position_func = sp.integrate(rate_func, self.x)
            print(f"Rate function: f'(x) = {rate_func}")
            print(f"Position function: f(x) = {position_func} + C")

        except:
            print("Could not find symbolic antiderivative")
            return

        a, b = interval

        # Calculate net change
        rate_numpy = sp.lambdify(self.x, rate_func, 'numpy')
        net_change, _ = integrate.quad(rate_numpy, a, b)

        # For visualization, assume C = 0
        position_numpy = sp.lambdify(self.x, position_func, 'numpy')

        x_vals = np.linspace(a - 1, b + 1, 1000)
        rate_vals = rate_numpy(x_vals)
        position_vals = position_numpy(x_vals)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot rate function
        ax1.plot(x_vals, rate_vals, 'b-', linewidth=2, label=f"Rate: f'(x) = {rate_func}")

        # Shade area under rate curve
        x_area = np.linspace(a, b, 1000)
        rate_area = rate_numpy(x_area)
        ax1.fill_between(x_area, rate_area, alpha=0.3, color='lightgreen',
                        label=f'Net change = {net_change:.4f}')

        ax1.axvline(x=a, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(x=b, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('x (time)')
        ax1.set_ylabel("f'(x) (rate)")
        ax1.set_title('Rate of Change Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot position function
        ax2.plot(x_vals, position_vals, 'g-', linewidth=2, label=f'Position: f(x) = {position_func}')

        # Mark change in position
        f_a = position_numpy(a)
        f_b = position_numpy(b)
        ax2.plot([a, b], [f_a, f_b], 'ro', markersize=8)
        ax2.plot([a, b], [f_a, f_b], 'r--', linewidth=2, alpha=0.7,
                label=f'Change: f({b}) - f({a}) = {f_b - f_a:.4f}')

        ax2.set_xlabel('x (time)')
        ax2.set_ylabel('f(x) (position)')
        ax2.set_title('Position Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"Net Change Theorem Verification:")
        print(f"∫[{a} to {b}] f'(x) dx = {net_change:.6f}")
        print(f"f({b}) - f({a}) = {f_b:.6f} - {f_a:.6f} = {f_b - f_a:.6f}")
        print(f"Error: {abs(net_change - (f_b - f_a)):.8f}")


def interactive_ftc_explorer():
    """Interactive exploration of FTC concepts."""
    print("=== Interactive FTC Explorer ===")
    print("Exploring different functions and their properties under FTC")

    test_functions = [
        "x**2",
        "sin(x)",
        "exp(x)",
        "1/(x**2 + 1)",
        "x*exp(-x**2/2)"
    ]

    ftc = FundamentalTheorem()

    for i, func in enumerate(test_functions):
        print(f"\n--- Function {i+1}: f(x) = {func} ---")

        try:
            # Test Riemann sums
            ftc.riemann_sum_visualization(func, (0, 2), 20)

            # Test FTC Part 1
            ftc.ftc_part1_demo(func, 0)

            # Test FTC Part 2
            ftc.ftc_part2_demo(func, (0, 2))

        except Exception as e:
            print(f"Error with function {func}: {e}")


if __name__ == "__main__":
    ftc = FundamentalTheorem()

    print("=== Fundamental Theorem of Calculus Demonstrations ===\n")

    # Riemann sum visualization
    print("1. Riemann Sum Approximation")
    ftc.riemann_sum_visualization("x**2", (0, 2), 10)

    print("\n2. FTC Part 1: Derivative of Integral")
    ftc.ftc_part1_demo("2*x", 0)

    print("\n3. FTC Part 2: Evaluating Definite Integrals")
    ftc.ftc_part2_demo("x**2", (0, 2))

    print("\n4. Mean Value Theorem for Integrals")
    ftc.mean_value_theorem_demo("x**3 - 3*x**2 + 2*x", (0, 3))

    print("\n5. Net Change Theorem")
    ftc.net_change_theorem_demo("2*x + 1", (0, 3))

    # Interactive exploration
    interactive_ftc_explorer()
