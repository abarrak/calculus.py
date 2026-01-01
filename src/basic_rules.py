"""
Basic Calculus Rules and Visualizations
=======================================

This module implements fundamental calculus rules with interactive visualizations
and educational demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Callable, List, Tuple, Union
import matplotlib.animation as animation
from matplotlib.widgets import Slider


class CalculusRules:
    """A class containing basic calculus rules with visualizations."""

    def __init__(self):
        self.x = sp.Symbol('x')
        self.h = sp.Symbol('h')

    def power_rule_demo(self, n: float = 3, x_range: Tuple[float, float] = (-3, 3)) -> None:
        """
        Demonstrate the power rule: d/dx(x^n) = n*x^(n-1)

        Args:
            n: Power exponent
            x_range: Range for x values
        """
        x_vals = np.linspace(x_range[0], x_range[1], 1000)

        # Original function and derivative
        y_original = x_vals**n
        y_derivative = n * x_vals**(n-1) if n != 0 else np.zeros_like(x_vals)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot original function
        ax1.plot(x_vals, y_original, 'b-', linewidth=2, label=f'f(x) = x^{n}')
        ax1.set_title(f'Original Function: f(x) = x^{n}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')

        # Plot derivative
        ax2.plot(x_vals, y_derivative, 'r-', linewidth=2, label=f"f'(x) = {n}x^{n-1}")
        ax2.set_title(f"Derivative: f'(x) = {n}x^{n-1}")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlabel('x')
        ax2.set_ylabel("f'(x)")

        plt.tight_layout()
        plt.show()

    def product_rule_visualization(self, func1_str: str = "x**2", func2_str: str = "sin(x)") -> None:
        """
        Visualize the product rule: (fg)' = f'g + fg'

        Args:
            func1_str: First function as string
            func2_str: Second function as string
        """
        # Parse functions
        f = sp.sympify(func1_str)
        g = sp.sympify(func2_str)

        # Calculate derivatives
        f_prime = sp.diff(f, self.x)
        g_prime = sp.diff(g, self.x)
        product = f * g
        product_derivative = sp.diff(product, self.x)
        product_rule_result = f_prime * g + f * g_prime

        # Convert to numpy functions
        f_func = sp.lambdify(self.x, f, 'numpy')
        g_func = sp.lambdify(self.x, g, 'numpy')
        product_func = sp.lambdify(self.x, product, 'numpy')
        product_deriv_func = sp.lambdify(self.x, product_derivative, 'numpy')

        x_vals = np.linspace(-2*np.pi, 2*np.pi, 1000)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot individual functions
        ax1.plot(x_vals, f_func(x_vals), 'b-', label=f'f(x) = {f}')
        ax1.plot(x_vals, g_func(x_vals), 'g-', label=f'g(x) = {g}')
        ax1.set_title('Individual Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot product
        ax2.plot(x_vals, product_func(x_vals), 'purple', linewidth=2, label=f'f(x)g(x) = {product}')
        ax2.set_title('Product f(x)g(x)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot derivative of product
        ax3.plot(x_vals, product_deriv_func(x_vals), 'r-', linewidth=2, label="(fg)'")
        ax3.set_title('Derivative of Product')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Show product rule components
        f_prime_func = sp.lambdify(self.x, f_prime, 'numpy')
        g_prime_func = sp.lambdify(self.x, g_prime, 'numpy')

        term1 = f_prime_func(x_vals) * g_func(x_vals)
        term2 = f_func(x_vals) * g_prime_func(x_vals)

        ax4.plot(x_vals, term1, '--', label="f'g", alpha=0.7)
        ax4.plot(x_vals, term2, '--', label="fg'", alpha=0.7)
        ax4.plot(x_vals, term1 + term2, 'r-', linewidth=2, label="f'g + fg'")
        ax4.set_title('Product Rule: (fg)\' = f\'g + fg\'')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def chain_rule_animation(self, outer_func: str = "sin(x)", inner_func: str = "x**2") -> None:
        """
        Create an animated demonstration of the chain rule.

        Args:
            outer_func: Outer function as string
            inner_func: Inner function as string
        """
        # Parse functions
        u = sp.sympify(inner_func)  # inner function
        f_u = sp.sympify(outer_func.replace('x', 'u'))  # outer function in terms of u

        # Create composite function
        composite = f_u.subs('u', u)

        # Calculate derivatives using chain rule
        du_dx = sp.diff(u, self.x)
        df_du = sp.diff(f_u, 'u')
        chain_rule_result = df_du.subs('u', u) * du_dx
        direct_derivative = sp.diff(composite, self.x)

        print(f"Inner function u(x) = {u}")
        print(f"Outer function f(u) = {f_u}")
        print(f"Composite function f(u(x)) = {composite}")
        print(f"Chain rule: df/dx = (df/du)(du/dx) = ({df_du}) * ({du_dx})")
        print(f"Simplified: {sp.simplify(chain_rule_result)}")
        print(f"Direct derivative: {sp.simplify(direct_derivative)}")
        print(f"Verification: {sp.simplify(chain_rule_result - direct_derivative) == 0}")

    def quotient_rule_demo(self, numerator: str = "x**2", denominator: str = "x+1") -> None:
        """
        Demonstrate the quotient rule: (f/g)' = (f'g - fg')/g^2

        Args:
            numerator: Numerator function as string
            denominator: Denominator function as string
        """
        f = sp.sympify(numerator)
        g = sp.sympify(denominator)

        f_prime = sp.diff(f, self.x)
        g_prime = sp.diff(g, self.x)

        quotient = f / g
        quotient_derivative = sp.diff(quotient, self.x)
        quotient_rule_result = (f_prime * g - f * g_prime) / (g**2)

        # Create visualization
        x_vals = np.linspace(-5, 5, 1000)

        # Filter out points where denominator is close to zero
        g_func = sp.lambdify(self.x, g, 'numpy')
        g_vals = g_func(x_vals)
        valid_mask = np.abs(g_vals) > 0.01
        x_vals_filtered = x_vals[valid_mask]

        f_func = sp.lambdify(self.x, f, 'numpy')
        quotient_func = sp.lambdify(self.x, quotient, 'numpy')
        quotient_deriv_func = sp.lambdify(self.x, quotient_derivative, 'numpy')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Plot numerator and denominator
        ax1.plot(x_vals, f_func(x_vals), 'b-', label=f'f(x) = {f}')
        ax1.plot(x_vals, g_func(x_vals), 'g-', label=f'g(x) = {g}')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_title('Numerator and Denominator')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot quotient
        ax2.plot(x_vals_filtered, quotient_func(x_vals_filtered), 'purple', linewidth=2,
                label=f'f(x)/g(x) = {f}/{g}')
        ax2.set_title('Quotient f(x)/g(x)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-10, 10)

        # Plot derivative
        ax3.plot(x_vals_filtered, quotient_deriv_func(x_vals_filtered), 'r-', linewidth=2,
                label="(f/g)'")
        ax3.set_title('Derivative of Quotient')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-10, 10)

        plt.tight_layout()
        plt.show()

        print(f"Quotient rule formula: (f/g)' = (f'g - fg')/g²")
        print(f"f'(x) = {f_prime}")
        print(f"g'(x) = {g_prime}")
        print(f"Result: {sp.simplify(quotient_rule_result)}")


def interactive_limit_demo():
    """Interactive demonstration of limits approaching a point."""
    def limit_visualization(func_str: str = "sin(x)/x", approach_point: float = 0):
        """Visualize limits as x approaches a point."""
        x = sp.Symbol('x')
        func = sp.sympify(func_str)

        # Calculate limit
        try:
            limit_val = float(sp.limit(func, x, approach_point))
            print(f"lim(x→{approach_point}) {func} = {limit_val}")
        except:
            limit_val = None
            print(f"Limit may not exist or is complex")

        # Create visualization
        x_vals = np.linspace(approach_point - 2, approach_point + 2, 1000)

        # Remove the point of approach to show discontinuity
        mask = np.abs(x_vals - approach_point) > 0.001
        x_filtered = x_vals[mask]

        func_numpy = sp.lambdify(x, func, 'numpy')

        try:
            y_vals = func_numpy(x_filtered)

            plt.figure(figsize=(12, 8))
            plt.plot(x_filtered, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')

            if limit_val is not None:
                plt.plot(approach_point, limit_val, 'ro', markersize=8,
                        label=f'Limit = {limit_val:.4f}')
                plt.axhline(y=limit_val, color='r', linestyle='--', alpha=0.5)

            plt.axvline(x=approach_point, color='g', linestyle='--', alpha=0.5,
                       label=f'x → {approach_point}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f'Limit as x approaches {approach_point}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            print(f"Error plotting function: {e}")

    # Example demonstrations
    print("=== Limit Demonstrations ===")
    limit_visualization("sin(x)/x", 0)
    limit_visualization("(x**2 - 1)/(x - 1)", 1)
    limit_visualization("(x**2 - 4)/(x - 2)", 2)


if __name__ == "__main__":
    # Create an instance and run demonstrations
    rules = CalculusRules()

    print("=== Basic Calculus Rules Demonstrations ===\n")

    # Power rule
    print("1. Power Rule Demonstration")
    rules.power_rule_demo(3)

    # Product rule
    print("\n2. Product Rule Visualization")
    rules.product_rule_visualization("x**2", "sin(x)")

    # Chain rule
    print("\n3. Chain Rule Example")
    rules.chain_rule_animation("sin(x)", "x**2")

    # Quotient rule
    print("\n4. Quotient Rule Demonstration")
    rules.quotient_rule_demo("x**2", "x+1")

    # Limits
    print("\n5. Limit Demonstrations")
    interactive_limit_demo()
