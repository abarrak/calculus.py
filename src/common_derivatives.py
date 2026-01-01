"""
Common Derivatives Library and Visualization
===========================================

This module provides a comprehensive library of common derivatives with
interactive visualizations and pattern recognition tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, List, Tuple, Callable
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D


class CommonDerivatives:
    """A comprehensive library of common derivatives with visualizations."""

    def __init__(self):
        self.x = sp.Symbol('x')
        self.a = sp.Symbol('a', positive=True)
        self.n = sp.Symbol('n')

        # Dictionary of common derivative formulas
        self.derivative_formulas = {
            # Basic functions
            "constant": {"function": "c", "derivative": "0", "example": "5"},
            "linear": {"function": "ax + b", "derivative": "a", "example": "3*x + 2"},
            "power": {"function": "x^n", "derivative": "n*x^(n-1)", "example": "x**3"},
            "square_root": {"function": "√x", "derivative": "1/(2√x)", "example": "sqrt(x)"},
            "reciprocal": {"function": "1/x", "derivative": "-1/x²", "example": "1/x"},

            # Exponential and logarithmic
            "exponential_e": {"function": "eˣ", "derivative": "eˣ", "example": "exp(x)"},
            "exponential_a": {"function": "aˣ", "derivative": "aˣ ln(a)", "example": "2**x"},
            "natural_log": {"function": "ln(x)", "derivative": "1/x", "example": "log(x)"},
            "log_a": {"function": "log_a(x)", "derivative": "1/(x ln(a))", "example": "log(x, 2)"},

            # Trigonometric functions
            "sin": {"function": "sin(x)", "derivative": "cos(x)", "example": "sin(x)"},
            "cos": {"function": "cos(x)", "derivative": "-sin(x)", "example": "cos(x)"},
            "tan": {"function": "tan(x)", "derivative": "sec²(x)", "example": "tan(x)"},
            "csc": {"function": "csc(x)", "derivative": "-csc(x)cot(x)", "example": "csc(x)"},
            "sec": {"function": "sec(x)", "derivative": "sec(x)tan(x)", "example": "sec(x)"},
            "cot": {"function": "cot(x)", "derivative": "-csc²(x)", "example": "cot(x)"},

            # Inverse trigonometric functions
            "arcsin": {"function": "arcsin(x)", "derivative": "1/√(1-x²)", "example": "asin(x)"},
            "arccos": {"function": "arccos(x)", "derivative": "-1/√(1-x²)", "example": "acos(x)"},
            "arctan": {"function": "arctan(x)", "derivative": "1/(1+x²)", "example": "atan(x)"},

            # Hyperbolic functions
            "sinh": {"function": "sinh(x)", "derivative": "cosh(x)", "example": "sinh(x)"},
            "cosh": {"function": "cosh(x)", "derivative": "sinh(x)", "example": "cosh(x)"},
            "tanh": {"function": "tanh(x)", "derivative": "sech²(x)", "example": "tanh(x)"},
        }

    def display_derivative_table(self) -> None:
        """Display a formatted table of common derivatives."""
        print("=" * 80)
        print("COMMON DERIVATIVES REFERENCE TABLE")
        print("=" * 80)
        print(f"{'Function':<20} {'Derivative':<25} {'Example':<20}")
        print("-" * 80)

        for name, formula in self.derivative_formulas.items():
            print(f"{formula['function']:<20} {formula['derivative']:<25} {formula['example']:<20}")

        print("=" * 80)

    def visualize_function_and_derivative(self, func_str: str, x_range: Tuple[float, float] = (-3, 3),
                                        title: str = None) -> None:
        """
        Visualize a function and its derivative side by side.

        Args:
            func_str: Function as string (SymPy format)
            x_range: Range for x values
            title: Optional title for the plot
        """
        func = sp.sympify(func_str)
        derivative = sp.diff(func, self.x)

        # Convert to numpy functions
        try:
            func_numpy = sp.lambdify(self.x, func, ['numpy', 'math'])
            deriv_numpy = sp.lambdify(self.x, derivative, ['numpy', 'math'])
        except:
            print(f"Could not convert function to numpy: {func}")
            return

        x_vals = np.linspace(x_range[0], x_range[1], 1000)

        try:
            y_vals = func_numpy(x_vals)
            dy_vals = deriv_numpy(x_vals)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot original function
            ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')
            ax1.set_title(f'Function: f(x) = {func}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot derivative
            ax2.plot(x_vals, dy_vals, 'r-', linewidth=2, label=f"f'(x) = {derivative}")
            ax2.set_title(f"Derivative: f'(x) = {derivative}")
            ax2.set_xlabel('x')
            ax2.set_ylabel("f'(x)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            if title:
                fig.suptitle(title, fontsize=16)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting function {func}: {e}")

    def demonstrate_derivative_patterns(self) -> None:
        """Demonstrate common derivative patterns with visualizations."""

        patterns = [
            ("Power Functions", ["x**2", "x**3", "x**4", "x**0.5"]),
            ("Exponential Functions", ["exp(x)", "2**x", "exp(-x)"]),
            ("Trigonometric Functions", ["sin(x)", "cos(x)", "tan(x)"]),
            ("Logarithmic Functions", ["log(x)", "log(x**2)", "x*log(x)"]),
        ]

        for pattern_name, functions in patterns:
            print(f"\n=== {pattern_name} ===")

            fig, axes = plt.subplots(2, len(functions), figsize=(4*len(functions), 8))
            if len(functions) == 1:
                axes = axes.reshape(2, 1)

            for i, func_str in enumerate(functions):
                try:
                    func = sp.sympify(func_str)
                    derivative = sp.diff(func, self.x)

                    func_numpy = sp.lambdify(self.x, func, ['numpy', 'math'])
                    deriv_numpy = sp.lambdify(self.x, derivative, ['numpy', 'math'])

                    if "log" in func_str:
                        x_vals = np.linspace(0.1, 5, 1000)
                    elif "exp" in func_str:
                        x_vals = np.linspace(-2, 2, 1000)
                    else:
                        x_vals = np.linspace(-3, 3, 1000)

                    y_vals = func_numpy(x_vals)
                    dy_vals = deriv_numpy(x_vals)

                    # Plot function
                    axes[0, i].plot(x_vals, y_vals, 'b-', linewidth=2)
                    axes[0, i].set_title(f'f(x) = {func}')
                    axes[0, i].grid(True, alpha=0.3)
                    axes[0, i].set_xlabel('x')
                    axes[0, i].set_ylabel('f(x)')

                    # Plot derivative
                    axes[1, i].plot(x_vals, dy_vals, 'r-', linewidth=2)
                    axes[1, i].set_title(f"f'(x) = {derivative}")
                    axes[1, i].grid(True, alpha=0.3)
                    axes[1, i].set_xlabel('x')
                    axes[1, i].set_ylabel("f'(x)")

                except Exception as e:
                    print(f"Error with function {func_str}: {e}")
                    continue

            plt.suptitle(f'{pattern_name} and Their Derivatives', fontsize=16)
            plt.tight_layout()
            plt.show()

    def slope_field_visualization(self, func_str: str, x_range: Tuple[float, float] = (-3, 3),
                                y_range: Tuple[float, float] = (-3, 3)) -> None:
        """
        Create a slope field visualization for a derivative.

        Args:
            func_str: Function whose derivative defines the slope field
            x_range: Range for x values
            y_range: Range for y values
        """
        func = sp.sympify(func_str)
        derivative = sp.diff(func, self.x)

        try:
            # Create slope function
            slope_func = sp.lambdify(self.x, derivative, 'numpy')
            func_numpy = sp.lambdify(self.x, func, 'numpy')

            # Create grid
            x = np.linspace(x_range[0], x_range[1], 20)
            y = np.linspace(y_range[0], y_range[1], 15)
            X, Y = np.meshgrid(x, y)

            # Calculate slopes at each point
            slopes = slope_func(X)

            # Normalize for better visualization
            dx = np.ones_like(slopes)
            dy = slopes

            # Normalize vectors
            magnitude = np.sqrt(dx**2 + dy**2)
            dx_norm = dx / magnitude * 0.1
            dy_norm = dy / magnitude * 0.1

            plt.figure(figsize=(12, 8))

            # Plot slope field
            plt.quiver(X, Y, dx_norm, dy_norm, slopes, cmap='viridis', alpha=0.7)

            # Plot the actual function if it exists in our y range
            x_continuous = np.linspace(x_range[0], x_range[1], 1000)
            y_continuous = func_numpy(x_continuous)

            # Only plot parts within y_range
            mask = (y_continuous >= y_range[0]) & (y_continuous <= y_range[1])
            if np.any(mask):
                plt.plot(x_continuous[mask], y_continuous[mask], 'r-', linewidth=3,
                        label=f'f(x) = {func}')

            plt.xlim(x_range)
            plt.ylim(y_range)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f"Slope Field for f'(x) = {derivative}")
            plt.colorbar(label='Slope value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            print(f"Error creating slope field: {e}")

    def derivative_chain_rule_examples(self) -> None:
        """Demonstrate chain rule with complex composite functions."""

        examples = [
            "sin(x**2)",
            "exp(cos(x))",
            "log(sin(x))",
            "sqrt(1 + x**2)",
            "(x**2 + 1)**3",
            "sin(exp(x))"
        ]

        print("\n=== Chain Rule Examples ===")

        for func_str in examples:
            try:
                func = sp.sympify(func_str)
                derivative = sp.diff(func, self.x)

                print(f"\nFunction: f(x) = {func}")
                print(f"Derivative: f'(x) = {derivative}")
                print(f"Simplified: f'(x) = {sp.simplify(derivative)}")

                # Visualize if possible
                self.visualize_function_and_derivative(func_str, (-2, 2),
                                                     f"Chain Rule: f(x) = {func}")

            except Exception as e:
                print(f"Error with function {func_str}: {e}")

    def higher_order_derivatives(self, func_str: str = "x**4 - 4*x**3 + 6*x**2 - 4*x + 1",
                                n_derivatives: int = 4) -> None:
        """
        Calculate and visualize higher-order derivatives.

        Args:
            func_str: Function to differentiate
            n_derivatives: Number of derivatives to calculate
        """
        func = sp.sympify(func_str)
        derivatives = [func]

        # Calculate successive derivatives
        current = func
        for i in range(n_derivatives):
            current = sp.diff(current, self.x)
            derivatives.append(current)

        # Create visualizations
        fig, axes = plt.subplots(1, len(derivatives), figsize=(4*len(derivatives), 6))
        if len(derivatives) == 1:
            axes = [axes]

        x_vals = np.linspace(-2, 3, 1000)
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, deriv in enumerate(derivatives):
            try:
                deriv_numpy = sp.lambdify(self.x, deriv, 'numpy')
                y_vals = deriv_numpy(x_vals)

                order = "f(x)" if i == 0 else f"f^({i})(x)"
                axes[i].plot(x_vals, y_vals, color=colors[i % len(colors)],
                           linewidth=2, label=f'{order} = {deriv}')
                axes[i].set_title(f'{order}')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlabel('x')
                axes[i].set_ylabel(f'{order}')

                # Add zero crossings
                zero_crossings = []
                for j in range(len(y_vals)-1):
                    if y_vals[j] * y_vals[j+1] < 0:
                        zero_crossings.append(x_vals[j])

                if zero_crossings:
                    axes[i].plot(zero_crossings, [0]*len(zero_crossings), 'ko', markersize=6)

            except Exception as e:
                print(f"Error plotting derivative {i}: {e}")

        plt.suptitle(f'Higher Order Derivatives of f(x) = {func}', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Print derivatives
        print(f"\nHigher-order derivatives of f(x) = {func}:")
        for i, deriv in enumerate(derivatives):
            order = "f(x)" if i == 0 else f"f^({i})(x)"
            print(f"{order} = {deriv}")

    def critical_points_analysis(self, func_str: str = "x**3 - 3*x**2 + 2") -> None:
        """
        Find and analyze critical points using derivatives.

        Args:
            func_str: Function to analyze
        """
        func = sp.sympify(func_str)
        first_derivative = sp.diff(func, self.x)
        second_derivative = sp.diff(first_derivative, self.x)

        # Find critical points (where f'(x) = 0)
        critical_points = sp.solve(first_derivative, self.x)
        critical_points = [float(cp.evalf()) for cp in critical_points if cp.is_real]

        print(f"Function: f(x) = {func}")
        print(f"First derivative: f'(x) = {first_derivative}")
        print(f"Second derivative: f''(x) = {second_derivative}")
        print(f"Critical points: {critical_points}")

        # Evaluate second derivative at critical points
        func_numpy = sp.lambdify(self.x, func, 'numpy')
        first_deriv_numpy = sp.lambdify(self.x, first_derivative, 'numpy')
        second_deriv_numpy = sp.lambdify(self.x, second_derivative, 'numpy')

        x_vals = np.linspace(min(critical_points)-2, max(critical_points)+2, 1000)
        y_vals = func_numpy(x_vals)
        y_prime = first_deriv_numpy(x_vals)
        y_double_prime = second_deriv_numpy(x_vals)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Plot function
        ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')

        # Mark critical points
        for cp in critical_points:
            y_cp = func_numpy(cp)
            second_deriv_at_cp = second_deriv_numpy(cp)

            if second_deriv_at_cp > 0:
                color, marker, point_type = 'red', 'o', 'Local minimum'
            elif second_deriv_at_cp < 0:
                color, marker, point_type = 'green', '^', 'Local maximum'
            else:
                color, marker, point_type = 'orange', 's', 'Inflection point'

            ax1.plot(cp, y_cp, marker, color=color, markersize=10,
                    label=f'{point_type} at x={cp:.2f}')

            print(f"At x = {cp:.2f}: f''(x) = {second_deriv_at_cp:.2f} → {point_type}")

        ax1.set_title('Function with Critical Points')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')

        # Plot first derivative
        ax2.plot(x_vals, y_prime, 'r-', linewidth=2, label=f"f'(x) = {first_derivative}")
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for cp in critical_points:
            ax2.plot(cp, 0, 'ko', markersize=8)

        ax2.set_title('First Derivative')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel("f'(x)")

        # Plot second derivative
        ax3.plot(x_vals, y_double_prime, 'g-', linewidth=2, label=f"f''(x) = {second_derivative}")
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for cp in critical_points:
            second_val = second_deriv_numpy(cp)
            ax3.plot(cp, second_val, 'ko', markersize=8)

        ax3.set_title('Second Derivative')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('x')
        ax3.set_ylabel("f''(x)")

        plt.tight_layout()
        plt.show()


def derivative_game():
    """Interactive game for practicing derivatives."""

    cd = CommonDerivatives()

    # Sample functions for the game
    functions = [
        "x**2", "x**3", "sin(x)", "cos(x)", "exp(x)", "log(x)",
        "x*sin(x)", "exp(x)*cos(x)", "x**2 + 3*x + 1", "sqrt(x)"
    ]

    print("\n=== Derivative Practice Game ===")
    print("I'll show you a function, and you can see its derivative!")

    for i, func_str in enumerate(functions[:5]):  # Show first 5
        print(f"\nFunction {i+1}: f(x) = {func_str}")

        func = sp.sympify(func_str)
        derivative = sp.diff(func, cd.x)

        input("Press Enter to see the derivative...")
        print(f"Answer: f'(x) = {derivative}")

        # Show visualization
        cd.visualize_function_and_derivative(func_str, (-2, 2))


if __name__ == "__main__":
    cd = CommonDerivatives()

    print("=== Common Derivatives Library ===\n")

    # Display reference table
    cd.display_derivative_table()

    # Demonstrate patterns
    print("\n1. Demonstrating Derivative Patterns")
    cd.demonstrate_derivative_patterns()

    # Chain rule examples
    print("\n2. Chain Rule Examples")
    cd.derivative_chain_rule_examples()

    # Higher order derivatives
    print("\n3. Higher Order Derivatives")
    cd.higher_order_derivatives("x**4 - 4*x**3 + 6*x**2", 3)

    # Critical points analysis
    print("\n4. Critical Points Analysis")
    cd.critical_points_analysis("x**3 - 3*x**2 + 2")

    # Slope field
    print("\n5. Slope Field Visualization")
    cd.slope_field_visualization("x**2", (-2, 2), (-1, 4))

    # Interactive game
    derivative_game()
