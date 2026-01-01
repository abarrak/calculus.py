'''
' Common Integrals Library and Visualization
' ------------------------------------------
'
' This module provides a comprehensive library of common integrals with
' interactive visualizations, area calculations, and integration techniques.
'
' @file: common_integrals.py
' @authors: Claude Sonnet 4, Abdullah Barrak.
' @date: 1/1/2026
'''

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.patches as patches
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')


class CommonIntegrals:
    """A comprehensive library of common integrals with visualizations."""

    def __init__(self):
        self.x = sp.Symbol('x')
        self.a = sp.Symbol('a', positive=True)
        self.n = sp.Symbol('n')
        self.C = sp.Symbol('C')  # Integration constant

        # Dictionary of common integral formulas
        self.integral_formulas = {
            # Basic functions
            "constant": {
                "function": "c",
                "integral": "cx + C",
                "example": "5",
                "antiderivative": "5*x"
            },
            "power": {
                "function": "x^n",
                "integral": "x^(n+1)/(n+1) + C",
                "example": "x**3",
                "antiderivative": "x**4/4"
            },
            "reciprocal": {
                "function": "1/x",
                "integral": "ln|x| + C",
                "example": "1/x",
                "antiderivative": "log(x)"
            },
            "square_root": {
                "function": "√x",
                "integral": "(2/3)x^(3/2) + C",
                "example": "sqrt(x)",
                "antiderivative": "2*x**(3/2)/3"
            },

            # Exponential functions
            "exponential_e": {
                "function": "eˣ",
                "integral": "eˣ + C",
                "example": "exp(x)",
                "antiderivative": "exp(x)"
            },
            "exponential_a": {
                "function": "aˣ",
                "integral": "aˣ/ln(a) + C",
                "example": "2**x",
                "antiderivative": "2**x/log(2)"
            },

            # Trigonometric functions
            "sin": {
                "function": "sin(x)",
                "integral": "-cos(x) + C",
                "example": "sin(x)",
                "antiderivative": "-cos(x)"
            },
            "cos": {
                "function": "cos(x)",
                "integral": "sin(x) + C",
                "example": "cos(x)",
                "antiderivative": "sin(x)"
            },
            "tan": {
                "function": "tan(x)",
                "integral": "-ln|cos(x)| + C",
                "example": "tan(x)",
                "antiderivative": "-log(cos(x))"
            },
            "sec_squared": {
                "function": "sec²(x)",
                "integral": "tan(x) + C",
                "example": "sec(x)**2",
                "antiderivative": "tan(x)"
            },
            "csc_squared": {
                "function": "csc²(x)",
                "integral": "-cot(x) + C",
                "example": "csc(x)**2",
                "antiderivative": "-cot(x)"
            },

            # Rational functions
            "rational_basic": {
                "function": "1/(x²+a²)",
                "integral": "(1/a)arctan(x/a) + C",
                "example": "1/(x**2 + 1)",
                "antiderivative": "atan(x)"
            },
            "rational_sqrt": {
                "function": "1/√(a²-x²)",
                "integral": "arcsin(x/a) + C",
                "example": "1/sqrt(1 - x**2)",
                "antiderivative": "asin(x)"
            }
        }

    def display_integral_table(self) -> None:
        """Display a formatted table of common integrals."""
        print("=" * 90)
        print("COMMON INTEGRALS REFERENCE TABLE")
        print("=" * 90)
        print(f"{'Function':<20} {'Integral':<30} {'Example':<25}")
        print("-" * 90)

        for name, formula in self.integral_formulas.items():
            print(f"{formula['function']:<20} {formula['integral']:<30} {formula['example']:<25}")

        print("=" * 90)

    def visualize_integral_as_area(self, func_str: str, interval: Tuple[float, float] = (0, 2),
                                  n_rectangles: int = 50) -> None:
        """
        Visualize definite integral as area under the curve.

        Args:
            func_str: Function to integrate
            interval: Integration bounds
            n_rectangles: Number of rectangles for Riemann sum
        """
        func = sp.sympify(func_str)

        try:
            func_numpy = sp.lambdify(self.x, func, 'numpy')
            a, b = interval

            # Calculate area
            area, error = integrate.quad(func_numpy, a, b)

            # Create x values for smooth curve
            x_vals = np.linspace(a - 0.5, b + 0.5, 1000)
            y_vals = func_numpy(x_vals)

            # Area under curve
            x_area = np.linspace(a, b, 1000)
            y_area = func_numpy(x_area)

            # Riemann rectangles
            dx = (b - a) / n_rectangles
            x_rects = np.linspace(a, b - dx, n_rectangles)
            y_rects = func_numpy(x_rects + dx/2)  # Midpoint rule

            plt.figure(figsize=(12, 8))

            # Plot function
            plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')

            # Fill area under curve
            plt.fill_between(x_area, y_area, alpha=0.3, color='lightblue',
                           label=f'Area = {area:.4f}')

            # Draw Riemann rectangles
            for i, (x_rect, y_rect) in enumerate(zip(x_rects, y_rects)):
                rect = patches.Rectangle((x_rect, 0), dx, y_rect,
                                       linewidth=0.5, edgecolor='red',
                                       facecolor='none', alpha=0.7)
                plt.gca().add_patch(rect)

            # Mark boundaries
            plt.axvline(x=a, color='red', linestyle='--', alpha=0.7, label=f'x = {a}')
            plt.axvline(x=b, color='red', linestyle='--', alpha=0.7, label=f'x = {b}')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f'∫[{a} to {b}] {func} dx = {area:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # Calculate symbolic integral if possible
            try:
                symbolic_integral = sp.integrate(func, self.x)
                definite_symbolic = sp.integrate(func, (self.x, a, b))
                print(f"Indefinite integral: ∫ {func} dx = {symbolic_integral} + C")
                print(f"Definite integral: ∫[{a} to {b}] {func} dx = {definite_symbolic}")
            except:
                print("Symbolic integration not available for this function")

        except Exception as e:
            print(f"Error visualizing integral: {e}")

    def compare_integration_methods(self, func_str: str, interval: Tuple[float, float] = (0, 2)) -> None:
        """
        Compare different numerical integration methods.

        Args:
            func_str: Function to integrate
            interval: Integration bounds
        """
        func = sp.sympify(func_str)
        func_numpy = sp.lambdify(self.x, func, 'numpy')
        a, b = interval

        # Different methods
        methods = {
            'Left Riemann': 'left',
            'Right Riemann': 'right',
            'Midpoint': 'midpoint',
            'Trapezoidal': 'trapezoidal'
        }

        # Test with different numbers of subdivisions
        n_values = np.array([5, 10, 20, 50, 100, 200])

        # Get exact value for comparison
        exact_value, _ = integrate.quad(func_numpy, a, b)

        plt.figure(figsize=(15, 10))

        for i, (method_name, method_type) in enumerate(methods.items()):
            errors = []

            for n in n_values:
                dx = (b - a) / n

                if method_type == 'left':
                    x_points = np.linspace(a, b - dx, n)
                    y_points = func_numpy(x_points)
                elif method_type == 'right':
                    x_points = np.linspace(a + dx, b, n)
                    y_points = func_numpy(x_points)
                elif method_type == 'midpoint':
                    x_points = np.linspace(a + dx/2, b - dx/2, n)
                    y_points = func_numpy(x_points)
                elif method_type == 'trapezoidal':
                    x_points = np.linspace(a, b, n + 1)
                    y_points = func_numpy(x_points)
                    # Trapezoidal rule
                    approx = dx * (y_points[0]/2 + np.sum(y_points[1:-1]) + y_points[-1]/2)
                    errors.append(abs(approx - exact_value))
                    continue

                if method_type != 'trapezoidal':
                    approx = np.sum(y_points) * dx
                    errors.append(abs(approx - exact_value))

            # Plot convergence
            plt.subplot(2, 2, i + 1)
            plt.loglog(n_values, errors, 'o-', linewidth=2, markersize=6)
            plt.xlabel('Number of subdivisions')
            plt.ylabel('Absolute error')
            plt.title(f'{method_name} Rule')
            plt.grid(True, alpha=0.3)

        plt.suptitle(f'Integration Method Convergence for f(x) = {func}', fontsize=14)
        plt.tight_layout()
        plt.show()

        print(f"Exact value: {exact_value:.8f}")
        print("\nFinal errors with n=200:")
        for method_name in methods.keys():
            final_error = errors[-1] if method_name == list(methods.keys())[-1] else 0
            print(f"{method_name}: {final_error:.2e}")

    def integration_by_parts_demo(self, u_str: str = "x", dv_str: str = "exp(x)") -> None:
        """
        Demonstrate integration by parts: ∫u dv = uv - ∫v du

        Args:
            u_str: u function
            dv_str: dv function
        """
        u = sp.sympify(u_str)
        dv = sp.sympify(dv_str)

        # Calculate du and v
        du = sp.diff(u, self.x)
        v = sp.integrate(dv, self.x)

        # Integration by parts formula
        uv = u * v
        v_du_integral = sp.integrate(v * du, self.x)
        result = uv - v_du_integral

        # Original integral
        original_integrand = u * dv

        print("=== Integration by Parts Demonstration ===")
        print(f"∫ u dv where u = {u}, dv = {dv}")
        print(f"du = {du}")
        print(f"v = {v}")
        print(f"∫ u dv = uv - ∫ v du")
        print(f"= ({u})({v}) - ∫ ({v})({du}) dx")
        print(f"= {uv} - ∫ {v * du} dx")
        print(f"= {uv} - ({v_du_integral})")
        print(f"= {sp.simplify(result)} + C")

        # Verify by differentiation
        verification = sp.diff(result, self.x)
        simplified_verification = sp.simplify(verification)
        simplified_original = sp.simplify(original_integrand)

        print(f"\nVerification by differentiation:")
        print(f"d/dx[{result}] = {verification}")
        print(f"Simplified: {simplified_verification}")
        print(f"Original integrand: {simplified_original}")
        print(f"Match: {sp.simplify(simplified_verification - simplified_original) == 0}")

        # Visualization
        try:
            self.visualize_integral_as_area(str(original_integrand), (0, 2))
        except:
            print("Could not create visualization")

    def substitution_method_demo(self, func_str: str = "2*x*exp(x**2)",
                                substitution: str = "x**2") -> None:
        """
        Demonstrate u-substitution method.

        Args:
            func_str: Function to integrate
            substitution: Substitution u = ...
        """
        func = sp.sympify(func_str)
        u = sp.sympify(substitution)

        print("=== U-Substitution Demonstration ===")
        print(f"∫ {func} dx")
        print(f"Let u = {u}")

        # Calculate du/dx
        du_dx = sp.diff(u, self.x)
        print(f"Then du = {du_dx} dx")
        print(f"So dx = du/({du_dx})")

        # Try to express the integrand in terms of u
        # This is a simplified demonstration
        try:
            # Express func in terms of u
            x_in_terms_of_u = sp.solve(sp.Eq(u, substitution), self.x)[0]
            func_in_u = func.subs(self.x, x_in_terms_of_u)

            print(f"Substituting: ∫ {func_in_u} * (1/{du_dx}) du")

            # Integrate with respect to u
            integrand_u = func_in_u / du_dx
            integral_u = sp.integrate(integrand_u, u)

            # Substitute back
            final_result = integral_u.subs(u, substitution)

            print(f"= ∫ {integrand_u} du")
            print(f"= {integral_u} + C")
            print(f"= {final_result} + C")

            # Verify
            verification = sp.diff(final_result, self.x)
            print(f"\nVerification: d/dx[{final_result}] = {sp.simplify(verification)}")
            print(f"Original: {sp.simplify(func)}")
            print(f"Match: {sp.simplify(verification - func) == 0}")

        except Exception as e:
            print(f"Symbolic substitution failed: {e}")
            print("This demonstrates the concept, but symbolic manipulation is complex.")

    def improper_integrals_demo(self) -> None:
        """Demonstrate improper integrals with convergence analysis."""

        examples = [
            ("1/x**2", (1, sp.oo), "Convergent"),
            ("1/x", (1, sp.oo), "Divergent"),
            ("exp(-x)", (0, sp.oo), "Convergent"),
            ("1/sqrt(x)", (0, 1), "Convergent"),
            ("1/x", (0, 1), "Divergent")
        ]

        print("=== Improper Integrals Demonstration ===")

        for func_str, (a, b), expected in examples:
            print(f"\n∫[{a} to {b}] {func_str} dx")

            func = sp.sympify(func_str)

            try:
                # Calculate improper integral symbolically
                result = sp.integrate(func, (self.x, a, b))
                print(f"Result: {result}")
                print(f"Expected: {expected}")

                # Visualize for finite bounds
                if b != sp.oo and a != 0 or (a == 0 and '1/x' not in func_str):
                    # Create visualization
                    if b == sp.oo:
                        b_viz = 10  # Use large finite value for visualization
                    else:
                        b_viz = float(b)

                    if a == 0:
                        a_viz = 0.01  # Avoid singularity
                    else:
                        a_viz = float(a)

                    try:
                        self.visualize_integral_as_area(func_str, (a_viz, b_viz))
                    except:
                        pass

            except Exception as e:
                print(f"Could not evaluate: {e}")

    def area_between_curves(self, func1_str: str = "x**2", func2_str: str = "x + 2",
                           interval: Optional[Tuple[float, float]] = None) -> None:
        """
        Calculate and visualize area between two curves.

        Args:
            func1_str: First function
            func2_str: Second function
            interval: Integration bounds (if None, find intersections)
        """
        func1 = sp.sympify(func1_str)
        func2 = sp.sympify(func2_str)

        print(f"=== Area Between Curves ===")
        print(f"f(x) = {func1}")
        print(f"g(x) = {func2}")

        # Find intersection points if interval not provided
        if interval is None:
            intersections = sp.solve(func1 - func2, self.x)
            real_intersections = [float(pt.evalf()) for pt in intersections if pt.is_real]

            if len(real_intersections) >= 2:
                a, b = sorted(real_intersections)[:2]
                interval = (a, b)
                print(f"Intersection points: {real_intersections}")
            else:
                print("Could not find two intersection points, using default interval")
                interval = (-2, 2)

        a, b = interval
        print(f"Integration interval: [{a}, {b}]")

        # Determine which function is on top
        func1_numpy = sp.lambdify(self.x, func1, 'numpy')
        func2_numpy = sp.lambdify(self.x, func2, 'numpy')

        mid_point = (a + b) / 2
        f1_mid = func1_numpy(mid_point)
        f2_mid = func2_numpy(mid_point)

        if f1_mid > f2_mid:
            top_func, bottom_func = func1, func2
            top_numpy, bottom_numpy = func1_numpy, func2_numpy
            top_str, bottom_str = func1_str, func2_str
        else:
            top_func, bottom_func = func2, func1
            top_numpy, bottom_numpy = func2_numpy, func1_numpy
            top_str, bottom_str = func2_str, func1_str

        # Calculate area
        difference = top_func - bottom_func
        area = sp.integrate(difference, (self.x, a, b))

        print(f"Top function: {top_str}")
        print(f"Bottom function: {bottom_str}")
        print(f"Area = ∫[{a} to {b}] ({top_str}) - ({bottom_str}) dx")
        print(f"Area = ∫[{a} to {b}] {difference} dx = {float(area.evalf()):.4f}")

        # Visualization
        x_vals = np.linspace(a - 1, b + 1, 1000)
        y1_vals = func1_numpy(x_vals)
        y2_vals = func2_numpy(x_vals)

        # Area between curves
        x_area = np.linspace(a, b, 1000)
        y1_area = func1_numpy(x_area)
        y2_area = func2_numpy(x_area)

        plt.figure(figsize=(12, 8))

        # Plot both functions
        plt.plot(x_vals, y1_vals, 'b-', linewidth=2, label=f'f(x) = {func1}')
        plt.plot(x_vals, y2_vals, 'r-', linewidth=2, label=f'g(x) = {func2}')

        # Fill area between curves
        plt.fill_between(x_area, y1_area, y2_area, alpha=0.3, color='green',
                        label=f'Area = {float(area.evalf()):.4f}')

        # Mark intersection points
        if interval == (a, b):
            plt.plot([a, b], [func1_numpy(a), func1_numpy(b)], 'ko',
                    markersize=8, label='Intersection points')

        # Mark integration bounds
        plt.axvline(x=a, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=b, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Area Between f(x) = {func1} and g(x) = {func2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def volume_of_revolution(self, func_str: str = "sqrt(x)",
                           interval: Tuple[float, float] = (0, 4),
                           method: str = "disk") -> None:
        """
        Calculate volume of revolution using disk/washer method.

        Args:
            func_str: Function to revolve
            interval: Integration bounds
            method: "disk" or "shell"
        """
        func = sp.sympify(func_str)
        a, b = interval

        print(f"=== Volume of Revolution ({method} method) ===")
        print(f"Function: f(x) = {func}")
        print(f"Revolving around x-axis from x = {a} to x = {b}")

        if method == "disk":
            # Disk method: V = π ∫[a to b] [f(x)]² dx
            volume_integrand = sp.pi * func**2
            volume = sp.integrate(volume_integrand, (self.x, a, b))

            print(f"Disk method: V = π ∫[{a} to {b}] [f(x)]² dx")
            print(f"V = π ∫[{a} to {b}] [{func}]² dx")
            print(f"V = π ∫[{a} to {b}] {func**2} dx")
            print(f"V = {float(volume.evalf()):.4f}")

        # Visualization
        func_numpy = sp.lambdify(self.x, func, 'numpy')

        fig = plt.figure(figsize=(15, 10))

        # 2D view
        ax1 = fig.add_subplot(221)
        x_vals = np.linspace(a, b, 1000)
        y_vals = func_numpy(x_vals)

        ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {func}')
        ax1.fill_between(x_vals, y_vals, alpha=0.3, color='lightblue')
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Function to be revolved')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 3D visualization (simplified)
        ax2 = fig.add_subplot(222, projection='3d')

        # Create revolution surface
        x_3d = np.linspace(a, b, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        X_3d, Theta = np.meshgrid(x_3d, theta)

        # Radius at each x
        R = func_numpy(X_3d)
        Y_3d = R * np.cos(Theta)
        Z_3d = R * np.sin(Theta)

        ax2.plot_surface(X_3d, Y_3d, Z_3d, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.set_title('Solid of Revolution')

        # Cross-sectional disks
        ax3 = fig.add_subplot(223)

        # Show several disks
        n_disks = 8
        x_disks = np.linspace(a, b, n_disks)

        for i, x_disk in enumerate(x_disks):
            radius = func_numpy(x_disk)
            circle = patches.Circle((x_disk, 0), radius,
                                  fill=False, edgecolor='red', alpha=0.7)
            ax3.add_patch(circle)

            # Show the radius
            ax3.plot([x_disk, x_disk], [0, radius], 'r--', alpha=0.7)

        ax3.plot(x_vals, y_vals, 'b-', linewidth=2)
        ax3.plot(x_vals, -y_vals, 'b-', linewidth=2)
        ax3.set_xlim(a, b)
        ax3.set_ylim(-max(y_vals)*1.1, max(y_vals)*1.1)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Cross-sectional disks')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

        # Volume calculation visualization
        ax4 = fig.add_subplot(224)

        volume_integrand_vals = np.pi * (y_vals**2)
        ax4.plot(x_vals, volume_integrand_vals, 'g-', linewidth=2,
                label=f'π[f(x)]² = π[{func}]²')
        ax4.fill_between(x_vals, volume_integrand_vals, alpha=0.3, color='lightgreen')
        ax4.set_xlabel('x')
        ax4.set_ylabel('π[f(x)]²')
        ax4.set_title('Volume integrand')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def integral_practice_game():
    """Interactive game for practicing integrals."""

    ci = CommonIntegrals()

    # Sample functions for practice
    functions = [
        "x**2", "x**3", "sin(x)", "cos(x)", "exp(x)",
        "1/x", "1/(x**2 + 1)", "x*exp(x)", "sin(x)*cos(x)"
    ]

    print("\n=== Integral Practice Game ===")
    print("I'll show you a function and its integral!")

    for i, func_str in enumerate(functions[:5]):  # Show first 5
        print(f"\nProblem {i+1}: ∫ {func_str} dx = ?")

        func = sp.sympify(func_str)
        try:
            integral = sp.integrate(func, ci.x)

            input("Press Enter to see the answer...")
            print(f"Answer: ∫ {func} dx = {integral} + C")

            # Show visualization
            ci.visualize_integral_as_area(func_str, (0, 2))

        except Exception as e:
            print(f"Could not integrate: {e}")


if __name__ == "__main__":
    ci = CommonIntegrals()

    print("=== Common Integrals Library ===\n")

    # Display reference table
    ci.display_integral_table()

    # Basic integral visualization
    print("\n1. Basic Integral Visualization")
    ci.visualize_integral_as_area("x**2", (0, 2))

    # Integration methods comparison
    print("\n2. Integration Methods Comparison")
    ci.compare_integration_methods("sin(x)", (0, np.pi))

    # Integration by parts
    print("\n3. Integration by Parts")
    ci.integration_by_parts_demo("x", "exp(x)")

    # U-substitution
    print("\n4. U-Substitution Method")
    ci.substitution_method_demo("2*x*exp(x**2)", "x**2")

    # Improper integrals
    print("\n5. Improper Integrals")
    ci.improper_integrals_demo()

    # Area between curves
    print("\n6. Area Between Curves")
    ci.area_between_curves("x**2", "x + 2")

    # Volume of revolution
    print("\n7. Volume of Revolution")
    ci.volume_of_revolution("sqrt(x)", (0, 4))

    # Practice game
    integral_practice_game()
