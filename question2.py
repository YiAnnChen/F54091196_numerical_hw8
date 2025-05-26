import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)


a, b = -1, 1
n_points = 1000
x_vals = np.linspace(a, b, n_points)
f_vals = f(x_vals)


A = np.vstack([np.ones_like(x_vals), x_vals, x_vals**2]).T


ATA = A.T @ A
ATf = A.T @ f_vals
coeffs = np.linalg.solve(ATA, ATf)  # [c0, c1, c2]


p_vals = coeffs[0] + coeffs[1]*x_vals + coeffs[2]*x_vals**2


error_vals = (f_vals - p_vals)**2
L2_error = np.trapz(error_vals, x_vals)


print("Least Squares Polynomial Approximation (degree 2):")
print(f"P2(x) = {coeffs[0]:.4f} + {coeffs[1]:.4f}x + {coeffs[2]:.4f}x^2")
print(f"L2 Error = {L2_error:.6f}")


