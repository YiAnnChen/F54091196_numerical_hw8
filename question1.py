import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

# 原始資料
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# (a) 多項式擬合 (二次)
poly2 = Polynomial.fit(x, y, deg=2)
y_poly2 = poly2(x)
error_poly2 = np.sum((y - y_poly2) ** 2)

# 顯示多項式參數
coeffs_poly2 = poly2.convert().coef
print("a. Polynomial degree 2 approximation:")
print(f"   y = {coeffs_poly2[0]:.4f} + {coeffs_poly2[1]:.4f}x + {coeffs_poly2[2]:.4f}x^2")
print(f"   Error = {error_poly2:.6f}\n")

# (b) 指數擬合 y = b * e^(a * x)
def model_exp(x, a, b):
    return b * np.exp(a * x)

params_exp, _ = curve_fit(model_exp, x, y)
y_exp = model_exp(x, *params_exp)
error_exp = np.sum((y - y_exp) ** 2)

print("b. Exponential approximation (y = b * e^(a * x)):")
print(f"   y = {params_exp[1]:.4f} * exp({params_exp[0]:.4f} * x)")
print(f"   Error = {error_exp:.6f}\n")

# (c) 冪次擬合 y = b * x^n
def model_power(x, n, b):
    return b * x ** n

params_power, _ = curve_fit(model_power, x, y)
y_power = model_power(x, *params_power)
error_power = np.sum((y - y_power) ** 2)

print("c. Power approximation (y = b * x^n):")
print(f"   y = {params_power[1]:.4f} * x^{params_power[0]:.4f}")
print(f"   Error = {error_power:.6f}\n")

