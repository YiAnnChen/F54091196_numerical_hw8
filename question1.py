import numpy as np
import matplotlib.pyplot as plt

# 原始資料
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# (a) 二次多項式 y = a2*x^2 + a1*x + a0
coeffs_quad = np.polyfit(x, y, 2)
y_quad = np.polyval(coeffs_quad, x)
error_quad = np.sum((y - y_quad) ** 2)

# (b) 指數形式 y = b * e^(a * x)
# 取 log 轉為線性關係 ln(y) = ln(b) + a*x
log_y = np.log(y)
A_exp = np.vstack([x, np.ones(len(x))]).T
a_exp, log_b_exp = np.linalg.lstsq(A_exp, log_y, rcond=None)[0]
b_exp = np.exp(log_b_exp)
y_exp = b_exp * np.exp(a_exp * x)
error_exp = np.sum((y - y_exp) ** 2)

# (c) 次方形式 y = b * x^n
# 取 log 轉為線性關係 ln(y) = ln(b) + n*ln(x)
log_x = np.log(x)
A_pow = np.vstack([log_x, np.ones(len(x))]).T
n_pow, log_b_pow = np.linalg.lstsq(A_pow, np.log(y), rcond=None)[0]
b_pow = np.exp(log_b_pow)
y_pow = b_pow * x ** n_pow
error_pow = np.sum((y - y_pow) ** 2)

# 顯示結果
print("=== Least Squares Approximation Results ===")
print(f"(a) Quadratic: y = {coeffs_quad[0]:.4f}x² + {coeffs_quad[1]:.4f}x + {coeffs_quad[2]:.4f}")
print(f"    Error = {error_quad:.4f}")
print(f"(b) Exponential: y = {b_exp:.4f} * e^({a_exp:.4f}x)")
print(f"    Error = {error_exp:.4f}")
print(f"(c) Power: y = {b_pow:.4f} * x^{n_pow:.4f}")
print(f"    Error = {error_pow:.4f}")


