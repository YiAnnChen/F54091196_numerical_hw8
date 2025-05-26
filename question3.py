import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 原始函數 f(x) = x^2 * sin(x)
def f(x):
    return x**2 * np.sin(x)

# 三角基底函數（a_0/2, cos(2πkx), sin(2πkx)）
def trig_basis(k, x):
    if k == 0:
        return np.ones_like(x)
    elif k % 2 == 1:
        return np.cos((k // 2 + 1) * 2 * np.pi * x)  # cos
    else:
        return np.sin((k // 2) * 2 * np.pi * x)      # sin

# (a) 離散最小平方法係數計算
def S4_coefficients(m, n, x_samples, y_samples):
    coeffs = []
    for k in range(2 * n + 1):
        phi_k = trig_basis(k, x_samples)
        c_k = np.dot(y_samples, phi_k) * 2 / m
        coeffs.append(c_k)
    return coeffs

# 建立 S_4(x) 函數
def S4(x, coeffs):
    result = coeffs[0] / 2  # a0/2
    for k in range(1, len(coeffs)):
        result += coeffs[k] * trig_basis(k, x)
    return result

# 參數設定
m = 16            # 離散點數
n = 4             # 三角多項式階數
x_samples = np.linspace(0, 1, m, endpoint=False)
y_samples = f(x_samples)
coeffs = S4_coefficients(m, n, x_samples, y_samples)

# (b) 計算 ∫₀¹ S₄(x) dx
integral_S4, _ = quad(lambda x: S4(x, coeffs), 0, 1)

# (c) 計算真值 ∫₀¹ x^2 sin(x) dx
integral_true, _ = quad(f, 0, 1)

# (d) 計算平方誤差 E(S₄) = ∫₀¹ (f(x) - S₄(x))² dx
error, _ = quad(lambda x: (f(x) - S4(x, coeffs))**2, 0, 1)

# 顯示結果
print("=== (a) Trigonometric Least Squares Polynomial S₄(x) ===")
print("S₄(x) = (a₀)/2 + Σ [aₖ cos(2πk x) + bₖ sin(2πk x)], k=1~4")
for i, c in enumerate(coeffs):
    basis_type = "a₀/2" if i == 0 else ("a" + str(i//2 + 1) if i % 2 == 1 else "b" + str(i//2))
    print(f"  {basis_type} = {c:.6f}")

print("\n=== (b) ∫₀¹ S₄(x) dx ≈ {:.6f}".format(integral_S4))
print("=== (c) ∫₀¹ x² sin(x) dx ≈ {:.6f}".format(integral_true))
print("=== (d) Error E(S₄) ≈ {:.6f}".format(error))


