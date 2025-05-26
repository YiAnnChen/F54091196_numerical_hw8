import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 基本設定
m = 16
n = 4
f = lambda x: x**2 * np.sin(x)
x_vals = np.linspace(0, 1, m, endpoint=False)
y_vals = f(x_vals)

# (a) 計算傅立葉係數
a0 = (1/m) * np.sum(y_vals)
ak = [(2/m) * np.sum(y_vals * np.cos(2 * np.pi * k * x_vals)) for k in range(1, n+1)]
bk = [(2/m) * np.sum(y_vals * np.sin(2 * np.pi * k * x_vals)) for k in range(1, n+1)]

# 輸出係數 a₀, a₁~a₄, b₁~b₄
print("===== (a) Fourier Coefficients of S₄(x) =====")
print(f"a₀ = {a0:.6f}")
for k in range(1, n+1):
    print(f"a{k} = {ak[k-1]:.6f},  b{k} = {bk[k-1]:.6f}")
print()

# 定義 S₄(x)
def S4(x):
    result = a0
    for k in range(1, n+1):
        result += ak[k-1] * np.cos(2 * np.pi * k * x) + bk[k-1] * np.sin(2 * np.pi * k * x)
    return result

# (b) ∫₀¹ S₄(x) dx
integral_S4, _ = quad(S4, 0, 1)

# (c) ∫₀¹ x² sin(x) dx
integral_fx, _ = quad(f, 0, 1)
difference = abs(integral_fx - integral_S4)

# (d) E(S₄)
S4_vals = np.array([S4(x) for x in x_vals])
error = np.sum((y_vals - S4_vals) ** 2)

# 輸出 b, c, d 結果
print("===== (b), (c), (d) Results =====")
print(f"(b) ∫₀¹ S₄(x) dx ≈ {integral_S4:.6f}")
print(f"(c) ∫₀¹ x² sin(x) dx ≈ {integral_fx:.6f}")
print(f"    差值 ≈ {difference:.6f}")
print(f"(d) E(S₄) ≈ {error:.6f}")


