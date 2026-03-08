import math

import matplotlib.pyplot as plt
import numpy as np


def pf_once(p):
    return (2 * p**2 - p * (2 * (1 - p)) ** (-(1 - p) / p)) / (2 * p - 1)


def calculate_pf_upper_bound(p):
    """
    计算 p_f 的上限临界值，适用范围 0.5 < p < 1
    """
    A = (2.0 * (1.0 - p)) ** ((1.0 - p) / p)
    # 分子: 2 - 2*p - 2*p*(1-p)*A
    num = 2.0 - 2.0 * p - 2.0 * p * (1.0 - p) * A
    # 分母: (1-p)*(1-2p)/p * A
    den = (1.0 - p) * (1.0 - 2.0 * p) / p * A
    return num / den


for p in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    # print(0.5 * (1 - p) * (1.0 - (2.0 * (1.0 - p)) ** (1.0 / p)))
    print(p, p - pf_once(p))


p_values = np.linspace(0.51, 0.95, 200)

# 计算两个函数的值
y_once = [p - pf_once(p) for p in p_values]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(p_values, y_once, label="pf_once(p)", linestyle="--", linewidth=2)

# 图表装饰
plt.xlabel("p", fontsize=12)
plt.ylabel("p_f", fontsize=12)
plt.title("Comparison of $p_f$ Boundary Functions ($0.5 < p < 1$)", fontsize=14)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)

# 限制 y 轴范围，防止 pf_once 的极端值破坏图像比例
plt.ylim(0.1, 0.5)

plt.show()
