"""
visualize beta distribution
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
"""
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

# a, b = 1, 9
# mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')  # mean,variance,(Fisher's) skew,kurtosis
# mean, var = beta.stats(a, b)  # 默认只返回 mean/var


aa, bb = 1, 9
# aa, bb = 5, 5


plt.figure()

for i in range(4):
    a, b = aa * 10 ** i, bb * 10 ** i
    left = beta.ppf(0.01, a, b)  # 返回 cdf=0.01 对应的 pdf 边界
    right = beta.ppf(0.99, a, b)

    x = np.linspace(left, right, 100)
    y = beta.pdf(x, a, b)  # 计算 x 每个取值对应的概率密度

    plt.plot(x, y, label=f'a={a}, b={b}')

plt.legend()
plt.show()
