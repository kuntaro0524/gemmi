import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 線形関数の定義
def linear_func(x, a, b):
    return a * x + b

# データ数
n = 100

# xの範囲
x_min, x_max = 0, 10

# 誤差の範囲
error_range = 0.5

# xの配列を生成
x_values = np.linspace(x_min, x_max, n)

# 2つのランダムな誤差を含むyの配列を生成
y1_values = 3 * x_values + 5 + np.random.uniform(-error_range, error_range, n)
y2_values = 4 * x_values + 7 + np.random.uniform(-error_range, error_range, n)

# 最小二乗法でパラメータを決定
params1, _ = curve_fit(linear_func, x_values, y1_values)
params2, _ = curve_fit(linear_func, x_values, y2_values)

# 最小二乗法で求めたパラメータを表示
print("Line 1: a = {:.3f}, b = {:.3f}".format(params1[0], params1[1]))
print("Line 2: a = {:.3f}, b = {:.3f}".format(params2[0], params2[1]))

# 相対スケールと相対切片を計算
relative_scale = params2[0] / params1[0]
relative_intercept = params2[1] - params1[1]

print("Relative scale:", relative_scale)
print("Relative intercept:", relative_intercept)

# 新しい配列を生成
new_y1_values = relative_scale * y1_values + relative_intercept

sum1=sum(np.power(y1_values-y2_values,2.0))
sum2=sum(np.power(new_y1_values-y2_values,2.0))
print(sum1,sum2)
# プロット
plt.plot(x_values, y1_values, 'o', label='Original Line 1')
plt.plot(x_values, y2_values, 'o', label='Original Line 2')
plt.plot(x_values, new_y1_values, 'o', label='Transformed Line 1')
plt.legend()
plt.savefig("test.png")
