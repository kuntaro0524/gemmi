import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# 線形関数の定義
def linear_func(x, a, b):
    return a * x + b

# データ数
n = 100

# xの範囲
x_min, x_max = 0, 10

# 誤差の範囲
error_range = 3.0

# xの配列を生成
x_values = np.linspace(x_min, x_max, n)

# 2つのランダムな誤差を含むyの配列を生成
y1_values = 10 * x_values + 5 + np.random.uniform(-error_range, error_range, n)
y2_values = 4 * x_values + 7 + np.random.uniform(-error_range, error_range, n)

# 最小二乗法でパラメータを決定 (Y1, Y2を重ね合わせるための線形変換)
params1, _ = curve_fit(linear_func, y1_values, y2_values)

# 一方、単純にそれぞれの直線を求めて相対的な傾きの違いなどを検討してみる
linear1, _1 = curve_fit(linear_func, x_values, y1_values)
linear2, _2 = curve_fit(linear_func, x_values, y2_values)

print(linear1,linear2)
# Relative slope ratio
rel_scale = linear2[0]/linear1[0]
rel_intersect = linear2[1]-linear1[1]
print(rel_scale,rel_intersect)
print(params1)

new_y2 = linear_func(y1_values, params1[0], params1[1])

plt.plot(x_values,y1_values)
plt.plot(x_values,y2_values)
plt.plot(x_values,new_y2)
plt.show()