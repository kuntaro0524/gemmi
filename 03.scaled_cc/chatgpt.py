import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import random

# ダミーデータを作成する
ds2=np.arange(0,10,1.0)
noise1 = np.random.normal(loc=0,scale=2,size=len(ds2))
noise2 = np.random.normal(loc=0,scale=1,size=len(ds2))
valA = ds2 * 2.0 + 10.0 + noise1
valB = ds2 * 4.0 + 5.0 + noise2

print(random.uniform(10,1000))

plt.scatter(ds2,valA)
plt.show()

data={'valA': valA, 'valB':valB}
df = pd.DataFrame(data)

# 最小二乗法によってscaleとoffsetを求める
regressor = LinearRegression()
regressor.fit(df[['valA']], df['valB'])
scale = regressor.coef_[0]
offset = regressor.intercept_

new_array = scale*df['valA'] + offset

plt.scatter(ds2,valA,label="valA")
plt.scatter(ds2,valB,label="valB")
plt.scatter(ds2,new_array,label="valC")
plt.legend()

plt.show()
# plt.show()
# plt.scatter(ds2,valB,label="valB")
# plt.show()

print(f"scale: {scale}, offset: {offset}")