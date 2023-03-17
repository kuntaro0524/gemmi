import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import random
from scipy.stats import pearsonr

# ダミーデータを作成する
ds2=np.arange(0,10,0.1)
noise1 = np.random.normal(loc=0,scale=2,size=len(ds2))
noise2 = np.random.normal(loc=0,scale=1,size=len(ds2))
valA = ds2 * -2.0 + 5.0 + noise1
valB = ds2 * -8.0 + 15.0 + noise2

print(random.uniform(10,1000))

#plt.scatter(ds2,valA)
#plt.show()

data={'valA': valA, 'valB':valB}
df = pd.DataFrame(data)

# 最小二乗法によってscaleとoffsetを求める
regressor = LinearRegression()
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,")
print(type(df[['valA']]))
print(type(df['valA']))
print(df[['valA']],df['valA'])
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,")
#regressor.fit(df[['valA']], df['valB'])
regressor.fit(df[['valA']], df['valB'])
scale = regressor.coef_[0]
offset = regressor.intercept_

new_array = scale*df['valA'] + offset

# Correlation coefficient tests
print(df['valA'])
corr1,p1 = pearsonr(df['valA'],df['valB'])
corr2,p2 = pearsonr(new_array,df['valB'])
print("Before:",corr1)
print("After :",corr2)

# Difference 
diff1 = np.power((df['valA'] - df['valB']),2)
sum1 = np.sum(diff1)
diff2 = np.power((new_array - df['valB']),2)
sum2 = np.sum(diff2)

comment1="sum(A-B)^2=%8.2f\nsum(A'-B)^2=%8.2f"%(sum1,sum2)
comment2="CC(orig)=%8.3f CC(mod)=%8.3f"%(corr1,corr2)

plt.scatter(ds2,valA,label="valA")
plt.scatter(ds2,valB,label="valB")
plt.scatter(ds2,new_array,label="scaled A values")
plt.text(1,-40,comment1)
plt.text(1,-60,comment2)
plt.legend()

plt.show()
# plt.show()
# plt.scatter(ds2,valB,label="valB")
# plt.show()

print(f"scale: {scale}, offset: {offset}")