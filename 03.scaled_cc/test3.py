import sys
import gemmi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

mtz = gemmi.read_mtz_file(sys.argv[1])

cell1 = mtz.cell
print(cell1.a,cell1.b,cell1.c)
print(mtz.spacegroup)
print(mtz.title)
print(mtz.history)
print(mtz.nreflections)
print(mtz.sort_order)
print(mtz.min_1_d2)
print(mtz.max_1_d2)

imean=mtz.get_float("IMEAN")

# make DataFrame from MTZ file
mtz_data = np.array(mtz, copy=False)
mtz_df = pd.DataFrame(data=mtz_data, columns=mtz.column_labels())
# (optional) store Miller indices as integers
mtz_df1 = mtz_df.astype({label: 'int32' for label in 'HKL'})
cell1 = mtz.cell
print(cell1.a,cell1.b,cell1.c)

# 線形関数の定義
def linear_func(x, a, b):
    return a * x + b

def abc_stars(cells):
    a=cells.a
    b=cells.b
    c=cells.c
    alpha=np.radians(cells.alpha)
    beta=np.radians(cells.beta)
    gamma=np.radians(cells.gamma)

    # Volume of cells
    cos2alpha=np.power(np.cos(alpha),2.0)
    cos2beta=np.power(np.cos(beta),2.0)
    cos2gamma=np.power(np.cos(gamma),2.0)
    cosabg=np.cos(alpha)*np.cos(beta)*np.cos(gamma)
    vcell=a*b*c*np.sqrt(1-cos2alpha-cos2beta-cos2gamma+2*cosabg)

    # astar
    astar = b*c*np.sin(alpha)/vcell
    bstar = c*a*np.sin(beta)/vcell
    cstar = a*b*np.sin(gamma)/vcell

    return astar,bstar,cstar

# the second mtz file
mtz2 = gemmi.read_mtz_file(sys.argv[2])
cell2 = mtz2.cell
mtz_data2 = np.array(mtz2, copy=False)
mtz_df2 = pd.DataFrame(data=mtz_data2, columns=mtz2.column_labels())
# (optional) store Miller indices as integers
mtz_df2 = mtz_df2.astype({label: 'int32' for label in 'HKL'})
mtz_df2_new=mtz_df2.rename(columns={"IMEAN": "IMEAN2"})

# merge DataFrames
df = pd.merge(mtz_df1, mtz_df2_new, on=['H', 'K', 'L'])
df_cc=df.corr()

# Scaling 
bfac = 0.0
scale = 1.0

def make_scaled_intensity(df, cells, scale, B):
    df = calc_drelated(df,cells)
    # scale, B 
    logscaled_imean1 = np.log10(df['IMEAN'])
    scaled_log_imean1 = logscaled_imean1 * B  + scale
    plt.plot(logscaled_imean1, scaled_log_imean1)
    plt.show()
    df['IMEAN_scaled'] = np.power(10.0,scaled_log_imean1)

    print("OKAOKAY")
    plt.scatter(df['dstar2'],scaled_log_imean1)
    plt.scatter(df['dstar2'],logscaled_imean1)
    plt.show()

    return df

def calc_drelated(df,cells):
    # calculation of a*, b*, c* from unit cell parameters
    astar,bstar,cstar=abc_stars(cells)
    # dstar vector 
    df['dstar'] = df['H'] * astar + df['K'] * bstar + df['L'] * cstar
    # dstar2
    df['dstar2'] = df['dstar']*df['dstar']
    return df

def do(df,cells):
    # calculation of a*, b*, c* from unit cell parameters
    astar,bstar,cstar=abc_stars(cells)
    # dstar vector 
    df['dstar'] = df['H'] * astar + df['K'] * bstar + df['L'] * cstar
    # dstar2
    df['dstar2'] = df['dstar']*df['dstar']

    # Filtering I > 0.0
    filter_i1 = df['IMEAN'] > 0.0
    filter_i2 = df['IMEAN2'] > 0.0
    df_ = df[filter_i1 & filter_i2]

    lnIMEAN=np.log(df_['IMEAN'])

    # Fitting the curve
    # 最小二乗法でパラメータを決定
    params1, _ = curve_fit(linear_func, df_['dstar2'], lnIMEAN)
    print("Equation1:",params1)

    lnIMEAN2=np.log(df_['IMEAN2'])
    params2, _ = curve_fit(linear_func, df_['dstar2'], lnIMEAN2)
    print("Equation2:",params2)

    # Expected values
    yexp1 = linear_func(df_['dstar2'],params1[0],params1[1])
    yexp2 = linear_func(df_['dstar2'],params2[0],params2[1])

    # Correction overlap 1 to 2 
    # 相対スケールと相対切片を計算
    relative_scale = params2[0] / params1[0]
    relative_intercept = params2[1] - params1[1]
    new_y1 = relative_scale * params1[0]*df_['dstar2'] + params1[1] + relative_intercept

    #plt.scatter(df_['dstar2'],lnIMEAN,s=1,alpha=0.1,color="red")
    # plt.scatter(df_['dstar2'],yexp1,s=1,alpha=0.1,color="green")
    #plt.scatter(df_['dstar2'],lnIMEAN2,s=1,alpha=0.1,color="blue")
    #plt.scatter(df_['dstar2'],yexp2,s=5,alpha=0.5,marker='^',color="orange")
    #plt.scatter(df_['dstar2'],new_y1,s=1,alpha=0.5,marker='o',color="blue")
    #plt.show()

#df = make_scaled_intensity(df,cell1,10.0,10)
do(df,cell1)

# Correlation coefficient of 
cc=df['IMEAN'].corr(df['IMEAN2'])

print(df['IMEAN'],df['IMEAN2'])
print("###################")
print(df)
print("###################")

comment1="%8.3f%8.3f%8.3f(%s)\n"% (cell1.a,cell1.b,cell1.c,sys.argv[1])
comment2="%8.3f%8.3f%8.3f(%s)\n"% (cell2.a,cell2.b,cell2.c,sys.argv[2])

cell_diff = cell1.b-cell2.b
comment3="CC=%8.3f CellDiff=%8.3f"%(cc, cell_diff)

title_s=comment1+comment2+comment3

# plot FP from MTZ as a function of F_meas_au from mmCIF
plt.rc('font', size=14)
plt.figure(figsize=(8, 8))
plt.scatter(x=df['IMEAN'], y=df['IMEAN2'], marker=',', s=1, linewidths=0)
plt.title(title_s)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.savefig("test.png")
#plt.show()
