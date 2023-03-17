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

def easy_plot(xa,ya,mabiki=1000):
    plt.scatter(xa,ya,s=1,alpha=0.1)

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

    df_['lnIMEAN1']=np.log(df_['IMEAN'])
    df_['lnIMEAN2']=np.log(df_['IMEAN2'])

    # Fitting the curve
    # 最小二乗法でY1とY2をフィットする
    params1, _ = curve_fit(linear_func, df_['lnIMEAN2'], df_['lnIMEAN1'])
    print("PARAMS:",params1)
    df_['lnIMEAN2_'] = linear_func(df_['lnIMEAN2'],params1[0],params1[1])

    # Selection
    print(len(df_))
    sel = df_.index % 5000 ==0
    dfplt=df_[sel]
    print(len(dfplt))
    plt.scatter(dfplt['dstar2'],dfplt['lnIMEAN1'],s=10,alpha=1.0,label="imean1",color="red")
    plt.scatter(dfplt['dstar2'],dfplt['lnIMEAN2_'],s=10,alpha=1.0,label="imean2_",color="blue")
    plt.scatter(dfplt['dstar2'],dfplt['lnIMEAN2'],s=10,alpha=1.0,label="imean2",color="orange")
    plt.legend()

    sum1=sum(np.power(dfplt['lnIMEAN1']-dfplt['lnIMEAN2'],2.0))
    sum2=sum(np.power(dfplt['lnIMEAN1']-dfplt['lnIMEAN2_'],2.0))
    print(sum1,sum2)
    plt.show()

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
