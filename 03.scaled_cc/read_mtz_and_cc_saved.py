import sys
import gemmi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

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

def scale_intensity(h,k,l,cells,i_array1,i_array2):
    # calculation of a*, b*, c* from unit cell parameters
    astar,bstar,cstar=abc_stars(cells)
    # dstar vector 
    dstar = h * astar + k * bstar + l * cstar
    # dstar2
    dstar2 = dstar*dstar
    # Filtering I > 0.0

    # Log intensity
    logI1 = np.log10(i_array1)
    logI2 = np.log10(i_array2)

    print(len(dstar2), len(logI1), len(logI2))

    # Linear regression
    regressor = LinearRegression()
    xdata = np.array(logI1).reshape(-1,1)
    regressor.fit(xdata,logI2)
    scale = regressor.coef_[0]
    offset = regressor.intercept_
    logI1_second = scale*df['logI1'] + offset

    plt.scatter(dstar2,logI1,s=0.1,alpha=0.5)
    plt.scatter(dstar2,logI2,s=0.1,alpha=0.5)
    plt.scatter(dstar2,logI1_second,s=0.1,alpha=0.5)
    plt.show()
    
scale_intensity(df['H'],df['K'],df['L'],cell1,df['IMEAN'], df['IMEAN2'])

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