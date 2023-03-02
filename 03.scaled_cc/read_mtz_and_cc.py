import sys
import gemmi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

# Correlation coefficient of 
cc=df['IMEAN'].corr(df['IMEAN2'])

print(df['IMEAN'],df['IMEAN2'])

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