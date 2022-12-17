import sys
import gemmi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mtz = gemmi.read_mtz_file(sys.argv[1])

print(mtz.cell)
print(mtz.spacegroup)
print(mtz.title)
print(mtz.history)
print(mtz.nreflections)
print(mtz.sort_order)
print(mtz.min_1_d2)
print(mtz.max_1_d2)

imean=mtz.get_float("IMEAN")

# make DataFrame from MTZ file
mtz = gemmi.read_mtz_file(sys.argv[1])
mtz_data = np.array(mtz, copy=False)
mtz_df = pd.DataFrame(data=mtz_data, columns=mtz.column_labels())
# (optional) store Miller indices as integers
mtz_df1 = mtz_df.astype({label: 'int32' for label in 'HKL'})
print(mtz.cell)

#print(mtz_df1.describe())

# the second mtz file
mtz2 = gemmi.read_mtz_file(sys.argv[2])
print(mtz2.cell)
mtz_data2 = np.array(mtz2, copy=False)
mtz_df2 = pd.DataFrame(data=mtz_data2, columns=mtz2.column_labels())
# (optional) store Miller indices as integers
mtz_df2 = mtz_df2.astype({label: 'int32' for label in 'HKL'})
mtz_df2_new=mtz_df2.rename(columns={"IMEAN": "IMEAN2"})
print(mtz_df2.columns)
#print(mtz_df2_new.describe())

# merge DataFrames
df = pd.merge(mtz_df1, mtz_df2_new, on=['H', 'K', 'L'])

print(df.describe())

# plot FP from MTZ as a function of F_meas_au from mmCIF
plt.rc('font', size=8)
plt.figure(figsize=(2, 2))
plt.scatter(x=df['IMEAN'], y=df['IMEAN2'],
               marker=',', s=1, linewidths=0)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.show()
