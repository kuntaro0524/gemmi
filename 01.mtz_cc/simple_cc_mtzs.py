import sys
import gemmi
import numpy as np
import pandas as pd

mtz = gemmi.read_mtz_file(sys.argv[1])
imean=mtz.get_float("IMEAN")

# make DataFrame from MTZ file
mtz_data = np.array(mtz, copy=False)
mtz_df = pd.DataFrame(data=mtz_data, columns=mtz.column_labels())
# (optional) store Miller indices as integers
mtz_df1 = mtz_df.astype({label: 'int32' for label in 'HKL'})

# the second mtz file
mtz2 = gemmi.read_mtz_file(sys.argv[2])
mtz_data2 = np.array(mtz2, copy=False)
mtz_df2 = pd.DataFrame(data=mtz_data2, columns=mtz2.column_labels())
# (optional) store Miller indices as integers
mtz_df2 = mtz_df2.astype({label: 'int32' for label in 'HKL'})
mtz_df2_new=mtz_df2.rename(columns={"IMEAN": "IMEAN2"})

# merge DataFrames
df = pd.merge(mtz_df1, mtz_df2_new, on=['H', 'K', 'L'])

# Correlation coefficient of 
cc=df['IMEAN'].corr(df['IMEAN2'])
print(cc)
