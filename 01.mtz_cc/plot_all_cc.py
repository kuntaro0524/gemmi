import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(sys.argv[1])

for (hue),proc_group  in df.groupby("hue"):
	plt.clf()
	sns.scatterplot(x='diff', y='cc',  data=proc_group,s=3,alpha=0.1, palette="muted")
	plt.xlim([-4,4])
	plt.savefig("%05d.jpg" % hue)
plt.show()
