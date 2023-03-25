import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import os
import sys
import numpy as np
import arviz.labels as azl
import pandas as pd
from scipy.stats import wasserstein_distance
import seaborn as sns

mpl.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Palatino', 'serif'],
    # "font.serif" : ["Computer Modern Serif"],
})

_SQRT2 = np.sqrt(2)

def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2
#### Load all our posteriors which we want to compare

# files = [
#         "20221203_132725_test0"
#         ,"20221203_175118_test1"
#         ,"20221203_220904_test2"
#         ,"20221204_003208_test3"
#         ,"20221204_024242_test4"
# ]


files = [
            "20221213_165731_test0",
            "20221213_205114_test1",
            "20221214_005301_test2",
            "20221214_041640_test3",
            "20221214_080100_test4"
        ]
# columns of data frame which stores the earth mover distance
cols = ['$\\bf{Test\:1}$', '$\\bf{Test\:2}$', '$\\bf{Test\:3}$' , '$\\bf{Test\:4}$', '$\\bf{Test\:5}$']

file_to_cols = dict(zip(files, cols))
posterior_labels = ["p0tor", "p1tor", "p2tor", "p3tor", "p0loss", "p1loss", "p2loss", "sigmaLOV"]
label = "p3tor"

df = pd.DataFrame(0, index = cols, columns = cols)
for file1 in files:
    idata1 = az.from_netcdf('../calibration/ART/results/' + file1 + ".nc")

    print(file1)
    
    for file2 in files:
        idata2 = az.from_netcdf('../calibration/ART/results/' + file2 + ".nc")

        # For each of our 1-D posteriors , we will compare the earth movers distance
        # for label in posterior_labels:
        dist1 = np.array(idata1['posterior'][label].values).flatten()
        dist2 = np.array(idata2['posterior'][label].values).flatten()

        max_among_both = np.maximum(np.max(dist1,axis=0), np.max(dist2,axis=0))
        # Normalize the distributions with the max
        dist1 = dist1/max_among_both
        dist2 = dist2/max_among_both

        df.loc[file_to_cols[file1],file_to_cols[file2]] = wasserstein_distance(dist1,dist2)
        # dist1 /= dist1.sum()
        # dist2 /= dist2.sum()
        # print(hellinger(dist1, dist2))
    
    # print(df)
    # df.to_csv("./results/earth_movers2/" + file1 + ".csv")


fig = mpl.figure()
ax = fig.add_subplot(111)
ax.axis('off')
vals = np.around(df.values, 2)
normal = (df - df.min()) / (df.max() - df.min())
the_table=ax.table(cellText=vals, rowLabels=df.index, colLabels=df.columns, 
                   loc='center', cellColours=mpl.cm.summer(normal), fontsize = 20, cellLoc = 'center',
                   colWidths=[0.215 for x in cols])

# fig.savefig("table.png")

save = int(sys.argv[1])

if(save):
    mpl.savefig(f"./images/art_long_wasDist.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art_long_wasDist.png", facecolor = 'w', dpi = 600)
print(df)
mpl.show()





