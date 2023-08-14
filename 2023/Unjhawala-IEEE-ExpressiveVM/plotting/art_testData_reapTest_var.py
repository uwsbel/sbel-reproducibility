import sys
import matplotlib.pyplot as mpl
import scipy as sp
import aesara
import aesara.tensor as tt
import pymc as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import pickle
import time
import random
# import our reduced order models
import rom



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


# sub_folder = "r" + sys.argv[1]
sub_folders = ["r1", "r2"]



fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (8,4))

for sub_folder in sub_folders:
    path = "../calibration/ART/data/ART1_040623_mocap_reapTest/" +sub_folder + "/"
    input_path = "../calibration/ART/inputs/ART1_040623_mocap_reapTest/" + sub_folder + "/" 
    for test in range(1,10):
        if((sub_folder == "r1") & (test == 7)):
            continue
        if((sub_folder == "r2") & (test == 8)):
            continue
        if((sub_folder == "r2") & (test == 3)):
            continue
        data_file = "traj_" +sub_folder + "_r" + str(test) + ".csv"
        fileName_con =  input_path + "/" + sub_folder + "_r" + str(test)  + ".txt"
        data_ART1 = pd.read_csv(path + data_file, sep = ",", header = "infer")

        if(sub_folder == "r1"):
            axes.plot(data_ART1['x'], data_ART1['y'], "--", label = f'Experiment {sub_folder[1]}, Test {test}')
        else:
            axes.plot(data_ART1['x'], data_ART1['y'], label = f'Experiment {sub_folder[1]}, Test {test}')


        # cycle markers
        # markers = ['o', 's', 'D', '^', '*', 'h', 'x', 'p', '+', 'h' , '>']
        # pts = np.arange(0,1201,120)

        # for i,point in enumerate(pts):
        #     try:
        #         axes.scatter(data_ART1.loc[int(point/10),'x'],data_ART1.loc[int(point/10),'y'],marker = markers[i],s = 50, c = 'tab:red')

        #     except:

        #         axes.scatter(data_ART1.loc[data_ART1['x'].shape[0]-1,'x'],data_ART1.loc[data_ART1['x'].shape[0]-1,'y'],marker = markers[i],s = 50, c = 'tab:red')


axes.legend()

axes.set_xlabel('X (m)')
axes.set_ylabel('Y (m)')
mpl.show()