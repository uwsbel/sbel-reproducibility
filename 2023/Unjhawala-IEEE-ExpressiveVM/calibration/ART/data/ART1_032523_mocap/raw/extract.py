import math
import sys
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd


file = sys.argv[1] # without extension
sub_folder = sys.argv[2]
head = ['x', 'y', 'gtx', 'gty', 'ekfx', 'ekfy', 'heading', 'throttle', 'steering', 'timeStamp']
data = pd.read_csv("./" + sub_folder + "/" +  file + ".csv", header = None, sep = ',')

data.columns = head

# Correct timestamp -> First we drop the first row and then use the second row time and subract all times with that
data = data.iloc[1:]

data['timeStamp'] = data['timeStamp'] - data.loc[1,'timeStamp']

# Correct the position to 0,0
init_loc = [data.loc[1,'gtx'], data.loc[1,'gty']]
init_head = data.loc[1,'heading']

data['gtx'] = data['gtx'] - init_loc[0]
data['gty'] = data['gty'] - init_loc[1]
# data['heading'] =data['heading'] - init_head


print(data['timeStamp'])
inp_path = "/home/unjhawala/projectlets/misc/2022/DataDrivenModSim/BayesianCalibration/ART/inputs/ART1_032523_mocap/" + sub_folder + "/"


inps = pd.DataFrame()

inps['timeStamp'] = data['timeStamp']
inps['steering'] = data['steering']
inps['throttle'] = data['throttle']
inps['braking'] = np.zeros_like(data['steering'])


# extract the time and the throttle and steering to generate the input test file
np.savetxt(inp_path + file + '.txt', inps.values, fmt='%f')
    


# Save x y data in another csv
traj = pd.DataFrame()

traj['timeStamp'] = data['timeStamp']
traj['x'] = data['gtx']
traj['y'] = data['gty']
traj['yaw'] = data['heading']


traj.to_csv("../" + sub_folder + "/traj_" + file + ".csv", sep = ',')


# mpl.plot(traj['x'], traj['y'])
# mpl.plot(traj['timeStamp'], traj['yaw'])

mpl.show()