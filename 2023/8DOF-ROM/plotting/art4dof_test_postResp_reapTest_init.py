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
import rom4
import logging
from cycler import cycler
"""
Run as python3 art_test_postResp_reapTest.py 1(Test number from which we take the input) 2(folder number r1 or r2) 0(flag to save)
"""
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

color_list = ['r', 'g', 'b', 'm', 'c', 'k', 'tab:brown']

mpl.rc('axes', prop_cycle=(cycler('color', color_list)))

# In case we need to flip the switch....
rpm2rad = np.pi / 30
py_2PI = 6.283185307179586476925286766559
def convert_rad(data):
        data[' yaw'] = np.where((data[' yaw'] < -0.01), data[' yaw'] + py_2PI, data[' yaw'])
        return data

def flipXY(data):
        data['x'] = -data['x']
        data['y'] = -data['y']
        # data[' vx'] = -data[' vx']
        # data[' vy'] = -data[' vy']
        return data
def traj_len(coords):
    coords['x_diff'] = coords['x'].diff()
    coords['y_diff'] = coords['y'].diff()

    coords['distance'] = (coords['x_diff']**2 + coords['y_diff']**2)**0.5

    total_distance = coords['distance'].sum()
    return total_distance

######################### Extract the posetrior ##################################
fileNameTh = "20230609_134138_test0"
filename1 = "20230609_134715" 


idataTh = az.from_netcdf("../calibration/ART4dof/results/" + fileNameTh + ".nc")
idata1 = az.from_netcdf("../calibration/ART4dof/results/" + filename1 + ".nc")





sub_folder = "r" + sys.argv[1] # No r
path = "../calibration/ART4dof/data/ART1_040623_mocap_reapTest/" +sub_folder + "/"
input_path = "../calibration/ART4dof/inputs/ART1_040623_mocap_reapTest/" + sub_folder + "/" 
per_test = 20
################################ Sample 100 parameters from it ##############################
n = per_test*7
idataTh_n = az.extract(idataTh, num_samples=n)
idata1_n = az.extract(idata1, num_samples=n)

##################################### Ramp response 20*10 lines #####################################
#create an empy list that stores all the 100 numpy array of responses
model_post = [None]*n

# Loop through all the data files 
actual_test = 0 # Needed because some tests are not good
for test in range(1,10):
    if((sub_folder == "r1") & (test == 7)):
        continue
    if((sub_folder == "r1") & (test == 3)):
        continue
    if((sub_folder == "r2") & (test == 8)):
        continue
    if((sub_folder == "r2") & (test == 3)):
        continue

    input_file = "r" + str(test) # No r
    data_file = "traj_" + sub_folder + "_" + input_file + ".csv"
    fileName_con =  input_path + "/" + sub_folder+ "_" +  input_file  + ".txt"

    # The ART data
    data_ART = pd.read_csv(path + data_file, sep = ",", header = "infer")
    # print(data_ART.columns)


    endTime = data_ART.iloc[-1,1]

    print(f"Subfolder : {sub_folder}, test file {data_file}")


    driverData = rom4.vector_entry()
    rom4.driverInput(driverData,fileName_con)

    ## Loop over each posterior sample
    
    for i in range(per_test):
        # lets get our vector of doubles which will hold the controls at each time
        controls = rom4.vector_double(4,0)

        veh1_param = rom4.VehParam()

        # Initialize our vehicle state in each iteration 
        veh1_st = rom4.VehStates()


        veh1_param._step = 0.001
        step = veh1_param._step


        # Set the parameters to the posteriors - crucial section
        ################################################################################################################################        

        #Steering
        veh1_param._steerMap = idata1_n['beta'][i + actual_test*per_test].values.item()
        
        # Throttle
        veh1_param._c0 = idataTh_n['c0'][i + actual_test*per_test].values.item()*veh1_param._c0
        veh1_param._tau0 = idataTh_n['tau0'][i + actual_test*per_test].values.item()*veh1_param._tau0
        veh1_param._omega0 = idataTh_n['omega0'][i + actual_test*per_test].values.item()*veh1_param._omega0
        #################################################################################################################################

        # run forest run
        mod = []
        veh1_st._theta = data_ART.loc[0,'yaw'] + 3.14159
        mod.append([0, 0, 0, veh1_st._theta, 0])
        t = 0
        timeStepNo = 0
        while(t < (endTime - step/10)):


            # get the controls for the time step
            rom4.getControls(controls, driverData, t)
            rom4.solve_any(veh1_st, veh1_param, controls)

            t += step
            timeStepNo += 1
            #append timestep results
            if(timeStepNo % 10 == 0):
                mod.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])
                    
        
        
        model_post[i + actual_test*per_test] = np.array(mod)
    actual_test = actual_test + 1
############################################################ Mean response #############################################################
# For each of our initial conditions, we will have a posterior mean response. We will finally take another mean of this to form 
# a posterior mean response expectation
actual_test = 0 # Needed because some tests are not good
test = 0

if(sub_folder == "r1"):
    post_mean = [None]*7
else:
    post_mean = [None]*7

for test in range(1,10):
    if((sub_folder == "r1") & (test == 7)):
        continue
    if((sub_folder == "r1") & (test == 3)):
        continue
    if((sub_folder == "r2") & (test == 8)):
        continue
    if((sub_folder == "r2") & (test == 3)):
        continue
    input_file = "r" + str(test) # No r
    data_file = "traj_" + sub_folder + "_" + input_file + ".csv"
    fileName_con =  input_path + "/" + sub_folder+ "_" +  input_file  + ".txt"

    # The ART data
    data_ART = pd.read_csv(path + data_file, sep = ",", header = "infer")
    # print(data_ART.columns)


    endTime = data_ART.iloc[-1,1]

    driverData = rom4.vector_entry()

    # lets fill this up from4 our data file
    rom4.driverInput(driverData,fileName_con)

    # lets get our vector of doubles which will hold the controls at each time
    controls = rom4.vector_double(4,0)

    veh1_param = rom4.VehParam()

    # Initialize our vehicle state in each iteration 
    veh1_st = rom4.VehStates()


    veh1_param._step = 0.001
    step = veh1_param._step


    # print(endTime)

    # veh1_st._psi = data_ART.loc[0,'yaw']
    # run forest run
    mod_mean = []
    # Add the inital conditions
    veh1_st._psi = data_ART.loc[0,'yaw'] + 3.14159
    mod.append([0, 0, 0, veh1_st._theta, 0])
    t = 0
    timeStepNo = 0

    #  Set the tire parameters based on the nc files, using mean response
    veh1_param._steerMap = np.mean(idata1['posterior']['beta'].values)
            

    #### Throttle
    veh1_param._c0 = np.mean(idataTh['posterior']['c0'].values)*veh1_param._c0
    veh1_param._tau0 = np.mean(idataTh['posterior']['tau0'].values)*veh1_param._tau0
    veh1_param._omega0 = np.mean(idataTh['posterior']['omega0'].values)*veh1_param._omega0



    while(t < (endTime + step)):
        # get the controls for the time step
        rom4.getControls(controls, driverData, t)
        rom4.solve_any(veh1_st, veh1_param, controls)

        t += step
        timeStepNo += 1
        #append timestep results
        if(timeStepNo % 10 == 0):
            mod_mean.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])


    post_mean[actual_test] = np.array(mod_mean)
    actual_test = actual_test + 1


# Take the mean of all the arrays in post_mean to get the expectation of the posterior mean -> Cut based on the smallest test
shapes = []
for pm in post_mean:
    shapes.append(pm.shape[0])

cutter = min(shapes)
post_mean = [pm[:cutter,:] for pm in post_mean]
print(post_mean[0].shape)
post_mean = np.array(post_mean)
# post_mean = post_mean[:,:cutter,:]
exp_post_mean = post_mean.mean(axis = 0)

# # Extract the data 
data = pd.read_csv(path + data_file, header = 'infer', sep = ',')

## To remove some stupid printing
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True
############################################################

fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (8,6))

for i in range(0,n):
    axes.plot(model_post[i][:,[1]],model_post[i][:,[2]],'b',alpha = 0.1)


axes.plot(exp_post_mean[:,1],exp_post_mean[:,2],'y',label = '4dof expectation of mean-posterior Response')


axes.set_ylim([-data.loc[data['x'].shape[0]-1,'x'],data.loc[data['x'].shape[0]-1,'x']])



axes.set_xlabel('X (m)')
axes.set_ylabel('Y (m)')



# Error analysis
pts2 = np.arange(0,1201,12)

all_test_mean = []
all_test_mean_dist = []

points_data = data.iloc[:,2:4]
points_response = exp_post_mean[:,1:3]
lv1 = []
for i, point in enumerate(pts2):
    try:
        lv1.append(np.sum(np.sqrt((points_data.iloc[int(point/10),1] - points_response[point,1])**2 + (points_data.iloc[int(point/10),0] - points_response[point,0])**2)))
    except:
        lv1.append(np.sum(np.sqrt((points_data.iloc[-1,1] - points_response[-1,1])**2 + (points_data.iloc[-1,0] - points_response[-1,0])**2)))
all_test_mean.append(sum(lv1)/len(lv1))
print(f"Mean is {sum(lv1)/len(lv1)}")
total_distance = traj_len(data)
all_test_mean_dist.append(sum(lv1)/(len(lv1)*total_distance))
print(f"Mean ED / Total distance = {sum(lv1)/(len(lv1)*total_distance)}\n")
print(exp_post_mean.shape)

# Plotting all the other tests of the same experiment
actual_test = 1
all_test_mean = []
all_test_mean_dist = []
for test in range(1,10):
    if(str(test) == input_file): # Dont plot the already plotted data
        continue
    if((sub_folder == "r1") & (test == 7)):
        continue
    if((sub_folder == "r1") & (test == 3)):
        continue
    if((sub_folder == "r2") & (test == 8)):
        continue
    if((sub_folder == "r2") & (test == 3)):
        continue
    data_file = "traj_" + sub_folder + "_r" + str(test) + ".csv"
    data = pd.read_csv(path + data_file, header = 'infer', sep = ',')
    axes.plot(data['x'], data['y'], label = f'Test {actual_test}')
    points_data = data.iloc[:,2:4]
    lv1 = []
    for i, point in enumerate(pts2):
        try:
            lv1.append(np.sum(np.sqrt((points_data.iloc[int(point/10),1] - points_response[point,1])**2 + (points_data.iloc[int(point/10),0] - points_response[point,0])**2)))
        except:
            lv1.append(np.sum(np.sqrt((points_data.iloc[-1,1] - points_response[-1,1])**2 + (points_data.iloc[-1,0] - points_response[-1,0])**2)))
    print(f"Subfolder : {sub_folder}, test file {data_file}, Actual test {actual_test}")
    all_test_mean.append(sum(lv1)/len(lv1))
    print(f"Mean is {sum(lv1)/len(lv1)}")
    total_distance = traj_len(data)
    all_test_mean_dist.append(sum(lv1)/(len(lv1)*total_distance))
    print(f"Mean ED / Total distance = {sum(lv1)/(len(lv1)*total_distance)}\n")
    print(exp_post_mean.shape)

    markers = ['o', 's', 'D', '^', '*', 'h', 'x', 'p', '+', 'h' , '>']
    pts = np.arange(0,cutter + 51,120)

    for i,point in enumerate(pts):
        try:
            axes.scatter(data.loc[int(point/10),'x'],data.loc[int(point/10),'y'],marker = markers[i],s = 50, c = color_list[actual_test-1])

            if(actual_test == 1):
                axes.scatter(exp_post_mean[point,[1]],exp_post_mean[point,[2]],marker = markers[i], s = 50, c = 'tab:olive')
        except:

            axes.scatter(data.loc[data['x'].shape[0]-1,'x'],data.loc[data['x'].shape[0]-1,'y'],marker = markers[i],s = 50, c = color_list[actual_test-1])

            if(actual_test == 1):
                axes.scatter(exp_post_mean[-1,[1]],exp_post_mean[-1,[2]],marker = markers[i], s = 50, c = 'tab:olive')
    actual_test = actual_test + 1

print(f"Across all test mean for Subfolder {sub_folder} : {sum(all_test_mean)/ len(all_test_mean)}\n Across all test mean / dist : {sum(all_test_mean_dist)/ len(all_test_mean_dist)}")

axes.legend()
fig.tight_layout()

save = int(sys.argv[2])
if(save):
    mpl.savefig("./images/art4dof_test_perTest_" + sub_folder + ".eps",format='eps', dpi=3000)
    mpl.savefig("./images/art4dof_test_perTest_" + sub_folder +  ".png", dpi=600)

mpl.show()
