import sys
import matplotlib.pyplot as mpl
import arviz as az
import rom4
import pandas as pd
import numpy as np

"""
Command line arguments 
1) nc results file name without the extension
2) Test number
3) Experiment based on the normalized steering applied
4) Flag for whether we want to save the file or not
"""

py_2PI = 6.283185307179586476925286766559
def convert_rad(data):
        data[' yaw'] = np.where((data[' yaw'] < -0.01), data[' yaw'] + py_2PI, data[' yaw'])
        return data

def flipXY(data):
        data[' x'] = -data[' x']
        data[' y'] = -data[' y']
        data[' vx'] = -data[' vx']
        data[' vy'] = -data[' vy']
        return data
# coords needs to be a dataframe
def traj_len(coords):
    coords['x_diff'] = coords[' x'].diff()
    coords['y_diff'] = coords[' y'].diff()

    coords['distance'] = (coords['x_diff']**2 + coords['y_diff']**2)**0.5

    total_distance = coords['distance'].sum()
    return total_distance



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


rpm2rad = np.pi / 30.
vehicle = "ART4dof"

filename = "20230609_125452"
idata_acc = az.from_netcdf("../calibration/" + vehicle + "/results/" + filename + ".nc")


# set the number of sampels to be drawn
#############################################################################
n = 100
#############################################################################
idata_acc_n = az.extract(idata_acc, num_samples=100)


full_file = "test" + sys.argv[1]
test_type = "1s-rampt-10s"
ART_path = "/ART1_021923/"


fileName_con = "../calibration/" + vehicle+ "/inputs" + ART_path + test_type + "/" + full_file + ".txt"


# Get the simulation end time from the data
data = pd.read_csv( "../calibration/" + vehicle + "/data" + ART_path +  test_type  + "/" + full_file +  ".csv", sep = ",", header = "infer")
# end time of the simulation
endTime = data.iloc[-1,0]

############################################################## 100 Simulation ######################################################################
model_post = [None]*n # Stores a list of the dataframes for each simulation


for i in range (n):
    # lets get a vector of entries going 
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


    ################# Get the parameters from the  nc file ################################################
    veh1_param._steerMap = idata_acc_n['beta'][i].values.item()

###############################################################################################################


    # Create an empty dataframe with column names
    results = []
    results.append([0, 0, 0, 0, 0])
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
            results.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])
    

    model_post[i] = np.array(results)


#################################################################### Mean posterior response ################################################

# lets get a vector of entries going 
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

################# Get the parameters from the  nc file ################################################
veh1_param._steerMap = np.mean(idata_acc['posterior']['beta'].values)
print(np.mean(idata_acc['posterior']['beta'].values))

###############################################################################################################


# Create an empty dataframe with column names
results = []
results.append([0, 0, 0, 0, 0])
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
            results.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])
    
model_post_mean = np.array(results)




############################################################## Calculating the mean RMSE ##################################################################
# First get all the correct test files

nonCali_tests =["0", "2", "4", "5"]

data = pd.read_csv( "../calibration/" + vehicle + "/data" + ART_path +  test_type  + "/" + f"test{nonCali_tests[0]}" +  ".csv", sep = ",", header = "infer") 
data4 = convert_rad(data)
data4 = flipXY(data)


data2 = pd.read_csv( "../calibration/" + vehicle + "/data" + ART_path +  test_type  + "/" + f"test{nonCali_tests[1]}" +  ".csv", sep = ",", header = "infer")
data2 = convert_rad(data2)
data2 = flipXY(data2)


data3 = pd.read_csv( "../calibration/" + vehicle + "/data" + ART_path +  test_type  + "/" + f"test{nonCali_tests[2]}" +  ".csv", sep = ",", header = "infer")
data3 = convert_rad(data3)
data3 = flipXY(data3)


data4 = pd.read_csv( "../calibration/" + vehicle + "/data" + ART_path +  test_type  + "/" + f"test{nonCali_tests[3]}" +  ".csv", sep = ",", header = "infer")
data4 = convert_rad(data4)
data4 = flipXY(data4)

print(f"Test type = {test_type}, full file = {full_file}")

points_data = data.iloc[:,1:3]
points_response = model_post_mean[:,1:3]
lv1 = np.sum(np.sqrt((points_data.iloc[:1000,1] - points_response[:1000,1])**2 + (points_data.iloc[:1000,0] - points_response[:1000,0])**2))/points_data.shape[0]


points_data2 = data2.iloc[:,1:3]
points_response2 = model_post_mean[:,1:3]
lv2 = np.sum(np.sqrt((points_data2.iloc[:1000,1] - points_response2[:1000,1])**2 + (points_data2.iloc[:1000,0] - points_response2[:1000,0])**2))/points_data2.shape[0]


points_data3 = data3.iloc[:,1:3]
points_response3 = model_post_mean[:,1:3]
lv3 = np.sum(np.sqrt((points_data3.iloc[:1000,1] - points_response3[:1000,1])**2 + (points_data3.iloc[:1000,0] - points_response3[:1000,0])**2))/points_data3.shape[0]


points_data4 = data4.iloc[:,1:3]
points_response4 = model_post_mean[:,1:3]
lv4 = np.sum(np.sqrt((points_data4.iloc[:1000,1] - points_response4[:1000,1])**2 + (points_data4.iloc[:1000,0] - points_response4[:1000,0])**2))/points_data4.shape[0]


############################################################## Calculating trajectory length of data ################################## 
total_distance = traj_len(points_data)
total_distance2 = traj_len(points_data2)
total_distance3 = traj_len(points_data3)
total_distance4 = traj_len(points_data4)

################################################################## Plotting ########################################################

import warnings
warnings.filterwarnings("ignore")
fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (6,3))

# Trajectory
axes.plot(data[' x'],data[' y'],'r', label ='Test 1')
axes.plot(data2[' x'],data2[' y'], label = 'Test 2')
axes.plot(data3[' x'],data3[' y'], label = 'Test 3')
axes.plot(data4[' x'],data4[' y'], label = 'Test 4')


for i in range(0,n):
    axes.plot(model_post[i][:,[1]],model_post[i][:,[2]],'b',alpha = 0.2)
axes.plot(model_post_mean[:,[1]],model_post_mean[:,[2]],'y',label = 'Posterior Mean response')

axes.set_xlabel("X (m)")
axes.set_ylabel("Y (m)")
axes.legend()

# cycle markers
markers = ['o', 's', 'D', '^', '*', 'h', 'x', 'p', '+', 'h' , '>']
pts = np.arange(0,1001,100)

for i,point in enumerate(pts):
    axes.scatter(data.loc[point,' x'],data.loc[point,' y'],marker = markers[i],s = 50, c = 'tab:red')
    axes.scatter(data2.loc[point,' x'],data2.loc[point,' y'],marker = markers[i],s = 50, c = 'tab:blue')
    axes.scatter(data3.loc[point,' x'],data3.loc[point,' y'],marker = markers[i],s = 50, c = 'tab:orange')
    axes.scatter(data4.loc[point,' x'],data4.loc[point,' y'],marker = markers[i],s = 50, c = 'tab:green')
    axes.scatter(model_post_mean[point,[1]],model_post_mean[point,[2]],marker = markers[i], s = 50, c = 'tab:olive')

fig.tight_layout()
save = int(sys.argv[2])



################################################### Plotting the growth of the mean ED with the total distance ###########################

fig2, axes2 = mpl.subplots(nrows = 1, ncols = 1, figsize = (6,3))
fig3, axes3 = mpl.subplots(nrows = 1, ncols = 1, figsize = (6,3))

resps = [points_response, points_response2, points_response3, points_response4]
datas = [points_data, points_data2, points_data3, points_data4]



for whi,r in enumerate(resps): # for each data
    mEDs = []
    dists = []
    mEDs_by_dists = []
    for i in range(0,1000,10): # every 10 points
        # Compute the mean ED and the distance uptill the i points
        dists.append(traj_len(datas[whi][:i]))
        mEDs.append(np.sum(np.sqrt((datas[whi].iloc[:i,1] - r[:i,1])**2 + (datas[whi].iloc[:i,0] - r[:i,0])**2))/datas[whi][:i].shape[0])
        mEDs_by_dists.append(mEDs[int(i/10)]/dists[int(i/10)])

    axes2.plot(dists, mEDs,label = f'Test {whi}')
    axes3.plot(dists, mEDs_by_dists, label = f'Test {whi}')


axes2.set_xlabel("Distance traveled [m]")
axes2.set_ylabel("Mean Euclidean distance [m]")

axes3.set_xlabel("Distance traveled [m]")
axes3.set_ylabel(r"Mean ED / $\mathcal{l}$ [-]")

axes2.legend()
axes3.legend()

fig2.tight_layout()
fig3.tight_layout()

if(save):
    fig.savefig(f'./images/art4dof_lat_{test_type}.eps', format='eps', dpi=3000)
    fig.savefig(f'./images/art4dof_lat_{test_type}.png', facecolor = 'w', dpi = 600)
    fig3.savefig(f'./images/scale_free4dof_{test_type}.eps', format='eps', dpi=3000)
    fig3.savefig(f'./images/scale_free4dof_{test_type}.png', facecolor = 'w', dpi = 600)
mpl.show()


print(f"Test 1 = {lv1}")
print(f"Test 2 = {lv2}")
print(f"Test 3 = {lv3}")
print(f"Test 4 = {lv4}")
print(f"Mean Eucledian distance only till stop for test type {test_type} = {(lv1 + lv2 + lv3 + lv4)/4}")

meanED_l = [lv1 / total_distance, lv2 / total_distance2, lv3 / total_distance3, lv4 / total_distance4]
print(f'Test 1 Total distance : {total_distance} and the Mean ED by the Total distance is : {meanED_l[0]}')
print(f'Test 2 Total distance : {total_distance2} and the Mean ED by the Total distance is : {meanED_l[1]}')
print(f'Test 3 Total distance : {total_distance3} and the Mean ED by the Total distance is : {meanED_l[2]}')
print(f'Test 4 Total distance : {total_distance4} and the Mean ED by the Total distance is : {meanED_l[3]}')



print(f'Mean ED / l across tests {sum(meanED_l)/ len(meanED_l)}')

