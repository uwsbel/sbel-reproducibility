import sys
import matplotlib.pyplot as mpl
import arviz as az
import rom4
import pandas as pd
import numpy as np

"""
Posterior response for longitudinal dynamics test for ART

Command line inputs
1) flag to specify if the plots need to be saved
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

# Load the nc file
rpm2rad = np.pi / 30.
vehicle = "ART4dof"
filename = "20230609_123136_test0"
idata_acc = az.from_netcdf("../calibration/" + vehicle + "/results/" + filename + ".nc")

# set the number of sampels to be drawn
#############################################################################
n = 100
#############################################################################
idata_acc_n = az.extract(idata_acc, num_samples=100)

# The input and data files
fileName_con = "../calibration/" + vehicle + "/inputs/multi_run_acc/ramp/test0.txt"
fileName_con_st = "../calibration/" + vehicle + "/inputs/multi_run_acc/full_throttle/test1.txt"





# simulation end time 
endTime_st = 9.5009
endTime = 11.009


##################################### Ramp response 100 lines #####################################
#create an empy list that stores all the 100 numpy array of responses
model_post = [None]*n

# fill up our driver data from the file
driverData = rom4.vector_entry()
rom4.driverInput(driverData,fileName_con)

## Loop over each posterior sample
for i in range(n):
    # lets get our vector of doubles which will hold the controls at each time
    controls = rom4.vector_double(4,0)

    veh1_param = rom4.VehParam()

    # Initialize our vehicle state in each iteration 
    veh1_st = rom4.VehStates()


    veh1_param._step = 0.001
    step = veh1_param._step


    # Set the parameters to the posteriors - crucial section
    ################################################################################################################################
    veh1_param._c0 = idata_acc_n['c0'][i].values.item()*veh1_param._c0
    veh1_param._tau0 = idata_acc_n['tau0'][i].values.item()*veh1_param._tau0
    veh1_param._omega0 = idata_acc_n['omega0'][i].values.item()*veh1_param._omega0

    #################################################################################################################################

    # run forest run
    mod = []
    t = 0
    timeStepNo = 0
    while(t<endTime):
        # get the controls for the time step
        rom4.getControls(controls, driverData, t)
        rom4.solve_any(veh1_st, veh1_param, controls)

        #append timestep results
        #append timestep results
        if(timeStepNo % 10 == 0):
            mod.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])


        t += step
        timeStepNo += 1
        
    # save all our models
    model_post[i] = np.array(mod)

################################################### Ramp mean response ##################################
controls = rom4.vector_double(4,0)
veh1_param = rom4.VehParam()
# Initialize our vehicle state in each iteration 
veh1_st = rom4.VehStates()


veh1_param._step = 0.001
step = veh1_param._step

# Set the parameters to the posteriors - crucial section
################################################################################################################################
veh1_param._c0 = np.mean(idata_acc['posterior']['c0'].values)*veh1_param._c0
veh1_param._tau0 = np.mean(idata_acc['posterior']['tau0'].values)*veh1_param._tau0
veh1_param._omega0 = np.mean(idata_acc['posterior']['omega0'].values)*veh1_param._omega0

#################################################################################################################################

# run forest run
mod = []
t = 0
timeStepNo = 0
while(t<endTime):
        # get the controls for the time step
        rom4.getControls(controls, driverData, t)
        rom4.solve_any(veh1_st, veh1_param, controls)

        #append timestep results
        #append timestep results
        if(timeStepNo % 10 == 0):
            mod.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])


        t += step
        timeStepNo += 1
# save all our models
model_post_mean = np.array(mod)


############################################### Step response 100 lines ##################################
#create an empy list that stores all the 100 numpy array of responses
model_post_st = [None]*n

# fill up our driver data from the file
driverData = rom4.vector_entry()
rom4.driverInput(driverData,fileName_con_st)

## Loop over each posterior sample
for i in range(n):
    # lets get our vector of doubles which will hold the controls at each time
    controls = rom4.vector_double(4,0)

    veh1_param = rom4.VehParam()

    # Initialize our vehicle state in each iteration 
    veh1_st = rom4.VehStates()


    veh1_param._step = 0.001
    step = veh1_param._step


    # Set the parameters to the posteriors - crucial section
    ################################################################################################################################
    veh1_param._c0 = idata_acc_n['c0'][i].values.item()*veh1_param._c0
    veh1_param._tau0 = idata_acc_n['tau0'][i].values.item()*veh1_param._tau0
    veh1_param._omega0 = idata_acc_n['omega0'][i].values.item()*veh1_param._omega0

    #################################################################################################################################

    # run forest run
    mod = []
    t = 0
    timeStepNo = 0
    while(t<endTime_st):
        # get the controls for the time step
        rom4.getControls(controls, driverData, t)
        rom4.solve_any(veh1_st, veh1_param, controls)

        #append timestep results
        #append timestep results
        if(timeStepNo % 10 == 0):
            mod.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])


        t += step
        timeStepNo += 1
        
    # save all our models
    model_post_st[i] = np.array(mod)

################################################### Step mean response ###############################

# lets get our vector of doubles which will hold the controls at each time
controls = rom4.vector_double(4,0)

veh1_param = rom4.VehParam()

# Initialize our vehicle state in each iteration 
veh1_st = rom4.VehStates()


veh1_param._step = 0.001
step = veh1_param._step

# Set the parameters to the posteriors - crucial section
################################################################################################################################
veh1_param._c0 = np.mean(idata_acc['posterior']['c0'].values)*veh1_param._c0
print(np.mean(idata_acc['posterior']['c0'].values))
veh1_param._tau0 = np.mean(idata_acc['posterior']['tau0'].values)*veh1_param._tau0
print(np.mean(idata_acc['posterior']['tau0'].values))
veh1_param._omega0 = np.mean(idata_acc['posterior']['omega0'].values)*veh1_param._omega0
print(np.mean(idata_acc['posterior']['omega0'].values))
#################################################################################################################################

# run forest run
mod = []
t = 0
timeStepNo = 0
while(t<endTime_st):
    # get the controls for the time step
    rom4.getControls(controls, driverData, t)
    rom4.solve_any(veh1_st, veh1_param, controls)

    #append timestep results
    #append timestep results
    if(timeStepNo % 10 == 0):
        mod.append([t, veh1_st._x, veh1_st._y, veh1_st._theta, veh1_st._v])


    t += step
    timeStepNo += 1
    

# save all our models
model_post_mean_st = np.array(mod)

# Data

test_nums_ramp = ['0', '1', '2', '3', '4']
test_nums_step = ['1', '2', '3', '5', '8']

cols = ['tab:gray', 'tab:pink', 'tab:green', 'tab:red', 'tab:brown']
fig, axes = mpl.subplots(nrows = 1, ncols = 2, figsize = (10,5), sharey = True)

for i in range(5):
    data = pd.read_csv("../calibration/" + vehicle + "/data/multi_run_acc/ramp/test" + test_nums_ramp[i] +  ".csv", sep=',',header = 'infer')
    data_st = pd.read_csv("../calibration/" + vehicle + "/data/multi_run_acc/full_throttle/test" + test_nums_step[i] + ".csv", sep=',',header = 'infer')
    data_st = data_st[:951]
    data = data[:1101]

    print(f"Ramp Test{test_nums_ramp[i]}")
    print(f"Longitudinal Velocity Error = {np.sqrt(np.sum((model_post_mean[:,4] - data['velo'])**2)/model_post_mean.shape[0])}")

    print(f"Step Test{test_nums_step[i]}")
    print(f"Longitudinal Velocity Error = {np.sqrt(np.sum((model_post_mean_st[:,4] - data_st['velo'])**2)/model_post_mean_st.shape[0])}")

    # plot the data
    axes[0].plot(data['time'],data['velo'],cols[i],label =f'Test {i + 1}',alpha = 0.5)
    axes[1].plot(data_st['time'],data_st['velo'],cols[i],label =f'Test {i + 1}',alpha = 0.5)


### Plots




# plot all the posteriors
### Ramp
for i in range(0,n):
    axes[0].plot(data['time'],model_post[i][:,[4]],'b',alpha = 0.2)

# plot the mean response
axes[0].plot(data['time'],model_post_mean[:,[4]],'y',label = 'Posterior Mean response')


axes[0].set_title("Ramp response")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Longitudinal velocity (m/s)")
axes[0].legend()



### Step
for i in range(0,n):
    axes[1].plot(data_st['time'],model_post_st[i][:,[4]],'b',alpha = 0.2)

# plot the mean response
axes[1].plot(data_st['time'],model_post_mean_st[:,[4]],'y',label = 'Posterior Mean response')

# plot the data

axes[1].set_title("Step response")
axes[1].set_xlabel("Time (s)")
axes[1].legend()


fig.tight_layout()
save = int(sys.argv[1])
if(save):
    mpl.savefig(f"./images/art4dof_train_long_postResp.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art4dof_train_long_postResp", facecolor = 'w', dpi = 600) 

mpl.show()