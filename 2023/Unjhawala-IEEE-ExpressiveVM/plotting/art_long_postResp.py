import sys
import matplotlib.pyplot as mpl
import arviz as az
import rom
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
vehicle = "ART"
filename = "20221214_080100_test4"
idata_acc = az.from_netcdf("../calibration/" + vehicle + "/results/" + filename + ".nc")

# set the number of sampels to be drawn
#############################################################################
n = 100
#############################################################################
idata_acc_n = az.extract(idata_acc, num_samples=100)

# The input and data files
fileName_con = "../calibration/" + vehicle + "/inputs/multi_run_acc/ramp/test4.txt"
fileName_con_st = "../calibration/" + vehicle + "/inputs/multi_run_acc/full_throttle/test8.txt"

# json files
fileName_veh = "../calibration/" + vehicle + "/jsons/dART_play.json"
fileName_tire = "../calibration/" + vehicle + "/jsons/dARTTM_play.json"



# simulation end time 
endTime_st = 9.5009
endTime = 11.009


##################################### Ramp response 100 lines #####################################
#create an empy list that stores all the 100 numpy array of responses
model_post = [None]*n

# fill up our driver data from the file
driverData = rom.vector_entry()
rom.driverInput(driverData,fileName_con)

## Loop over each posterior sample
for i in range(n):
    # lets get our vector of doubles which will hold the controls at each time
    controls = rom.vector_double(4,0)

    veh1_param = rom.VehicleParam()
    rom.setVehParamsJSON(veh1_param,fileName_veh)
    tire_param = rom.TMeasyParam()
    rom.setTireParamsJSON(tire_param,fileName_tire)

    # Initialize our vehicle state in each iteration 
    veh1_st = rom.VehicleState()
    rom.vehInit(veh1_st,veh1_param)

    tirelf_st = rom.TMeasyState()
    tirerf_st = rom.TMeasyState()
    tirelr_st = rom.TMeasyState()
    tirerr_st = rom.TMeasyState()
    rom.tireInit(tire_param)


    veh1_param._step = 0.001
    tire_param._step = 0.001
    step = veh1_param._step

    # Set the parameters to the posteriors - crucial section
    ################################################################################################################################

    veh1_param._powertrainMap[0]._y = idata_acc_n['p0tor'][i].values.item()
    veh1_param._powertrainMap[1]._y = idata_acc_n['p1tor'][i].values.item()
    veh1_param._powertrainMap[2]._y = idata_acc_n['p2tor'][i].values.item()
    veh1_param._powertrainMap[3]._y = idata_acc_n['p3tor'][i].values.item()
    

    veh1_param._lossesMap[0]._y = -idata_acc_n['p0loss'][i].values.item()
    veh1_param._lossesMap[1]._y = -idata_acc_n['p1loss'][i].values.item()
    veh1_param._lossesMap[2]._y = -idata_acc_n['p2loss'][i].values.item()

    #################################################################################################################################

    # run forest run
    mod = []
    t = 0
    timeStepNo = 0
    while(t<endTime):
        # get the controls for the time step
        rom.getControls(controls, driverData, t)

        #transfrom vehicle velocities to tire velocites
        rom.vehToTireTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

        # advanvce our 4 tires with this transformed velocity
        rom.tireAdv(tirelf_st, tire_param, veh1_st, veh1_param, controls)
        rom.tireAdv(tirerf_st, tire_param, veh1_st, veh1_param, controls)

        # rear wheel does not steer, so need to give it modified controls
        mod_controls = [controls[0],0,controls[2],controls[3]]

        rom.tireAdv(tirelr_st, tire_param, veh1_st, veh1_param, mod_controls)
        rom.tireAdv(tirerr_st, tire_param, veh1_st, veh1_param, mod_controls)

        rom.evalPowertrain(veh1_st, tirelf_st, tirerf_st, tirelr_st, tirerr_st, veh1_param, tire_param, controls)

        # transfrom tire forces into vehicle coordinate frame
        rom.tireToVehTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

        # copy useful stuff needed for the vehicle to advance
        fx = [tirelf_st._fx,tirerf_st._fx,tirelr_st._fx,tirerr_st._fx]
        fy = [tirelf_st._fy,tirerf_st._fy,tirelr_st._fy,tirerr_st._fy]
        huf = tirelf_st._rStat
        hur = tirerr_st._rStat

        # look at the vehicle go
        rom.vehAdv(veh1_st,veh1_param,fx,fy,huf,hur)


        #append timestep results
        if(timeStepNo % 10 == 0):
            mod.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                            veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                            tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])

        t += step
        timeStepNo += 1
        
    # save all our models
    model_post[i] = np.array(mod)

################################################### Ramp mean response ##################################
controls = rom.vector_double(4,0)

veh1_param = rom.VehicleParam()
rom.setVehParamsJSON(veh1_param,fileName_veh)
tire_param = rom.TMeasyParam()
rom.setTireParamsJSON(tire_param,fileName_tire)

# Initialize our vehicle state in each iteration 
veh1_st = rom.VehicleState()
rom.vehInit(veh1_st,veh1_param)

tirelf_st = rom.TMeasyState()
tirerf_st = rom.TMeasyState()
tirelr_st = rom.TMeasyState()
tirerr_st = rom.TMeasyState()
rom.tireInit(tire_param)


veh1_param._step = 0.001
tire_param._step = 0.001
step = veh1_param._step

# Set the parameters to the posteriors - crucial section
################################################################################################################################
veh1_param._powertrainMap[0]._y = np.mean(idata_acc['posterior']['p0tor'].values)
veh1_param._powertrainMap[1]._y = np.mean(idata_acc['posterior']['p1tor'].values)
veh1_param._powertrainMap[2]._y = np.mean(idata_acc['posterior']['p2tor'].values)
veh1_param._powertrainMap[3]._y = np.mean(idata_acc['posterior']['p3tor'].values)

veh1_param._lossesMap[0]._y = -np.mean(idata_acc['posterior']['p0loss'].values)
veh1_param._lossesMap[1]._y = -np.mean(idata_acc['posterior']['p1loss'].values) 
veh1_param._lossesMap[2]._y = -np.mean(idata_acc['posterior']['p2loss'].values)


#################################################################################################################################

# run forest run
mod = []
t = 0
timeStepNo = 0
while(t<endTime):
    # get the controls for the time step
    rom.getControls(controls, driverData, t)

    #transfrom vehicle velocities to tire velocites
    rom.vehToTireTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

    # advanvce our 4 tires with this transformed velocity
    rom.tireAdv(tirelf_st, tire_param, veh1_st, veh1_param, controls)
    rom.tireAdv(tirerf_st, tire_param, veh1_st, veh1_param, controls)

    # rear wheel does not steer, so need to give it modified controls
    mod_controls = [controls[0],0,controls[2],controls[3]]

    rom.tireAdv(tirelr_st, tire_param, veh1_st, veh1_param, mod_controls)
    rom.tireAdv(tirerr_st, tire_param, veh1_st, veh1_param, mod_controls)

    rom.evalPowertrain(veh1_st, tirelf_st, tirerf_st, tirelr_st, tirerr_st, veh1_param, tire_param, controls)

    # transfrom tire forces into vehicle coordinate frame
    rom.tireToVehTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

    # copy useful stuff needed for the vehicle to advance
    fx = [tirelf_st._fx,tirerf_st._fx,tirelr_st._fx,tirerr_st._fx]
    fy = [tirelf_st._fy,tirerf_st._fy,tirelr_st._fy,tirerr_st._fy]
    huf = tirelf_st._rStat
    hur = tirerr_st._rStat

    # look at the vehicle go
    rom.vehAdv(veh1_st,veh1_param,fx,fy,huf,hur)


    #append timestep results
    if(timeStepNo % 10 == 0):
        mod.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                        veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                        tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])

    t += step
    timeStepNo += 1

# save all our models
model_post_mean = np.array(mod)


############################################### Step response 100 lines ##################################
#create an empy list that stores all the 100 numpy array of responses
model_post_st = [None]*n

# fill up our driver data from the file
driverData = rom.vector_entry()
rom.driverInput(driverData,fileName_con_st)

## Loop over each posterior sample
for i in range(n):
    # lets get our vector of doubles which will hold the controls at each time
    controls = rom.vector_double(4,0)

    veh1_param = rom.VehicleParam()
    rom.setVehParamsJSON(veh1_param,fileName_veh)
    tire_param = rom.TMeasyParam()
    rom.setTireParamsJSON(tire_param,fileName_tire)

    # Initialize our vehicle state in each iteration 
    veh1_st = rom.VehicleState()
    rom.vehInit(veh1_st,veh1_param)

    tirelf_st = rom.TMeasyState()
    tirerf_st = rom.TMeasyState()
    tirelr_st = rom.TMeasyState()
    tirerr_st = rom.TMeasyState()
    rom.tireInit(tire_param)


    veh1_param._step = 0.001
    tire_param._step = 0.001
    step = veh1_param._step

    # Set the parameters to the posteriors - crucial section
    ################################################################################################################################

    veh1_param._powertrainMap[0]._y = idata_acc_n['p0tor'][i].values.item()
    veh1_param._powertrainMap[1]._y = idata_acc_n['p1tor'][i].values.item()
    veh1_param._powertrainMap[2]._y = idata_acc_n['p2tor'][i].values.item()
    veh1_param._powertrainMap[3]._y = idata_acc_n['p3tor'][i].values.item()
    

    veh1_param._lossesMap[0]._y = -idata_acc_n['p0loss'][i].values.item()
    veh1_param._lossesMap[1]._y = -idata_acc_n['p1loss'][i].values.item()
    veh1_param._lossesMap[2]._y = -idata_acc_n['p2loss'][i].values.item()

    #################################################################################################################################

    # run forest run
    mod = []
    t = 0
    timeStepNo = 0
    while(t<endTime_st):
        # get the controls for the time step
        rom.getControls(controls, driverData, t)

        #transfrom vehicle velocities to tire velocites
        rom.vehToTireTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

        # advanvce our 4 tires with this transformed velocity
        rom.tireAdv(tirelf_st, tire_param, veh1_st, veh1_param, controls)
        rom.tireAdv(tirerf_st, tire_param, veh1_st, veh1_param, controls)

        # rear wheel does not steer, so need to give it modified controls
        mod_controls = [controls[0],0,controls[2],controls[3]]

        rom.tireAdv(tirelr_st, tire_param, veh1_st, veh1_param, mod_controls)
        rom.tireAdv(tirerr_st, tire_param, veh1_st, veh1_param, mod_controls)

        rom.evalPowertrain(veh1_st, tirelf_st, tirerf_st, tirelr_st, tirerr_st, veh1_param, tire_param, controls)

        # transfrom tire forces into vehicle coordinate frame
        rom.tireToVehTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

        # copy useful stuff needed for the vehicle to advance
        fx = [tirelf_st._fx,tirerf_st._fx,tirelr_st._fx,tirerr_st._fx]
        fy = [tirelf_st._fy,tirerf_st._fy,tirelr_st._fy,tirerr_st._fy]
        huf = tirelf_st._rStat
        hur = tirerr_st._rStat

        # look at the vehicle go
        rom.vehAdv(veh1_st,veh1_param,fx,fy,huf,hur)


        #append timestep results
        if(timeStepNo % 10 == 0):
            mod.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                            veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                            tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])

        t += step
        timeStepNo += 1
        
    # save all our models
    model_post_st[i] = np.array(mod)

################################################### Step mean response ###############################

controls = rom.vector_double(4,0)

veh1_param = rom.VehicleParam()
rom.setVehParamsJSON(veh1_param,fileName_veh)
tire_param = rom.TMeasyParam()
rom.setTireParamsJSON(tire_param,fileName_tire)

# Initialize our vehicle state in each iteration 
veh1_st = rom.VehicleState()
rom.vehInit(veh1_st,veh1_param)

tirelf_st = rom.TMeasyState()
tirerf_st = rom.TMeasyState()
tirelr_st = rom.TMeasyState()
tirerr_st = rom.TMeasyState()
rom.tireInit(tire_param)


veh1_param._step = 0.001
tire_param._step = 0.001
step = veh1_param._step

# Set the parameters to the posteriors - crucial section
################################################################################################################################
veh1_param._powertrainMap[0]._y = np.mean(idata_acc['posterior']['p0tor'].values)
veh1_param._powertrainMap[1]._y = np.mean(idata_acc['posterior']['p1tor'].values)
veh1_param._powertrainMap[2]._y = np.mean(idata_acc['posterior']['p2tor'].values)
veh1_param._powertrainMap[3]._y = np.mean(idata_acc['posterior']['p3tor'].values)

veh1_param._lossesMap[0]._y = -np.mean(idata_acc['posterior']['p0loss'].values)
veh1_param._lossesMap[1]._y = -np.mean(idata_acc['posterior']['p1loss'].values) 
veh1_param._lossesMap[2]._y = -np.mean(idata_acc['posterior']['p2loss'].values)


#################################################################################################################################

# run forest run
mod = []
t = 0
timeStepNo = 0
while(t<endTime_st):
    # get the controls for the time step
    rom.getControls(controls, driverData, t)

    #transfrom vehicle velocities to tire velocites
    rom.vehToTireTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

    # advanvce our 4 tires with this transformed velocity
    rom.tireAdv(tirelf_st, tire_param, veh1_st, veh1_param, controls)
    rom.tireAdv(tirerf_st, tire_param, veh1_st, veh1_param, controls)

    # rear wheel does not steer, so need to give it modified controls
    mod_controls = [controls[0],0,controls[2],controls[3]]

    rom.tireAdv(tirelr_st, tire_param, veh1_st, veh1_param, mod_controls)
    rom.tireAdv(tirerr_st, tire_param, veh1_st, veh1_param, mod_controls)

    rom.evalPowertrain(veh1_st, tirelf_st, tirerf_st, tirelr_st, tirerr_st, veh1_param, tire_param, controls)

    # transfrom tire forces into vehicle coordinate frame
    rom.tireToVehTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

    # copy useful stuff needed for the vehicle to advance
    fx = [tirelf_st._fx,tirerf_st._fx,tirelr_st._fx,tirerr_st._fx]
    fy = [tirelf_st._fy,tirerf_st._fy,tirelr_st._fy,tirerr_st._fy]
    huf = tirelf_st._rStat
    hur = tirerr_st._rStat

    # look at the vehicle go
    rom.vehAdv(veh1_st,veh1_param,fx,fy,huf,hur)


    #append timestep results
    if(timeStepNo % 10 == 0):
        mod.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                        veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                        tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])

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
    print(f"Longitudinal Velocity Error = {np.sqrt(np.sum((model_post_mean[:,3] - data['velo'])**2)/model_post_mean.shape[0])}")

    print(f"Step Test{test_nums_step[i]}")
    print(f"Longitudinal Velocity Error = {np.sqrt(np.sum((model_post_mean_st[:,3] - data_st['velo'])**2)/model_post_mean_st.shape[0])}")

    # plot the data
    axes[0].plot(data['time'],data['velo'],cols[i],label =f'Test {i + 1}',alpha = 0.5)
    axes[1].plot(data_st['time'],data_st['velo'],cols[i],label =f'Test {i + 1}',alpha = 0.5)


### Plots




# plot all the posteriors
### Ramp
for i in range(0,n):
    axes[0].plot(data['time'],model_post[i][:,[3]],'b',alpha = 0.2)

# plot the mean response
axes[0].plot(data['time'],model_post_mean[:,[3]],'y',label = 'Posterior Mean response')


axes[0].set_title("Ramp response")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Longitudinal velocity (m/s)")
axes[0].legend()



### Step
for i in range(0,n):
    axes[1].plot(data_st['time'],model_post_st[i][:,[3]],'b',alpha = 0.2)

# plot the mean response
axes[1].plot(data_st['time'],model_post_mean_st[:,[3]],'y',label = 'Posterior Mean response')

# plot the data

axes[1].set_title("Step response")
axes[1].set_xlabel("Time (s)")
axes[1].legend()


fig.tight_layout()
save = int(sys.argv[1])
if(save):
    mpl.savefig(f"./images/art_train_long_postResp.eps", format='eps', dpi=3000) 
    mpl.savefig(f"./images/art_train_long_postResp", facecolor = 'w', dpi = 600) 

mpl.show()