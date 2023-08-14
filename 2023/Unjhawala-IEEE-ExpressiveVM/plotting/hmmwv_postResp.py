import sys
import matplotlib.pyplot as mpl
import arviz as az
import rom
import pandas as pd
import numpy as np

"""
Command line arguments 
1) nc results file name without the extension
2) Input file for the manuever without extension
3) Data file name we are comparing against without extension
4) Flag for whether we want to save the file or not
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


rpm2rad = np.pi / 30.
vehicle = "HMMWV"
filename = sys.argv[1]
idata_acc = az.from_netcdf("../calibration/" + vehicle + "/results/" + filename + ".nc")


# set the number of sampels to be drawn
#############################################################################
n = 100
#############################################################################

idata_acc_n = az.extract(idata_acc, num_samples=100)

test_file = sys.argv[2]

fileName_con = "../calibration/" + vehicle + "/inputs/" + test_file + ".txt"

# json files
fileName_veh = "../calibration/" + vehicle + "/jsons/HMMWV.json"
fileName_tire = "../calibration/" + vehicle + "/jsons/TMeasy.json"


# Get the simulation end time from the data
file_data = sys.argv[3]
data = pd.read_csv( "../calibration/" + vehicle + "/data/" + file_data + ".csv", sep = ",", header = "infer")
# end time of the simulation
endTime = data.iloc[-1,0]

############################################################## 100 Simulation ######################################################################
model_post = [None]*n # Stores a list of the dataframes for each simulation


for i in range (n):
    # lets get a vector of entries going 
    driverData = rom.vector_entry()

    # lets fill this up from our data file
    rom.driverInput(driverData,fileName_con)

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

    ################# Get the parameters from the  nc file ################################################


    tire_param._dfy0Pn = tire_param._dfy0Pn * idata_acc_n['f_dfy'][i].values.item()
    tire_param._dfy0P2n = tire_param._dfy0P2n * idata_acc_n['f_dfy'][i].values.item()

    tire_param._fymPn = tire_param._fymPn * idata_acc_n['f_fym'][i].values.item()
    tire_param._fymP2n = tire_param._fymP2n * idata_acc_n['f_fym'][i].values.item()

    tire_param._dfx0Pn = tire_param._dfx0Pn * idata_acc_n['f_dfx'][i].values.item()
    tire_param._dfx0P2n = tire_param._dfx0P2n * idata_acc_n['f_dfx'][i].values.item()

    tire_param._fxmPn = tire_param._fxmPn * idata_acc_n['f_fxm'][i].values.item()
    tire_param._fxmP2n = tire_param._fxmP2n * idata_acc_n['f_fxm'][i].values.item()

    veh1_param._maxSteer = veh1_param._maxSteer * idata_acc_n['f_maxSteer'][i].values.item()

    # Map factors
    loss_size = veh1_param._lossesMap.size()
    for k in range(loss_size):
        veh1_param._lossesMap[k]._y = veh1_param._lossesMap[k]._y  * idata_acc_n['f_loss'][i].values.item()

###############################################################################################################

    # Create the output dataframe 
    column_names = ['time', 'x', 'y', 'vx', 'vy', 'roll', 'yaw', 'wx', 'wz', 'omega_lf', 'omega_rf', 'omega_lr', 'omega_rr']

    # Create an empty dataframe with column names
    results = []
    results.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    t = 0
    timeStepNo = 0


    while(t < (endTime - step/10)):


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

        t += step
        timeStepNo += 1

        #append timestep results
        if(timeStepNo % 10 == 0):
            results.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                            veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                            tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])
            
    

    model_post[i] = np.array(results)


#################################################################### Mean posterior response ################################################

# lets get a vector of entries going 
driverData = rom.vector_entry()

# lets fill this up from our data file
rom.driverInput(driverData,fileName_con)

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

################# Get the parameters from the  nc file ################################################


loss_size = veh1_param._lossesMap.size()
for k in range(loss_size):
    veh1_param._lossesMap[k]._y = veh1_param._lossesMap[k]._y  * np.mean(idata_acc['posterior']['f_loss'].values)


veh1_param._maxSteer = veh1_param._maxSteer * np.mean(idata_acc['posterior']['f_maxSteer'].values)

tire_param._dfx0Pn = tire_param._dfx0Pn * np.mean(idata_acc['posterior']['f_dfx'].values)
tire_param._dfx0P2n = tire_param._dfx0P2n *  np.mean(idata_acc['posterior']['f_dfx'].values)

tire_param._fxmPn = tire_param._fxmPn *  np.mean(idata_acc['posterior']['f_fxm'].values)
tire_param._fxmP2n = tire_param._fxmP2n * np.mean(idata_acc['posterior']['f_fxm'].values)

tire_param._dfy0Pn = tire_param._dfy0Pn * np.mean(idata_acc['posterior']['f_dfy'].values)
tire_param._dfy0P2n = tire_param._dfy0P2n *  np.mean(idata_acc['posterior']['f_dfy'].values)

tire_param._fymPn = tire_param._fymPn *  np.mean(idata_acc['posterior']['f_fym'].values)
tire_param._fymP2n = tire_param._fymP2n * np.mean(idata_acc['posterior']['f_fym'].values)

###############################################################################################################

# Create the output dataframe 
column_names = ['time', 'x', 'y', 'vx', 'vy', 'roll', 'yaw', 'wx', 'wz', 'omega_lf', 'omega_rf', 'omega_lr', 'omega_rr']

# Create an empty dataframe with column names
results = []
results.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
t = 0
timeStepNo = 0


while(t < (endTime - step/10)):


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

    t += step
    timeStepNo += 1

    #append timestep results
    if(timeStepNo % 10 == 0):
        results.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                        veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                        tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])
model_post_mean = np.array(results)



#################################################################### Adding the noise to the data #################################################

data['vx'] = data['vx'] + np.random.normal(loc = 0., scale = 0.2,size = np.asarray(data['vx']).shape)
data['yaw'] = data['yaw'] + np.random.normal(loc = 0., scale = 0.02,size = np.asarray(data['yaw']).shape)


################################################################## Plotting -> Only plots the Long Vel and Yaw 

fig, axes = mpl.subplots(nrows = 1, ncols = 2, figsize = (10,5))

# Long velocity
axes[0].plot(data['time'],data['vx'],'r',label ='HMMWV')
for i in range(0,n):
    axes[0].plot(data['time'],model_post[i][:,[3]],'b',alpha = 0.2)
axes[0].plot(data['time'],model_post_mean[:,[3]],'y',label = 'Posterior Mean response')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("$V_x$ (m/s)")
axes[0].set_title("Longitudinal Velocity")
axes[0].legend(fontsize = 12)


# Yaw
axes[1].plot(data['time'],data['yaw'],'r',label ='HMMWV')
for i in range(0,n):
    axes[1].plot(data['time'],model_post[i][:,[6]],'b',alpha = 0.2)
axes[1].plot(data['time'],model_post_mean[:,[6]],'y',label = 'Posterior Mean response')
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("$\psi$ (rad)")
axes[1].set_title("Yaw")
axes[1].legend(fontsize = 12)

fig.tight_layout()

save = int(sys.argv[4])
if(save):
    fig.savefig(f'./images/hmmwv_{test_file}.eps', format='eps', dpi=3000)
    mpl.savefig(f'./images/hmmwv_{test_file}.png', facecolor = 'w', dpi = 600)
mpl.show()


############################################################## Calculating the mean RMSE ##################################################################
print(f"Test {test_file}")
print(f"Longitudinal Velocity Error = {np.sqrt(np.sum((model_post_mean[:,3] - data['vx'])**2)/model_post_mean.shape[0])}")
print(f"Yaw Angle Error = {np.sqrt(np.sum((model_post_mean[:,6] - data['yaw'])**2)/model_post_mean.shape[0])}")

