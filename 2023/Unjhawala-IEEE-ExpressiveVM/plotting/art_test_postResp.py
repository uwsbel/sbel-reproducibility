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
import logging

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
fileNameTh = "20221214_080100_test4"


filename02 = "20230302_183045"
filename04 = "20230302_181306"
filename06 = "20230302_115955"
filename08 = "20230302_115455"
filename1 = "20230302_180819" 


idataTh = az.from_netcdf("../calibration/ART/results/" + fileNameTh + ".nc")
idata1 = az.from_netcdf("../calibration/ART/results/" + filename1 + ".nc")
idata08 = az.from_netcdf("../calibration/ART/results/" + filename08 + ".nc")
idata06 = az.from_netcdf("../calibration/ART/results/" + filename06 + ".nc")
idata04 = az.from_netcdf("../calibration/ART/results/" + filename04 + ".nc")
idata02 = az.from_netcdf("../calibration/ART/results/" + filename02 + ".nc")



################################ Sample 100 parameters from it ##############################
n = 100
idataTh_n = az.extract(idataTh, num_samples=n)
idata1_n = az.extract(idata1, num_samples=n)
idata08_n = az.extract(idata08, num_samples=100)
idata06_n = az.extract(idata06, num_samples=100)
idata04_n = az.extract(idata04, num_samples=100)
idata02_n = az.extract(idata02, num_samples=100)

#################################################################################


sub_folder = "Sin_" + sys.argv[2]
path = "../calibration/ART/data/ART1_032523_mocap/" +sub_folder + "/"
input_path = "../calibration/ART/inputs/ART1_032523_mocapV2/" + sub_folder + "/" 

input_file = sys.argv[1]
data_file = "traj_" + input_file + ".csv"
fileName_con =  input_path + "/" + input_file  + ".txt"

# The ART data
data_ART = pd.read_csv(path + data_file, sep = ",", header = "infer")
# print(data_ART.columns)

# json parameters file names
fileName_veh = "../calibration/ART/jsons/dART_play.json"
fileName_tire = "../calibration/ART/jsons/dARTTM_play.json"


endTime = data_ART.iloc[-1,1]

print(f"Subfolder : {sub_folder}, test file {data_file}")

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

    veh1_param._powertrainMap[0]._y = idataTh_n['p0tor'][i].values.item() - 0.03
    veh1_param._powertrainMap[1]._y = idataTh_n['p1tor'][i].values.item() - 0.03
    veh1_param._powertrainMap[2]._y = idataTh_n['p2tor'][i].values.item() - 0.03
    veh1_param._powertrainMap[3]._y = idataTh_n['p3tor'][i].values.item() - 0.03
    

    veh1_param._lossesMap[0]._y = -idataTh_n['p0loss'][i].values.item()
    veh1_param._lossesMap[1]._y = -idataTh_n['p1loss'][i].values.item()
    veh1_param._lossesMap[2]._y = -idataTh_n['p2loss'][i].values.item()


    # Steering
    veh1_param._steerMap[0]._y = -idata1_n['f_10'][i].values.item()
    veh1_param._steerMap[-1]._y = idata1_n['f_10'][i].values.item()
            
    veh1_param._steerMap[1]._y = -idata08_n['f_08'][i].values.item()
    veh1_param._steerMap[-2]._y = idata08_n['f_08'][i].values.item()

    veh1_param._steerMap[2]._y = -idata06_n['f_06'][i].values.item()
    veh1_param._steerMap[-3]._y = idata06_n['f_06'][i].values.item()
    
    veh1_param._steerMap[3]._y = -idata04_n['f_04'][i].values.item()
    veh1_param._steerMap[-4]._y = idata04_n['f_04'][i].values.item()
    
    veh1_param._steerMap[4]._y = -idata02_n['f_02'][i].values.item()
    veh1_param._steerMap[-5]._y = idata02_n['f_02'][i].values.item()
    #################################################################################################################################

    # run forest run
    mod = []
    veh1_st._psi = data_ART.loc[0,'yaw']
    mod.append([0, 0, 0, 0, 0, 0, veh1_st._psi, 0, 0, 0, 0, 0, 0])
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
            mod.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                            veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                            tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])
            
    

    model_post[i] = np.array(mod)

############################################################ Mean response #############################################################
# lets get a vector of entries going 
driverData = rom.vector_entry()


# lets fill this up from our data file
rom.driverInput(driverData,fileName_con)

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


# print(endTime)

veh1_st._psi = data_ART.loc[0,'yaw']
# run forest run
mod_mean = []
# Add the inital conditions
mod_mean.append([0, 0, 0, 0, 0, 0, veh1_st._psi, 0, 0, 0, 0, 0, 0])
t = 0
timeStepNo = 0

#  Set the tire parameters based on the nc files, using mean response
veh1_param._steerMap[0]._y = -np.mean(idata1['posterior']['f_10'].values)
veh1_param._steerMap[-1]._y = np.mean(idata1['posterior']['f_10'].values)
        
veh1_param._steerMap[1]._y = -np.mean(idata08['posterior']['f_08'].values)
veh1_param._steerMap[-2]._y = np.mean(idata08['posterior']['f_08'].values)
            
veh1_param._steerMap[2]._y = -np.mean(idata06['posterior']['f_06'].values)
veh1_param._steerMap[-3]._y = np.mean(idata06['posterior']['f_06'].values)
                
veh1_param._steerMap[3]._y = -np.mean(idata04['posterior']['f_04'].values)
veh1_param._steerMap[-4]._y = np.mean(idata04['posterior']['f_04'].values)
                
veh1_param._steerMap[4]._y = -np.mean(idata02['posterior']['f_02'].values)
veh1_param._steerMap[-5]._y = np.mean(idata02['posterior']['f_02'].values)

#### Powertrain maps
veh1_param._powertrainMap[0]._y = np.mean(idataTh['posterior']['p0tor'].values) - 0.03
veh1_param._powertrainMap[1]._y = np.mean(idataTh['posterior']['p1tor'].values) - 0.03
veh1_param._powertrainMap[2]._y = np.mean(idataTh['posterior']['p2tor'].values) - 0.03
veh1_param._powertrainMap[3]._y = np.mean(idataTh['posterior']['p3tor'].values) - 0.03

veh1_param._lossesMap[0]._y = -np.mean(idataTh['posterior']['p0loss'].values)
veh1_param._lossesMap[1]._y = -np.mean(idataTh['posterior']['p1loss'].values) 
veh1_param._lossesMap[2]._y = -np.mean(idataTh['posterior']['p2loss'].values)



while(t < (endTime + step)):
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
        # print(t)
        # if(veh1_st._psi < -0.01):
        #     veh1_st._psi = veh1_st._psi + py_2PI
        # elif(veh1_st._psi > py_2PI):
        #     veh1_st._psi = veh1_st._psi - py_2PI
        # else:
            # psi = veh1_st._psi

        mod_mean.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                        veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                        tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])


mod_mean = np.array(mod_mean)

# Extract the data 
data = pd.read_csv(path + data_file, header = 'infer', sep = ',')

## To remove some stupid printing
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True
############################################################

fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (6,6))

for i in range(0,n):
    axes.plot(model_post[i][:,[1]],model_post[i][:,[2]],'b',alpha = 0.1)

axes.plot(data['x'], data['y'], 'r', label = 'ART mocap data')
axes.plot(mod_mean[:,1],mod_mean[:,2],'y',label = 'ROM mean posterior Response')




# print(data)
axes.set_ylim([-data.loc[data['x'].shape[0]-1,'x'],data.loc[data['x'].shape[0]-1,'x']])

axes.legend()

axes.set_xlabel('X (m)')
axes.set_ylabel('Y (m)')


# cycle markers
markers = ['o', 's', 'D', '^', '*', 'h', 'x', 'p', '+', 'h' , '>']
pts = np.arange(0,1201,120)

for i,point in enumerate(pts):
    try:
        axes.scatter(data.loc[int(point/10),'x'],data.loc[int(point/10),'y'],marker = markers[i],s = 50, c = 'tab:red')
        axes.scatter(mod_mean[point,[1]],mod_mean[point,[2]],marker = markers[i], s = 50, c = 'tab:olive')
    except:

        axes.scatter(data.loc[data['x'].shape[0]-1,'x'],data.loc[data['x'].shape[0]-1,'y'],marker = markers[i],s = 50, c = 'tab:red')
        axes.scatter(mod_mean[-1,[1]],mod_mean[-1,[2]],marker = markers[i], s = 50, c = 'tab:olive')



# Error analysis
pts2 = np.arange(0,1201,12)


points_data = data.iloc[:,2:4]
points_response = mod_mean[:,1:3]
lv1 = []
for i, point in enumerate(pts2):
    try:
        lv1.append(np.sum(np.sqrt((points_data.iloc[int(point/10),1] - points_response[point,1])**2 + (points_data.iloc[int(point/10),0] - points_response[point,0])**2)))
    except:
        lv1.append(np.sum(np.sqrt((points_data.iloc[-1,1] - points_response[-1,1])**2 + (points_data.iloc[-1,0] - points_response[-1,0])**2)))





fig.tight_layout()

save = int(sys.argv[3])
if(save):
    mpl.savefig("./images/art_test_" + sub_folder + "_" + input_file + ".eps",format='eps', dpi=3000)
    mpl.savefig("./images/art_test_" + sub_folder + "_" + input_file + ".png", dpi=600)

# mpl.show()
print(f"Mean is {sum(lv1)/len(lv1)}")
total_distance = traj_len(data)
print(f"Mean ED / Total distance = {sum(lv1)/(len(lv1)*total_distance)}\n")
print(mod_mean.shape)