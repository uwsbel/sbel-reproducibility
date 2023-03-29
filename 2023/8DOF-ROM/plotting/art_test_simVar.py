import sys
sys.path.append('/home/unjhawala/projectlets/model-repo/simple-vehicles/lang-c/interfaces')
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

#################################################################################

type_ = sys.argv[3]
input_file1 = sys.argv[1]
input_file2 = sys.argv[2]

######################################################################################### Simulation 1 #################################################
sub_folder = "Sin_" + type_
path = "../calibration/ART/data/ART1_032523_mocap/" +sub_folder + "/"
input_path = "../calibration/ART/inputs/ART1_032523_mocapV2/" + sub_folder + "/" 

data_file = "traj_" + input_file1 + ".csv"
fileName_con =  input_path + "/" + input_file1  + ".txt"

# The ART data
data_ART1 = pd.read_csv(path + data_file, sep = ",", header = "infer")

# json parameters file names
fileName_veh = "../calibration/ART/jsons/dART_play.json"
fileName_tire = "../calibration/ART/jsons/dARTTM_play.json"


endTime = data_ART1.iloc[-1,1]

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


print(endTime)

veh1_st._psi = data_ART1.loc[0,'yaw']
# run forest run
mod = []
# Add the inital conditions
mod.append([0, 0, 0, 0, 0, 0, veh1_st._psi, 0, 0, 0, 0, 0, 0])
t = 0
timeStepNo = 0

#  Set the tire parameters based on the nc files, we will do mean respose for now
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

        mod.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                        veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                        tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])


mod = np.array(mod)


######################################################################### Simulation 2 ##############################################################

sub_folder = "Sin_" + type_
path = "../calibration/ART/data/ART1_032523_mocap/" +sub_folder + "/"
input_path = "../calibration/ART/inputs/ART1_032523_mocap/" + sub_folder + "/" 

data_file = "traj_" + input_file2 + ".csv"
fileName_con =  input_path + "/" + input_file2  + ".txt"

# The ART data
data_ART2 = pd.read_csv(path + data_file, sep = ",", header = "infer")

# json parameters file names
fileName_veh = "../calibration/ART/jsons/dART_play.json"
fileName_tire = "../calibration/ART/jsons/dARTTM_play.json"


endTime = data_ART2.iloc[-1,1]

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


print(endTime)

veh1_st._psi = data_ART2.loc[0,'yaw']
# run forest run
mod2 = []
# Add the inital conditions
mod2.append([0, 0, 0, 0, 0, 0, veh1_st._psi, 0, 0, 0, 0, 0, 0])
t = 0
timeStepNo = 0

#  Set the tire parameters based on the nc files, we will do mean respose for now
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

        mod2.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                        veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                        tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])


mod2 = np.array(mod2)


#################################################################################### Plotting the 2 tests ##############################################

fig, axes = mpl.subplots(nrows = 1, ncols = 1, figsize = (6,6))




axes.plot(mod[:,1],mod[:,2],'y',label = f'ROM Experiment {1}')
axes.plot(mod2[:,1],mod2[:,2],'b',label = f'ROM Experiment {2}')

axes.plot(data_ART1['x'], data_ART1['y'], 'r', label = f'ART Experiment {1}')
axes.plot(data_ART2['x'], data_ART2['y'], 'k', label = f'ART Experiment {2}')

# print(data)
axes.set_ylim([-mod2[-1,1],mod2[-1,1]])

axes.legend()

axes.set_xlabel('X (m)')
axes.set_ylabel('Y (m)')


# cycle markers
markers = ['o', 's', 'D', '^', '*', 'h', 'x', 'p', '+', 'h' , '>']
pts = np.arange(0,1201,120)

for i,point in enumerate(pts):
    try:
        axes.scatter(mod[point,[1]],mod[point,[2]],marker = markers[i], s = 50, c = 'tab:olive')
        axes.scatter(mod2[point,[1]],mod2[point,[2]],marker = markers[i], s = 50, c = 'tab:blue')

        axes.scatter(data_ART1.loc[int(point/10),'x'],data_ART1.loc[int(point/10),'y'],marker = markers[i],s = 50, c = 'tab:red')
        axes.scatter(data_ART2.loc[int(point/10),'x'],data_ART2.loc[int(point/10),'y'],marker = markers[i],s = 50, c = 'tab:gray')
    except:

        axes.scatter(mod[-1,[1]],mod[-1,[2]],marker = markers[i], s = 50, c = 'tab:olive')
        axes.scatter(mod2[-1,[1]],mod2[-1,[2]],marker = markers[i], s = 50, c = 'tab:blue')

        axes.scatter(data_ART1.loc[data_ART1['x'].shape[0]-1,'x'],data_ART1.loc[data_ART1['x'].shape[0]-1,'y'],marker = markers[i],s = 50, c = 'tab:red')
        axes.scatter(data_ART2.loc[data_ART2['x'].shape[0]-1,'x'],data_ART2.loc[data_ART2['x'].shape[0]-1,'y'],marker = markers[i],s = 50, c = 'tab:gray')


fig.tight_layout()

save = int(sys.argv[4])
if(save):
    mpl.savefig("./images/art_test_comp.eps",format='eps', dpi=3000)
    mpl.savefig("./images/art_test_comp.png", dpi=600)


mpl.show()