# import our reduced order model
import rom
import time
import numpy as np


# get the file name of the vehicle controls
# fileName_con = "../inputs/test_set2.txt"
# fileName_con = "../inputs/st.txt"
fileName_con = "../inputs/ramp_steer2.txt"

# json parameters file names

fileName_veh = "../jsons/HMMWV.json"
# fileName_veh = "../jsons/dART.json"
fileName_tire = "../jsons/TMeasy.json"
# fileName_tire = "../jsons/dARTTM.json"


# lets get a vector of entries going 
driverData = rom.vector_entry()


# lets fill this up from our data file
rom.driverInput(driverData,fileName_con)

# lets get our vector of doubles which will hold the controls at each time
controls = rom.vector_double(4,0)


# intialize our vehicle parameters and states
veh1_param = rom.VehicleParam()
veh1_st = rom.VehicleState()

# set params from json file
rom.setVehParamsJSON(veh1_param,fileName_veh)

# initialzie vehicle
rom.vehInit(veh1_st,veh1_param)

#initialize our 4 tire states and 1 tire param
tire_param = rom.TMeasyParam()
tirelf_st = rom.TMeasyState()
tirerf_st = rom.TMeasyState()
tirelr_st = rom.TMeasyState()
tirerr_st = rom.TMeasyState()

# tire parameters from json
rom.setTireParamsJSON(tire_param,fileName_tire)

# initialize the tire
rom.tireInit(tire_param)


# end time of simulation
endTime = 14.509
# endTime = 10.009
# time step supplied
veh1_param._step = 0.001
tire_param._step = 0.001
step = veh1_param._step

timeStepNo = 0
t = 0
result = []

start = time.process_time()

# running multiple simulations 
# for i in range(1):
    # t = 0
    # result = []
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
        result.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                        veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                        tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])

    t += step
    timeStepNo += 1
    
    # reset everything
    # veh1_st = rom.VehicleState()
    # rom.vehInit(veh1_st,veh1_param)

    # tirelf_st = rom.TMeasyState()
    # tirerf_st = rom.TMeasyState()
    # tirelr_st = rom.TMeasyState()
    # tirerr_st = rom.TMeasyState()
    # rom.tireInit(tire_param)


stop = time.process_time()

print(f"Time take is {(stop - start)*1000}")

#write the list to a csv file 
np.savetxt("../outs/ramp_st_mod2_wrapped.csv", np.array(result), delimiter=",")