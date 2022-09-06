# accuracy tests
# step size 1e-3
# look at how step sizes influece half implicit and fully implicit 

import matplotlib.pyplot as plt
import numpy as np
import platform
from SimEngineMBD.example_models.slider_crank import run_slider_crank_frictional

from matplotlib import ticker

if platform.system() == 'Darwin':
    fileloc = "/Users/luning/Manuscripts/Conference/2022/IROS_Half_Implicit/images/"
if platform.system() == 'Linux':
    fileloc = "/home/luning/Manuscripts/Journal/2022/HalfImplicit/images/"


Fontsize = 120
LineWidth = 10
MarkerSize = 40
plt.rc('font', size=Fontsize)
plt.rc('legend', fontsize=Fontsize*1.2)
plt.rc('figure', titlesize=Fontsize)
plt.rc('lines', linewidth=LineWidth)
plt.rc('lines', markersize=MarkerSize)
plt.rc('axes', linewidth=LineWidth*0.5)


to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε', 'rA_half': 'rA_half'}

t_end = 2

# step_sizes = [5e-5, 1e-4, 5e-4, 1e-3]
num_bodies = 2
form = 'rA_half'
model_fn = run_slider_crank_frictional


step_size = 1e-3
tolerance = 1e-10

t = np.arange(step_size, t_end+step_size, step_size)

fric_coeff = 0.2

saveFig=True

    

# now do some plotting
fig, ax = plt.subplots(1,1,figsize=(50,30))

for mu in [0, 0.2, 0.4]:

    pos, velo, acc, F_joint_data, Tr_joint_data, fric_data, numItr,  _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end), '--friction_coeff', str(mu)])




    # hdl = ax[0,0]
    # hdl.plot(t[:],  pos[2,0,:], label='\mu = {}'.format(mu))
    # hdl.set_ylabel('slider position (m)')
    # hdl.set_title('dt = {}'.format(step_size))
    # hdl.grid(b=True, which='major', linestyle='-')
    # hdl.grid(b=True, which='minor', linestyle='--')
    # hdl.minorticks_on()
    # hdl.set_xlim([0, t_end])


    # hdl = ax[0,1]    
    # hdl.plot(t[:], velo[2,0,:], label='\mu = {}'.format(mu))
    # hdl.set_ylabel('slider velocity (m/s)')
    # hdl.grid(b=True, which='major', linestyle='-')
    # hdl.grid(b=True, which='minor', linestyle='--')
    # hdl.minorticks_on()
    # hdl.set_xlim([0, t_end])
    
    # hdl = ax[0,2]
    # hdl.plot(t[:],  acc[2,0,:], label='\mu = {}'.format(mu))
    # hdl.set_ylabel(r'slider acceleration $(m/s^2)$')
    # hdl.grid(b=True, which='major', linestyle='-')
    # hdl.grid(b=True, which='minor', linestyle='--')
    # hdl.minorticks_on()
    # hdl.set_xlim([0, t_end])
    
    # hdl = ax[1,0]
    # hdl.plot(t[:],  F_joint_data[0,:], label='\mu = {}'.format(mu))
    # hdl.set_ylabel('translational joint normal force (N)')
    # hdl.set_xlabel('time (sec)')
    # hdl.grid(b=True, which='major', linestyle='-')
    # hdl.grid(b=True, which='minor', linestyle='--')
    # hdl.minorticks_on()
    # hdl.set_xlim([0, t_end])
    
    
    # hdl = ax[1,1]
    # hdl.plot(t[:],  fric_data[0,:], label='\mu = {}'.format(mu))
    # hdl.set_ylabel('friction force (N)')
    # hdl.set_xlabel('time (sec)')
    # hdl.grid(b=True, which='major', linestyle='-')
    # hdl.grid(b=True, which='minor', linestyle='--')
    # hdl.minorticks_on()
    # hdl.set_xlim([0, t_end])
    
    hdl = ax
    hdl.plot(t[5:],  Tr_joint_data[0,5:], label=r'$\mu$ = {}'.format(mu))
    hdl.set_ylabel('driving torque (Nm)')
    hdl.set_xlabel('time (sec)')
    hdl.grid(b=True, which='major', linestyle='-')
    hdl.grid(b=True, which='minor', linestyle='--')
    hdl.minorticks_on()
    hdl.legend()
    hdl.set_xlim([0, t_end])
    hdl.set_title('slider crank model with joint friction', fontsize=Fontsize)




# pretty_title = 'Bar {}_{}'.format(body, to_xyz[component])
# myAx.set(title=pretty_title)
# myAx.set_xlim([0, t_end])
# myAx.grid(linestyle='--')
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)

# if body == 1: 
# myAx.set(xlabel='time (sec)')

# if component == 1:
# myAx.set(ylabel='pos diff cmp to \n tol=%.0E, dt=%.0E'%(tol, dt_ref))

# if body == 1 and component == 2:
# myAx.legend()
        

    # plt.xlabel('time')
    # plt.ylabel('position diff')
    # plt.title(title)
    # ax.legend()

filename = "slider_crank_joint_friction.png"
if saveFig == True:
    plt.savefig(fileloc + filename)