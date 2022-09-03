# accuracy tests
# step size 1e-3
# look at how step sizes influece half implicit and fully implicit 

import matplotlib.pyplot as plt
import numpy as np

from SimEngineMBD.example_models.slider_crank import run_slider_crank

from matplotlib import ticker


to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε', 'rA_half': 'rA_half'}

t_end = 1

# step_sizes = [5e-5, 1e-4, 5e-4, 1e-3]
num_bodies = 2
form = 'rA_half'
model_fn = run_slider_crank


step_size = 1e-3
tolerance = 1e-10

t = np.arange(step_size, t_end+step_size, step_size)

fric_coeff = 0.2


    
pos, velo, acc, F_joint_data, Tr_joint_data, numItr,  _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])

print("body 1 y direction %E solution: "%(pos[1, 1, -1]))

# now do some plotting
fig, ax = plt.subplots(2,3,figsize=(35,20))

ax[0,0].plot(t,  pos[2,0,:], label='')
ax[0,1].plot(t, velo[2,0,:], label='')
ax[0,2].plot(t,  acc[2,0,:], label='')



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
        
# plt.show(True)

    # plt.xlabel('time')
    # plt.ylabel('position diff')
    # plt.title(title)
    # ax.legend()

