# accuracy tests
# step size 1e-3
# look at how step sizes influece half implicit and fully implicit 

import matplotlib.pyplot as plt
import numpy as np

from SimEngineMBD.example_models.double_pendulum import run_double_pendulum
from matplotlib import ticker


to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε', 'rA_half': 'rA_half'}

t_end = 1

# step_sizes = [5e-5, 1e-4, 5e-4, 1e-3]
num_bodies = 2
form = 'rA'
model_fn = run_double_pendulum

# generate reference solution
dt_ref = 1e-5
tol = 1e-15/dt_ref**2
t_ref = np.arange(dt_ref, t_end+dt_ref, dt_ref)

print('===============')
print('Run reference solution at step size {} and tolerance {}'.format(dt_ref, tol))    
print('===============')

pos_ref, _, _, _, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tol), '--step_size', str(dt_ref), '-t', str(t_end)])



step_sizes = [1e-5, 1e-5, 1e-5]
tols = [1e-12, 1e-13, 1e-14]


# each subplot is for component y or z of body 0 or 1
fig, ax = plt.subplots(2,2,figsize=(35,20))
for ii in range(0, len(step_sizes)):
    step_size = step_sizes[ii]
    t = np.arange(step_size, t_end+step_size, step_size)
    tolerance = tols[ii]/step_size**2

    # cherry pick reference solution so arrays have same length
    spacing = round(step_size/dt_ref)
    pos_exact = np.full((num_bodies, 3, len(t)), np.nan)
    for i in range(0, len(t)):
        pos_exact[:, :, i] = pos_ref[:, :, (i+1)*spacing-1]


    print('===============')
    print('Run {}, step_size {}, tolerance {}'.format(form, step_size, tolerance))    
    print('===============')
        
    pos, _, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])

    print("body 1 y direction %E solution: "%(pos[1, 1, -1]))
    
    # now do some plotting
    for body in range(0, num_bodies):
        for component in range(1,3):
            myAx = ax[body, component-1]
            myAx.plot(t, np.abs(pos_exact[body, component, :] - pos[body, component, :]), label='tol=%.0E, dt=%.0E, avgItr=%.2f'%(tolerance, step_size, np.average(numItr)))
            pretty_title = 'Bar {}_{}'.format(body, to_xyz[component])
            myAx.set(title=pretty_title)
            myAx.set_xlim([0, t_end])
            myAx.grid(linestyle='--')
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
        
            if body == 1: 
                myAx.set(xlabel='time (sec)')
        
            if component == 1:
                myAx.set(ylabel='pos diff cmp to \n tol=%.0E, dt=%.0E'%(tol, dt_ref))
            
            if body == 1 and component == 2:
                myAx.legend()
            
    # plt.show(True)

        # plt.xlabel('time')
        # plt.ylabel('position diff')
        # plt.title(title)
        # ax.legend()

    