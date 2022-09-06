# how half and fully implicit solution compares
# with kinematics result (fully constrained problems, tol = 1e-10 at dt 1e-5)
# slider crank and four link problems
# refernce solution and dae solution from dynamics are pickled

import matplotlib.pyplot as plt
import numpy as np
import pickle
from SimEngineMBD.example_models.slider_crank import run_slider_crank
from SimEngineMBD.example_models.four_link import run_four_link


to_xyz = 'xyz'
pretty_form = {'rp': 'rp', 'rA': 'rA', 'reps': 'rε', 'rA_half': 'rA_half'}

num_bodies = 3

legends = []
# for form in forms:
#     if form == 'rA_half':
#         legends.append('half implicit')
#     else:
#         legends.append('fully implicit')


model_fn = run_four_link
# same step size
# check slides/tech report for tolerances
tolerance = 1e-10
step_sizes = [1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 4e-3, 1e-2, 2e-2, 4e-2]

need_ground_truth = False
dir_path = 'ground_truth/'

# t_end = 0.5
# dt_ref = 1e-5

t_end = 8
dt_ref = 1e-5


if need_ground_truth == False:
    # generate reference solution
    tol_ref = 1e-10
    t_ref = np.arange(dt_ref, t_end+dt_ref, dt_ref)
    
    pos_ref, vel_ref, acc_ref, _, _ = model_fn(['--form', 'rA', '--mode', 'kinematics', '--tol', str(tol_ref), '--step_size', str(dt_ref), '-t', str(t_end)])
    
    info = (model_fn.__name__[4:], tol_ref, dt_ref, t_end)
    pickle_name = '{}_kinematics_soln_all.pickle'.format(info[0])
    with open(dir_path + pickle_name, 'wb') as handle:
        pickle.dump((info, pos_ref, vel_ref, acc_ref, t_ref), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Sucessfully write ground truth!")
        
else:
    # ground truth solution already exist, load pickle
    pretty_name = model_fn.__name__[4:]
    pickle_name = '{}_kinematics_soln_all.pickle'.format(pretty_name)
    with open(dir_path + pickle_name, 'rb') as handle:
        info, pos_ref, vel_ref, acc_ref, t_ref = pickle.load(handle)
        _, tol_ref, dt_ref, t_end = info
        
    t_ref = np.arange(dt_ref, t_end+dt_ref, dt_ref)
    


# list of bodies and coordinates for plotting, i only care about body 1 x y z and body 2 x
# body_list = [1, 1, 1, 2]
# if model_fn.__name__ == 'run_four_link':
#     body_names = ['rotor', 'link 1', 'link 2']
# elif model_fn.__name__ == 'run_slider_crank':
#     body_names = ['crank', 'rod', 'slider']


# bodyname_list = []
# for body_id in body_list:
#     bodyname_list.append(body_names[body_id])
# coordinate_list = [0, 1, 2, 0]

Fontsize = 60
plt.rc('font', size=Fontsize)
plt.rc('legend', fontsize=Fontsize*0.9)
plt.rc('figure', titlesize=Fontsize*0.9)

# each subplot is for component [x,y,z] of body [0, 1, 2]
fig, ax = plt.subplots(1,1,figsize=(50, 35))


for form in ['rA', 'rA_half']:
    diff_list = []

    for ii, step_size in enumerate(step_sizes):

        t = np.arange(step_size, t_end+step_size, step_size)


        # cherry pick reference solution so arrays have same length
        spacing = round(step_size/dt_ref)
        velo_exact = np.full((num_bodies, 3, len(t)), np.nan)
        for i in range(0, len(t)):
            velo_exact[:, :, i] = vel_ref[:, :, (i+1)*spacing-1]


        print('===============')
        print('Run {}, step_size {}, tolerance {}'.format(form, step_size, tolerance))    
        print('===============')
        
        if form == 'rA':
        
            pos, velo, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance/step_size**2), '--step_size', str(step_size), '-t', str(t_end)])

        else:
            pos, velo, _, numItr, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tolerance), '--step_size', str(step_size), '-t', str(t_end)])
        
        
        pickle_name = '{}_{}_dynamics_dt_{}.pickle'.format(model_fn.__name__[4:], form, step_size)
        with open(dir_path + pickle_name, 'wb') as handle:
            pickle.dump((t, pos, velo, numItr), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        body = 1
        component = 2

        diff = np.abs(velo_exact[body, component, -1] - velo[body, component, -1])
        # diff_list.append(np.sqrt(np.sum(np.power(diff,2)))/np.size(diff))
        diff_list.append(diff)

    ax.loglog(step_sizes, np.array(diff_list), label=form)

ax.legend()
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, which='minor', linestyle='--')
ax.minorticks_on()
