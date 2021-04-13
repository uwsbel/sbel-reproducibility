from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from SimEngineMBD.example_models.single_pendulum import run_single_pendulum
from SimEngineMBD.example_models.four_link import run_four_link
from SimEngineMBD.example_models.slider_crank import run_slider_crank

model_fn = run_four_link
mode = 'dynamics'

# run_some_model_name -> Some_Model_Name
pretty_model_name = ' '.join([word.capitalize() for word in model_fn.__name__[4:].split('_')])

step_sizes = [1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2]
tols = [1e-10 / step**2 for step in step_sizes]

ticks = {'rA': '-+', 'rp': '-o', 'reps': '-x'}

plt.rcParams.update({'font.size': 30})
_, ax = plt.subplots()

for form in ['rA', 'rp', 'reps']:

    iters = []
    for steps, tol in zip(step_sizes, tols):

        args = ['--form', form, '--mode', mode, '--tol', str(tol), '--step_size', str(steps)]

        _, _, _, num_iters, _ = model_fn(args)

        iters.append(np.mean(num_iters))

    pretty_form = 'rε' if form == 'reps' else form
    ax.plot(step_sizes, iters, ticks[form], label=pretty_form, markersize=15, linewidth=7)

ax.set(xlabel='Step Size', ylabel='Average Iterations to Convergence', title='Iterations to Convergence, {} Model, {}'.format(pretty_model_name, mode.capitalize()))
ax.legend(loc='best', prop={'size': 24})
ax.set_xscale('log')

plt.gcf().set_size_inches(20, 12)
plt.savefig('tol-iters-convergence-kin', dpi=100)
