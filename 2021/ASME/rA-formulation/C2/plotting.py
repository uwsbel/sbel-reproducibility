import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

from four_link_oa import four_link
from slider_crank_oa import slider_crank
from single_pendulum_oa import single_pendulum

model_fn = four_link

step_sizes = [1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2]
tols = [1e-10 / step**2 for step in step_sizes]

ticks = {'rA': '-+', 'rp': '-o', 'reps': '-x'}

plt.rcParams.update({'font.size': 30})
_, ax = plt.subplots()

for form in ['rA', 'rp', 'reps']:

    iters = []
    for steps, tol in zip(step_sizes, tols):

        args = ['--form', form, '--mode', 'kinematics', '--tol', str(1e-10), '--step_size', str(steps)]

        _, _, _, num_iters, _ = model_fn(args)

        iters.append(np.mean(num_iters))

    pretty_form = 'rε' if form == 'reps' else form
    ax.plot(step_sizes, iters, ticks[form], label=pretty_form, markersize=15, linewidth=7)

ax.set(xlabel='Tolerance', ylabel='Average Iterations to Convergence', title='Iterations to Convergence, Four Link Model, Kinematics')
ax.legend(loc='best', prop={'size': 24})
ax.set_xscale('log')

plt.gcf().set_size_inches(20, 12)
plt.savefig('tol-iters-convergence-kin', dpi=100)