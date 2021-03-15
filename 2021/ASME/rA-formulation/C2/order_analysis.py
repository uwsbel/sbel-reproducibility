import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from four_link_oa import four_link
from slider_crank_oa import slider_crank
from single_pendulum_oa import single_pendulum

# step_sizes = np.logspace(-2, -5, 4)
step_sizes = [1e-3, 2e-3, 4e-3, 8e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1e-1]
tols = [1e-10 / step**2 for step in step_sizes]

to_xyz = {0: 'x', 1: 'y', 2: 'z'}    

plt.rcParams.update({'font.size': 30})

def run_model(args):
    form, model_fn, to_track = args
    body, component = to_track

    pos_exact, vel_exact, acc_exact, _, _ = model_fn(['--form', form, '--mode', 'kinematics', '--tol', '1e-12', '--tracked_body', str(body), '--tracked_component', str(component)])

    pretty_name = ' '.join([word.capitalize() for word in model_fn.__name__.split('_')])
    pretty_form = form if form != 'reps' else 'rε'

    # Start empty, we won't add the ones that fail to converge
    plot_steps = []
    pos_diff = []
    vel_diff = []
    acc_diff = []
    for i, steps in enumerate(step_sizes):
        try:
            pos, vel, acc, _, _ = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tols[i]), '--step_size', str(steps), '--tracked_body', str(body), '--tracked_component', str(component)])

            plot_steps.append(steps)
            pos_diff.append(np.abs(pos_exact[-1] - pos[-1]))
            vel_diff.append(np.abs(vel_exact[-1] - vel[-1]))
            acc_diff.append(np.abs(acc_exact[-1] - acc[-1]))
        except RuntimeError:
            print('{}-{}, step: {}, tol: {} failed to converge'.format(form, pretty_name, str(steps), str(tols[i])))

    title = '{} {} Order Analysis: Body {}, {}'.format(pretty_name, pretty_form, body, to_xyz[component])
    save_name = title.replace(' ', '_').replace(':', '').replace(',', '')

    print(pos_diff)
    print(vel_diff)
    print(acc_diff)

    _, ax = plt.subplots()
    ax.loglog(plot_steps, [2*step for step in plot_steps], label='Order 2 trendline', markersize=15, linewidth=7)
    ax.loglog(plot_steps, pos_diff, label='Position', marker='o', linestyle='-', markersize=15, linewidth=7)
    ax.loglog(plot_steps, vel_diff, label='Velocity', marker='x', linestyle='-', markersize=15, linewidth=7)
    ax.loglog(plot_steps, acc_diff, label='Acceleration', marker='+', linestyle='-', markersize=15, linewidth=7)
    ax.set(xlabel='Step Size', ylabel='Absolute Difference', title=title)

    ax.legend(loc='lower right', prop={'size': 24})

    plt.gcf().set_size_inches(20, 12)
    plt.savefig(save_name, dpi=100)

tasks = []

haug_combos = list(itertools.product(range(3), repeat=2))

for form in ['reps', 'rp']:
    for model_fn in [four_link, slider_crank]:
        if model_fn.__name__ == 'single_pendulum':
            combos = [(0, 0), (0, 1), (0, 2)]
        else:
            combos = haug_combos

        for combo in combos:
            tasks.append((form, model_fn, combo))

# for task in tasks:
#     run_model(task)

pool = Pool()
pool.map(run_model, tasks)
pool.close()
