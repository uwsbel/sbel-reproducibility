import pickle
import os
import numpy as np

import matplotlib.pyplot as plt

dir_path = './output/oa/'

plt.rcParams.update({'font.size': 30})


files = []
for f in os.listdir(dir_path):
    if f == 'steps.pickle':
        with open(dir_path + f, 'rb') as handle:
            step_sizes = pickle.load(handle)

        continue

    if f.endswith('.pickle') and f.startswith('Four Link'):
        files.append(f)

for file_name in files:
    with open(dir_path + file_name, 'rb') as handle:
        info, pos_diff, vel_diff, acc_diff = pickle.load(handle)

    name, form, body, component = info
    title = '{} {} Order Analysis: Body {}, {}'.format(name, form, body, component)

    fig, ax = plt.subplots()
    ax.loglog(step_sizes, [step*step for step in step_sizes], label='Order 2 trendline', markersize=15, linewidth=7)
    ax.loglog(step_sizes, pos_diff, label='Position', marker='o', linestyle='-', markersize=15, linewidth=7)
    ax.loglog(step_sizes, vel_diff, label='Velocity', marker='x', linestyle='-', markersize=15, linewidth=7)
    ax.loglog(step_sizes, acc_diff, label='Acceleration', marker='+', linestyle='-', markersize=15, linewidth=7)
    ax.set(xlabel='Step Size', ylabel='Absolute Difference', title=title)

    ax.legend(loc='lower right', prop={'size': 24})

    plt.gcf().set_size_inches(20, 12)
    plt.savefig('./output/oa/Four_Link_{}_OrderAnalysis_Body_{}_{}.png'.format(form, body, component))

#plt.show()