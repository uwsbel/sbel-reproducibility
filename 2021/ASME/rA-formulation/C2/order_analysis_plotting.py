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

    if f.endswith('.pickle') and (f.startswith('Four_Link') or f.startswith('Slider_Crank')):
        files.append(f)

for file_name in files:
    with open(dir_path + file_name, 'rb') as handle:
        info, pos_diff, vel_diff, acc_diff = pickle.load(handle)

    name, form, body, component = info
    title_name = ' '.join(name.split('_'))
    title = '{} {} Order Analysis: Body {}, {}'.format(title_name, form, body, component)

    _, ax = plt.subplots()
    ax.loglog(step_sizes, [2*step for step in step_sizes], label='Order 2 trendline', markersize=15, linewidth=7)
    ax.loglog(step_sizes, pos_diff, label='Position', marker='o', linestyle='-', markersize=15, linewidth=7)
    ax.loglog(step_sizes, vel_diff, label='Velocity', marker='x', linestyle='-', markersize=15, linewidth=7)
    ax.loglog(step_sizes, acc_diff, label='Acceleration', marker='+', linestyle='-', markersize=15, linewidth=7)
    ax.set(xlabel='Step Size', ylabel='Absolute Difference', title=title)

    ax.legend(loc='lower right', prop={'size': 24})

    plt.gcf().set_size_inches(20, 12)

    filename = '{}_{}_Order_Analysis_Body_{}_{}'.format(name, form, body, component)
    plt.savefig(filename)

# plt.show()