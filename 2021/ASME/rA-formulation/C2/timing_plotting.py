import pickle
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

dir_path = './output/timing/'

files = []
for f in os.listdir(dir_path):
    if f == 'mesh_params.pickle':
        with open(dir_path + f, 'rb') as handle:
            ss, MM = pickle.load(handle)
        
        continue

    if f.endswith('.pickle') and f.startswith('Four_Link'):
        files.append(f)

for file_name in files:
    with open(dir_path + file_name, 'rb') as handle:
        info, timing = pickle.load(handle)

    title = '{} {} Timing Analysis'.format(*info)

    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(np.log10(ss), np.log10(MM), timing)
    ax.set(xlabel='log( Step Size )', ylabel='log( (Step Size)^2 * Θ )')

plt.show()