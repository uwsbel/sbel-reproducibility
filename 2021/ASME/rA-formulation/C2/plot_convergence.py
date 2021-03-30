import pickle
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

dir_path = './output/surf/'
    
files = []
for f in os.listdir(dir_path):
    if f == 'mesh_params.pickle':
        with open(dir_path + f, 'rb') as handle:
            ss, MM = pickle.load(handle)
        
        continue

    if f.endswith('.pickle') and f.startswith('Slider_Crank'):
        files.append(f)

for file_name in files:
    if file_name.endswith('iterations.pickle'):
        with open(dir_path + file_name, 'rb') as handle:
            info, conv_iters = pickle.load(handle)

        title = '{} {}'.format(*info)

        fig = plt.figure()
        fig.suptitle(title + ': 1/Iterations to Convergence')
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(np.log10(ss), np.log10(MM), conv_iters)
        ax.set(xlabel='log( Step Size )', ylabel='log( (Step Size)^2 * Θ )')

        continue

    with open(dir_path + file_name, 'rb') as handle:
        info, pos_diff, vel_diff, acc_diff = pickle.load(handle)
    
    name, form, body, component = info
    title = '{} {}, Body {}, {}-component'.format(name, form, body, component)

    fig2 = plt.figure()
    fig2.suptitle(title + ': Absolute Value of Position Difference')
    ax2 = fig2.gca(projection='3d')
    surf2 = ax2.plot_surface(np.log10(ss), np.log10(MM), np.log10(pos_diff))
    ax2.set(xlabel='log( Step Size )', ylabel='log( (Step Size)^2 * Θ )')

    fig3 = plt.figure()
    fig3.suptitle(title + ': Absolute Value of Velocity Difference')
    ax3 = fig3.gca(projection='3d')
    surf3 = ax3.plot_surface(np.log10(ss), np.log10(MM), np.log10(vel_diff))
    ax3.set(xlabel='log( Step Size )', ylabel='log( (Step Size)^2 * Θ )')

    fig4 = plt.figure()
    fig4.suptitle(title + ': Absolute Value of Acceleration Difference')
    ax4 = fig4.gca(projection='3d')
    surf4 = ax4.plot_surface(np.log10(ss), np.log10(MM), np.log10(acc_diff))
    ax4.set(xlabel='log( Step Size )', ylabel='log( (Step Size)^2 * Θ )')

plt.show()