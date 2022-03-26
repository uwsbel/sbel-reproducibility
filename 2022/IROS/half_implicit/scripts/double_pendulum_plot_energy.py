# double pendulum simulation, step size 1e-3, simulation time 10 sec
# compare energy of the pendulum using half- and fully-implicit
# symplectic method conserves energy while fully-implicit adds numerical damping

import matplotlib.pyplot as plt
import numpy as np
import platform

from SimEngineMBD.example_models.double_pendulum import run_double_pendulum


# check operating system, file loc different
if platform.system() == 'Darwin':
    fileloc = "/Users/luning/Manuscripts/Conference/2022/IROS_Half_Implicit/images/"

filename = "double_pendulum_energy.png"

# flag for saving pic directly in image folder
saveFig = True


step_size = 1e-3

# run rA and rA_half
forms = ['rA_half', 'rA']
tol = 1e-10
tols = {"rA_half": tol, "rA": tol/step_size**2}
labels = {"rA_half": "half implicit", "rA": "fully implicit"}
colors = {"rA_half": "blue", "rA": "red"}

model_fn = run_double_pendulum

num_bodies = 2
t_end = 10
t = np.arange(0, t_end, step_size)


L = 2                                   # [m] - length of the bar
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar

gravity = 9.81

bar_L1 = 2*L
bar_L2 = L
pend_len = [bar_L1, bar_L2]

Ms = []
Js = []

Fontsize = 60
LineWidth = 10
MarkerSize = 40
plt.rc('font', size=Fontsize)
plt.rc('legend', fontsize=Fontsize*0.9)
plt.rc('figure', titlesize=Fontsize*0.9)
plt.rc('lines', linewidth=LineWidth)
plt.rc('lines', markersize=MarkerSize)
plt.rc('axes', linewidth=LineWidth*0.5)



for i in range(0, 2):

    V = pend_len[i] * w**2                 # [m^3] - bar volume
    m = ρ * V

    J_xx = 1/6 * m * w**2
    J_yz = 1/12 * m * (w**2 + pend_len[i]**2)
    J = np.diag([J_xx, J_yz, J_yz])        # [kg*m^2] - Inertia tensor of bar
    
    Ms.append(m) 
    Js.append(J)

fig, ax = plt.subplots(figsize=(35, 15))

for form in forms:
    pos, velo, _, omg, _, t_grid = model_fn(['--form', form, '--mode', 'dynamics', '--tol', str(tols[form]), '--step_size', str(step_size), '-t', str(t_end)])
    KE = np.zeros((1,len(t_grid)))
    PE = np.zeros((1,len(t_grid)))
    init_z = [0, -1]
    for i in range(0, len(t_grid)):
        body0_ke = 0.5 * Ms[0] * velo[0,:,i].T @ velo[0,:,i] + 0.5 * omg[0,:,i].T @ Js[0] @ omg[0,:,i]
        body1_ke = 0.5 * Ms[1] * velo[1,:,i].T @ velo[1,:,i] + 0.5 * omg[1,:,i].T @ Js[1] @ omg[1,:,i]
        KE[0,i] = body0_ke + body1_ke
        PE[0,i] = Ms[0]*gravity*(pos[0,2,i]- init_z[0]) + Ms[1]*gravity*(pos[1,2,i]- init_z[1])
    
    total = KE+PE
    ax.plot(t_grid, total[0,:], linestyle='-', color=colors[form], label=labels[form])
    ax.set(xlabel='time (sec)', ylabel=' KE+PE (J)')
    ax.legend(loc='center right')
    ax.grid(linestyle='--', linewidth=3)
    

if saveFig == True:
    plt.savefig(fileloc + filename)