import numpy as np
import matplotlib.pyplot as plt

from SimEngineMBD.utils.physics import I3
from SimEngineMBD.example_models.double_pendulum import setup_double_pendulum

sys, params = setup_double_pendulum(['--form', 'rA', '--mode', 'dynamics', '--end_time', '20'])
sys.initialize()

t_steps = int(params.t_end / params.h)
t_grid = np.linspace(0, params.t_end, t_steps, endpoint=True)

A0 = np.zeros((t_steps, 3, 3))
A1 = np.zeros((t_steps, 3, 3))

for i, t in enumerate(t_grid):
    sys.do_step(i, t)

    A0[i] = sys.bodies[0].A
    A1[i] = sys.bodies[1].A

A0_norms = [np.linalg.norm(A.T @ A - I3, ord='fro') for A in A0]
A1_norms = [np.linalg.norm(A.T @ A - I3, ord='fro') for A in A1]

plt.rcParams.update({'font.size': 30})
_, ax = plt.subplots()

ax.plot(t_grid, A0_norms, label='Body 0', markersize=15, linewidth=7)
ax.plot(t_grid, A1_norms, label='Body 1', markersize=15, linewidth=7)

ax.legend(loc='best', prop={'size': 24})
plt.rcParams.update({'text.usetex': True})
ax.set(xlabel='Step Size', ylabel='Frobenius Norm of $A^T A - I_3$', title='Rotation Matrix Non-Orthogonality')

plt.gcf().set_size_inches(20, 12)
plt.savefig('rotation-mat-ortho', dpi=100)