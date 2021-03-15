import numpy as np
from physics import R, Y_AXIS, X_AXIS, Z_AXIS

# Computes values associated with Haug's four-link mechanism
#   an (safer?) alternative to hand-calculations

radius = 2

ptA = np.zeros((3, 1))
ptB = np.array([[0, 0, radius]]).T

D_x = -4
D_y = -8.5

ptD = np.array([[D_x, D_y, 0]]).T

C_x = -7.5
C_len = 2*3.7
C_z = np.sqrt(C_len**2 - (C_x - D_x)**2)

ptC = np.array([[C_x, D_y, C_z]]).T

r2 = ptC + (ptB - ptC) / 2
r3 = ptD + (ptC - ptD) / 2

θ3 = np.arccos((D_x - C_x) / C_len)
A3 = R(Z_AXIS, -np.pi/2) @ R(X_AXIS, θ3)

y2 = ptC - ptB
θ2z = np.arccos((Y_AXIS.T @ y2) / np.linalg.norm(y2))
θ2x = np.arccos((Z_AXIS.T @ y2) / np.linalg.norm(y2))

A2 = R(Z_AXIS, -θ2z) @ R(X_AXIS, -(np.pi/2 - θ2x))
