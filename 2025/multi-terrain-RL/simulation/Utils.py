import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from simulation import Robots
import pychrono as chrono
import numpy as np
import matplotlib.pyplot as plt
def add_dir_arrow(system: chrono.ChSystemSMC, robot: Robots.A1Robot, target_vel: np.ndarray):
    # # create a arrow body
    arrowmesh = project_root + "/data/assets/utils/arrow.obj"
    print(arrowmesh)
    arrow = chrono.ChBody()
    arrow.SetPos(robot.get_base_pos() + chrono.ChVector3d(0, 0, 0.2))
    target_vel_dir = np.atan2(target_vel[1], target_vel[0])
    arrow.SetRot(chrono.QuatFromAngleZ(target_vel_dir))
    myasset = chrono.ChVisualShapeModelFile()
    myasset.SetFilename(arrowmesh)
    arrow.AddVisualShape(myasset)
    system.AddBody(arrow)
    link_lock = chrono.ChLinkLockLock()
    link_lock.Initialize(robot.robot.GetChBody("trunk"),arrow, chrono.ChFramed(chrono.ChVector3d(0, 0, 0),chrono.QUNIT))
    system.Add(link_lock)



