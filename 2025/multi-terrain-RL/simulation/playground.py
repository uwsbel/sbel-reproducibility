import pychrono as chrono
import pychrono.vsg as vsg

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import simulation.Robots as Robots
import time
import numpy as np
def main():
    # Create a Chrono system
    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(60)

    go2 = Robots.Go2Robot(system,vis_engine='vsg')
    go2.print_motor_list()

    # create contact material
    mat = chrono.ChContactMaterialSMC()
    mat.SetRestitution(0.01)
    mat.SetFriction(0.9)
    mat.SetGn(60.0)
    mat.SetKn(2e5)

    # Create a floor body
    floor = chrono.ChBodyEasyBox(10, 10, 0.1, 1000, True, True, mat)
    floor.SetPos(chrono.ChVector3d(0, 0, 0.0))
    floor.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/concrete.jpg"), 10, 10)
    floor.SetFixed(True)
    system.AddBody(floor)
    
    action_angle = np.array([0,-1,1.5]*4)
    
    vis_enabled = True

    if vis_enabled:
        # Create the irrlicht visualization
        vis = vsg.ChVisualSystemVSG()
        vis.AttachSystem(system)
        vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        vis.SetWindowSize(1024, 768)
        vis.SetWindowTitle("A1 robot Demo")
        vis.EnableSkyBox()
        vis.AddCamera(chrono.ChVector3d(1, 1, 2),chrono.ChVector3d(0, 0, 0))
        vis.SetLightIntensity(1.0)
        vis.Initialize()

    # Simulation loop
    step_size = 1e-3
    render_step_size = 1/50
    sim_step = 0
    render_steps = int(render_step_size / step_size)
    control_step_size = 1/50
    control_steps = int(control_step_size / step_size)

    start_time = time.time()
    end_sim_time = 20

    # while (system.GetChTime() < end_sim_time):
    while vis.Run():
        sim_time = system.GetChTime()

        if (sim_step % render_steps == 0) and vis_enabled:
            vis.BeginScene()
            vis.Render()
            vis.EndScene()

        system.DoStepDynamics(step_size)
        sim_step += 1
    else:
        real_time = time.time() - start_time
        print(f"Simulation time: {sim_time}")
        print(f"Real time: {real_time}")
        print(f"RTF: {real_time / sim_time}")

if __name__ == "__main__":
    main()