import pychrono as chrono
import pychrono.vsg as vsg
import pychrono.vehicle as veh
import pychrono.fsi as fsi

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import simulation.Robots as Robots
import time
import numpy as np

def CreateFSIQuadrupedBodies(robot, terrain):
    """
    Add quadruped robot feet and calf bodies as FSI solids to the CRM terrain.
    
    Args:
        robot: Quadruped robot object (A1Robot or Go2Robot)
        terrain: CRMTerrain object
    """
    # Create geometry for feet (smaller spheres/cylinders)
    foot_geometry = chrono.ChBodyGeometry()
    foot_radius = 0.025  # 2cm radius for feet
    foot_sphere = chrono.SphereShape(chrono.VNULL, foot_radius)
    foot_geometry.coll_spheres.append(foot_sphere)
    
    # Create geometry for calves (cylindrical shape)
    calf_geometry = chrono.ChBodyGeometry()
    calf_radius = 0.02  # 1.5cm radius for calf
    calf_height = 0.2   # 20cm height (typical calf length)
    calf_cylinder = chrono.CylinderShape(chrono.ChVector3d(0, 0, 0), chrono.QUNIT, calf_radius, calf_height)
    calf_geometry.coll_cylinders.append(calf_cylinder)
    
    # Add foot bodies as FSI solids
    for foot_name in robot.foot_bodies_list:
        foot_body = robot.robot.GetChBody(foot_name)
        print(f"Adding foot body: {foot_name}")
        try:
            num_bce = terrain.AddRigidBody(foot_body, foot_geometry, False)
            print(f"  Added {num_bce} BCE markers on {foot_name}")
        except Exception as e:
            print(f"  Error adding foot {foot_name}: {e}")
    
    # Add calf bodies as FSI solids
    for calf_name in robot.calf_bodies_list:
        calf_body = robot.robot.GetChBody(calf_name)
        print(f"Adding calf body: {calf_name}")
        try:
            num_bce = terrain.AddRigidBody(calf_body, calf_geometry, False)
            print(f"  Added {num_bce} BCE markers on {calf_name}")
        except Exception as e:
            print(f"  Error adding calf {calf_name}: {e}")

def main():
    # Set Chrono data path
    chrono.SetChronoDataPath(chrono.GetChronoDataPath())
    veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')
    
    # Create a Chrono system
    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    # Create quadruped robot
    init_state = chrono.ChFramed(chrono.ChVector3d(0, 0, 0.9), chrono.QuatFromAngleZ(0.0))
    robot_name = "go2"  # or "a1"
    
    if robot_name == "a1":
        robot = Robots.A1Robot(system, init_state)
        step_size = 5e-4  # Smaller timestep for FSI
    elif robot_name == "go2":
        robot = Robots.Go2Robot(system, init_state, vis_engine='vsg')
        step_size = 5e-4  # Smaller timestep for FSI
    
    robot.print_motor_list()
    
    # Set solver for FSI compatibility
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(150)
    
    # Problem settings
    target_speed = 1.0
    tend = 10.0
    verbose = True
    
    # Visualization settings
    render = True
    render_fps = 50
    visualization_sph = True
    visualization_bndry_bce = False
    visualization_rigid_bce = True
    
    # CRM material properties
    density = 1700
    cohesion = 5e3
    friction = 0.8
    youngs_modulus = 1e6
    poisson_ratio = 0.3
    
    # CRM active box dimension
    active_box_hdim = 1.0
    settling_time = 0
    
    # Set SPH spacing
    spacing = 0.02
    
    # Terrain dimensions
    terrain_length = 8
    terrain_width = 4
    terrain_height = 0.3
    
    # ----------------------
    # Create the CRM terrain
    # ----------------------
    terrain = veh.CRMTerrain(system, spacing)
    sysFSI = terrain.GetSystemFSI()
    terrain.SetVerbose(verbose)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    terrain.SetStepSizeCFD(step_size)
    
    # Set SPH parameters and soil material properties
    mat_props = fsi.ElasticMaterialProperties()
    mat_props.density = density
    mat_props.Young_modulus = youngs_modulus
    mat_props.Poisson_ratio = poisson_ratio
    mat_props.mu_I0 = 0.04
    mat_props.mu_fric_s = friction
    mat_props.mu_fric_2 = friction
    mat_props.average_diam = 0.005
    mat_props.cohesion_coeff = cohesion
    terrain.SetElasticSPH(mat_props)
    
    # Set SPH solver parameters
    sph_params = fsi.SPHParameters()
    sph_params.initial_spacing = spacing
    sph_params.d0_multiplier = 1
    sph_params.kernel_threshold = 0.8
    sph_params.artificial_viscosity = 0.5
    sph_params.consistent_gradient_discretization = False
    sph_params.consistent_laplacian_discretization = False
    sph_params.viscosity_type = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    sph_params.boundary_type = fsi.BoundaryMethod_ADAMI
    terrain.SetSPHParameters(sph_params)
    
    # Set output level
    terrain.SetOutputLevel(fsi.OutputLevel_STATE)
    
    # Add robot feet and calves as FSI solids
    print("Adding robot bodies as FSI solids...")
    CreateFSIQuadrupedBodies(robot, terrain)
    
    terrain.SetActiveDomain(chrono.ChVector3d(active_box_hdim, active_box_hdim, active_box_hdim))
    terrain.SetActiveDomainDelay(settling_time)
    
    # Construct rectangular terrain patch
    print("Create terrain...")
    terrain.Construct(chrono.ChVector3d(terrain_length, terrain_width, terrain_height), 
                     chrono.ChVector3d(0, 0, 0), 
                     (fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS))
    
    # Initialize the terrain system
    terrain.Initialize()
    
    aabb = terrain.GetSPHBoundingBox()
    print(f"  SPH particles:     {terrain.GetNumSPHParticles()}")
    print(f"  Bndry BCE markers: {terrain.GetNumBoundaryBCEMarkers()}")
    print(f"  SPH AABB:          {aabb.min}   {aabb.max}")
    
    # Robot control - simple walking pattern
    action_angle = np.array([0, -1.0, 1.5, 0, -1.0, 1.5, 0, -0.8, 1.5, 0, -0.8, 1.5])
    robot.actuate(action_angle)
    
    # Create run-time visualization
    vis = None
    if render:
        # FSI plugin
        col_callback = fsi.ParticleHeightColorCallback(aabb.min.z, aabb.max.z)
        visFSI = fsi.ChFsiVisualizationVSG(sysFSI)
        visFSI.EnableFluidMarkers(visualization_sph)
        visFSI.EnableBoundaryMarkers(visualization_bndry_bce)
        visFSI.EnableRigidBodyMarkers(visualization_rigid_bce)
        visFSI.SetSPHColorCallback(col_callback, chrono.ChColormap.Type_BROWN)

        # Create VSG visualization system
        vis = vsg.ChVisualSystemVSG()
        vis.AttachSystem(system)
        vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        vis.SetWindowSize(1280, 720)
        vis.SetWindowTitle(f"{robot_name.upper()} robot on CRM deformable terrain")
        vis.EnableSkyBox()
        vis.AddCamera(chrono.ChVector3d(5, 5, 5), chrono.ChVector3d(0, 0, 0))
        vis.SetLightIntensity(1.0)
        vis.AttachPlugin(visFSI)
        vis.Initialize()
    
    # ---------------
    # Simulation loop
    # ---------------
    render_step_size = 1/render_fps
    sim_step = 0
    render_steps = int(render_step_size / step_size)
    
    start_time = time.time()
    current_time = 0
    render_frame = 0
    
    print("Start simulation...")
    while current_time < tend:
        # Run-time visualization
        if render and current_time >= render_frame / render_fps:
            if not vis.Run():
                break
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            render_frame += 1
            
        # Print robot state
        if (sim_step % render_steps == 0):
            joint_pos = robot.get_joint_pos()
            formatted_joint_pos = " ".join(f"{val:.2f}" for val in joint_pos)
            print(f"t={current_time:.2f} joint_pos: [{formatted_joint_pos}]")
            
            contact_forces = robot.get_contact_force()
            formatted_contact = " ".join(f"{val:.1f}" for val in contact_forces)
            print(f"t={current_time:.2f} contact_forces: [{formatted_contact}]")

        # # Update robot actuation (simple example - could be more sophisticated)
        # if current_time > 2.0:  # Let robot settle first
        #     # Simple walking pattern - could implement more complex gait here
        #     walking_phase = (current_time - 2.0) * 2.0  # Walking frequency
        #     action_angle = np.array([
        #         0, -1.0 + 0.3*np.sin(walking_phase), 1.5 + 0.5*np.sin(walking_phase),
        #         0, -1.0 + 0.3*np.sin(walking_phase + np.pi), 1.5 + 0.5*np.sin(walking_phase + np.pi),
        #         0, -0.8 + 0.3*np.sin(walking_phase + np.pi), 1.5 + 0.5*np.sin(walking_phase + np.pi),
        #         0, -0.8 + 0.3*np.sin(walking_phase), 1.5 + 0.5*np.sin(walking_phase)
        #     ])
        #     robot.actuate(action_angle)
        
        # Advance system state
        system.DoStepDynamics(step_size)
        terrain.Advance(step_size)
        
        current_time += step_size
        sim_step += 1
    
    real_time = time.time() - start_time
    print(f"Simulation time: {current_time}")
    print(f"Real time: {real_time}")
    print(f"RTF: {real_time / current_time}")

if __name__ == "__main__":
    main() 