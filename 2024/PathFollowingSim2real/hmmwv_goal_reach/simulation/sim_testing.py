# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2014 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =============================================================================
from stable_baselines3 import PPO

import pychrono.core as chrono
import pychrono.irrlicht as irr
import pychrono.vehicle as veh
import math
import numpy as np
import csv
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from util.help_fun import relative_state_to_goal
import random
import argparse

import pychrono.postprocess as postprocess
"""
!!!! Set this path before running the demo!
"""
chrono.SetChronoDataPath(chrono.GetChronoDataPath())
veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
chassis_vis_type = chrono.VisualizationType_PRIMITIVES
suspension_vis_type = chrono.VisualizationType_PRIMITIVES
# Parse command line arguments
parser = argparse.ArgumentParser(description='ART Simulation')
parser.add_argument('ModelType',default='1', type=str, help='Type of Model Used for Training [1, 2, 3]')
parser.add_argument('exp_ind', type=int, default=0, help='index of the experiment')

args = parser.parse_args()
modeltype = args.ModelType
exp_ind = args.exp_ind


steering_vis_type = chrono.VisualizationType_PRIMITIVES
wheel_vis_type = chrono.VisualizationType_PRIMITIVES

# Collision type for chassis (PRIMITIVES, MESH, or NONE)
chassis_collision_type = veh.CollisionType_NONE

# Type of tire model (RIGID, TMEASY)
tire_model = veh.TireModelType_TMEASY

# Rigid terrain
# terrain_model = veh.RigidTerrain.BOX
terrainHeight = 0      # terrain height
terrainLength = 500.0  # size in X direction
terrainWidth = 500.0   # size in Y direction

# Poon chassis tracked by the camera
trackPoint = chrono.ChVector3d(3.0, 0.0, 0.2)


contact_vis = True

# Simulation step sizes
step_size = 1e-3
tire_step_size = step_size

# Time interval between two render frames
render_step_size = 1.0 / 20  # FPS = 50
# --------------
# Create systems
# --------------

# ref_traj = np.genfromtxt(project_root+'/data/ref_circle_10.csv', delimiter=',')
# Initial vehicle location and orientation
# initLoc = chrono.ChVector3d(np.random.uniform(-1,1), np.random.uniform(-1,1) , 0.5)
initLoc = chrono.ChVector3d(0.0,0, 0.9)
# initRot = chrono.QuatFromAngleZ(np.random.uniform(0, 2 * np.pi))
initRot = chrono.QuatFromAngleZ(0)
contact_method = chrono.ChContactMethod_NSC
contact_material = chrono.ChContactMaterialNSC()

# Create the ARTcar vehicle, set parameters, and initialize
hmmwv = veh.HMMWV_Full()
hmmwv.SetContactMethod(contact_method)
hmmwv.SetInitPosition(chrono.ChCoordsysd(initLoc, initRot))
hmmwv.SetEngineType(veh.EngineModelType_SHAFTS)
hmmwv.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
hmmwv.SetDriveType(veh.DrivelineTypeWV_AWD)
hmmwv.SetTireType(tire_model)
hmmwv.Initialize()

hmmwv.SetChassisVisualizationType(chrono.VisualizationType_PRIMITIVES)
hmmwv.SetSuspensionVisualizationType(chrono.VisualizationType_PRIMITIVES)
hmmwv.SetSteeringVisualizationType(chrono.VisualizationType_PRIMITIVES)
hmmwv.SetWheelVisualizationType(chrono.VisualizationType_PRIMITIVES)
hmmwv.SetTireVisualizationType(chrono.VisualizationType_PRIMITIVES)
hmmwv.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
print("Vehicle mass: ", hmmwv.GetVehicle().GetMass())

system = hmmwv.GetSystem()

# Create the terrain
terrain = veh.RigidTerrain(hmmwv.GetSystem())
patch = terrain.AddPatch(contact_material, 
    chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0), chrono.QUNIT), 
    terrainLength, terrainWidth)

patch.SetTexture(project_root+'/data/environment/grass.jpg', 10, 10)
#patch.SetColor(chrono.ChColor(0.8, 0.8, 0.5))
terrain.Initialize()


# create goal point
# r = np.random.uniform(30.0, 40.0)
# theta = np.random.uniform(0, 2 * np.pi)
# goal_x,goal_y = [r * np.cos(theta), r * np.sin(theta)]
# goal_x,goal_y = [20, 20]
goal_list = np.loadtxt(project_root + '/data/traj/goal_points.txt', skiprows=1, delimiter=" ")
goal_x, goal_y = goal_list[exp_ind]
print(f"Goal point: ({goal_x}, {goal_y})")
goal_body =  chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, 0.75, 0.125, 40, True, False, contact_material)
goal_body.SetPos(chrono.ChVector3d(goal_x, goal_y, 0.1))
goal_body.SetFixed(True)
shape = goal_body.GetVisualModel().GetShape(0)
vis_mat_path = chrono.ChVisualMaterial()
vis_mat_path.SetDiffuseColor(chrono.ChColor(0.0, 1.0, 0.0))
shape.AddMaterial(vis_mat_path)
system.Add(goal_body)

# Load the trained model
model_path = project_root + '/data/rl_model/ppo_' + modeltype
# model_path = project_root + '/data/rl_models/ppo_M2_2'
model = PPO.load(model_path)
# -------------------------------------
# Create the vehicle Irrlicht interface
# Create the driver system
# -------------------------------------

vis_on = False
if vis_on:
    # SetPathVisualization(ref_traj, hmmwv.GetSystem())
    vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle('dart')
    vis.SetWindowSize(1280, 1024)
    vis.SetChaseCamera(trackPoint, 6.0, 1.5)
    vis.Initialize()
    vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    vis.AddLightDirectional()
    vis.AddSkyBox()
    vis.AttachVehicle(hmmwv.GetVehicle())


driver = veh.ChDriver(hmmwv.GetVehicle())
driver.Initialize()


# ---------------
# Simulation loop
# ---------------

# output vehicle mass
print( "VEHICLE MASS: ",  hmmwv.GetVehicle().GetMass())

# Number of simulation steps between miscellaneous events
render_steps = math.ceil(render_step_size / step_size)

controlstepsize = 1.0 / 10  # control update rate
control_step = math.ceil(controlstepsize / step_size)
controls = [0,0]
# Initialize simulation frame counter s
realtime_timer = chrono.ChRealtimeStepTimer()
step_number = 0
render_frame = 0
saveframe = False
prev_steering = 0.0

# set up logging info
lateral_error = []
loggingdata = []

time = 0.0
end_time = 100.0
traj_save_file = project_root+'/data/traj/'+modeltype+'_'+str(exp_ind)+'.csv'
loggingdata = []

if saveframe:
    blender_exporter = postprocess.ChBlender(hmmwv.GetSystem())
    blender_exporter.SetBlenderUp_is_ChronoZ()
    blender_exporter.SetBasePath(project_root + '/data/blender/new_blender/')
    
    # blender_exporter.SetCamera(chrono.ChVector3d(13, -4, 3), chrono.ChVector3d(13, 5, 1), 50)
    blender_cam = chrono.ChCamera()
    blender_cam.SetAngle(90)
    blender_cam.SetPosition(chrono.ChVector3d(-2, -1, 1.0))
    blender_cam.SetAimPoint(chrono.ChVector3d(4, 1, 0.2))
    blender_cam.SetUpVector(chrono.ChVector3d(0, 0, 1))
    hmmwv.GetChassisBody().AddCamera(blender_cam)
    blender_exporter.SetLight(chrono.ChVector3d(20, 0, 3), chrono.ChColor(1, 1, 1), True)
    blender_exporter.AddAll()
    blender_exporter.ExportScript()

#while vis.Run() :
while time < end_time:
    time = hmmwv.GetSystem().GetChTime()
    if vis_on:
        # Render scene and output POV-Ray data
        if (step_number % render_steps == 0) :
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            if saveframe:
                # filename = './IMG/img_' + str(render_frame) +'.jpg' 
                # vis.WriteImageToFile(filename)
                blender_exporter.ExportData()
            render_frame += 1
    # # control update
    if (step_number % control_step == 0) :
        veh_x = hmmwv.GetVehicle().GetPos().x
        veh_y = hmmwv.GetVehicle().GetPos().y
        veh_theta = hmmwv.GetVehicle().GetRot().GetCardanAnglesXYZ().z
        veh_v = hmmwv.GetVehicle().GetSpeed()

        dist_to_end = np.sqrt((veh_x - goal_x)**2 + (veh_y - goal_y)**2)
        
        with open (traj_save_file,'a', encoding='UTF8') as savefile:
            save_writer = csv.writer(savefile, quoting=csv.QUOTE_NONE, escapechar=' ')
            save_writer.writerow([time, veh_x, veh_y, veh_theta, veh_v, goal_x, goal_y])
            savefile.close()


        if dist_to_end < 3.0:
            print("Reached the end of the path")
            driver.SetThrottle(0)
            driver.SetSteering(0)
            driver.SetBraking(1)
            print(f"reach the goal at time: {time}")
            break

        else: 
            veh_state = [veh_x, veh_y, veh_theta, veh_v]
            # callculate observation
            obs = relative_state_to_goal(veh_state, [goal_x, goal_y])
            
            # get the control from trained rl model:
            controls, _ = model.predict(obs, deterministic=True)
            throttle, steering = float(controls[0]), float(controls[1])  # scale steering to [-0.5, 0.5]
            
            # check if the steeering is smooth
            delta_steering = float(steering) - prev_steering
            if abs(delta_steering)> 0.1:
                prev_steering = prev_steering + 0.1 * delta_steering/abs(delta_steering)
            else:
                prev_steering = float(steering)
            
            if time > 1.0:
                driver.SetSteering(prev_steering)
                driver.SetThrottle(throttle)
                driver.SetBraking(0.0)
            else:
                driver.SetSteering(0.0)
                driver.SetThrottle(0.0)
                driver.SetBraking(1.0)
        

    
    
    driver_inputs = driver.GetInputs()

    # Update modules (process inputs from other modules)
    driver.Synchronize(time)
    terrain.Synchronize(time)
    hmmwv.Synchronize(time, driver_inputs, terrain)
    if vis_on:
        vis.Synchronize(time, driver_inputs)

    # Advance simulation for one timestep for all modules
    driver.Advance(step_size)
    terrain.Advance(step_size)
    hmmwv.Advance(step_size)
    if vis_on:
        vis.Advance(step_size)

    # Increment frame number
    step_number += 1

    # # Spin in place for real time to catch up
    # realtime_timer.Spin(step_size)