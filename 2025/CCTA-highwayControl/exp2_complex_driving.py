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

import pychrono.core as chrono
import pychrono.irrlicht as irr
import pychrono.vehicle as veh
# import pychrono.sensor as sens

import math
import numpy as np
from pandas import read_csv
import torch
from casadi import *
import time as tm
from controller import error_state, vir_veh_controller, mpc_wpts_solver_sedan3,mpc_wpts_solver_sedan, mpc_wpts_solver_sedan2, FindMinDistInd, CPWarningSystem, mpc_wpts_solver_sedan_dob
from merge_scenario import merging_scenario,parallel_scenario,serial_scenario
from rom_vehicle import simplifiedVehModel
import pygame

from LSTMHelper import GetGridInd, FindNbrs
import csv
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSETest
from torch.utils.data import DataLoader
import time
# import pandas as pd
import matplotlib.pyplot as plt
from DobCBFHelper import xdot_dob, DOBCBF_ACC, DOBCBF_ACC_switch, xdot_dob_switch, xdot_dob_poly, DOBCBF_dbeta, xdot_dob_dbeta

"""
!!!! Set this path before running the demo!
"""
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 5
args['grid_size'] = (13,3)
args['input_embedding_size'] = 32
args['train_flag'] = False
metric = 'rmse'
dobpara = {}

# Engine parameters
tau0 = 100.
omega0 = 1200.
c0 = 0.01
c1 = 0.02
Rwheel = 0.3
gamma = 1/3
Iwheel = 0.6
L =  4.52
delta = 0.626671  # 35.9 degree
flagcbf = False
# L =  2.5
# delta = 0.667  # 35.9 degree

brake_effect_coef = 0.9

gainx = 10.
gainy = 10.
vp_current = 0.
Tp_current = 0.
alpha = 1.0
beta = 0.
ref_beta = 0.
dbeta = 0.

dobpara['l0x'] = 2*gainx
dobpara['l1x'] = gainx**2
dobpara['l0y'] = gainy
dobpara['l1y'] = gainy**2

dob_mode = True
is_manual = False
DelayCompMode = True
WarningSystemMode = True
enable_joystick = False
CBFMode = True
td = 10 # time-delay  

cbfflavor = 'balance' # 'balance' or 'throttleonly'

IfSave = False

scene_prefix = 'complex'

if dob_mode:
    dob_prefix = 'dobON' 
else:
    dob_prefix = 'dobOFF' 
if DelayCompMode:
    DC_prefix = 'DCON' 
else:
    DC_prefix = 'DCOFF'

a_actual_list = []
a_est_list = []
a_model_list = []
Vcontrol_seq = []
Tcontrol_seq = []


flagcbf_list = []
hvalue_list = []

alpha_list = [] # store original input, input after delay compensation, uCBF
beta_list = [] # store original input, input after delay compensation, uCBF



dir_hvalue = scene_prefix + '_hvalue_' + dob_prefix + '_' + DC_prefix + '_' + cbfflavor + 'td' + f'{td}' + '.csv'
dir_flagcbf = scene_prefix + '_flagcbf_' + dob_prefix + '_' + DC_prefix + '_' + cbfflavor + 'td' + f'{td}' + '.csv'
dir_acc_value = scene_prefix + '_acc_value_' + dob_prefix + '_' + DC_prefix + '_' + cbfflavor + 'td' + f'{td}' + '.csv'
dir_alpha_value = scene_prefix + '_alpha_value_' + dob_prefix + '_' + DC_prefix + '_' + cbfflavor + 'td' + f'{td}' + '.csv'
dir_beta_value = scene_prefix + '_beta_value_' + dob_prefix + '_' + DC_prefix + '_' + cbfflavor + 'td' + f'{td}' + '.csv'
dir_veh_state = scene_prefix + '_veh_state_' + dob_prefix + '_' + DC_prefix + '_' + cbfflavor + 'td' + f'{td}' + '.npy'
    
# Time interval between two render frames
render_step_size = 1.0 / 50  # FPS = 50
control_step_size = 1.0 / 50 # FPS to run control = 20
mpc_step_size = 1.0 / 10
low_control_step_size = 1.0 / 50 # FPS to low level control
sur_control_step_size = 1.0 / 200
# low_control_step_size = 1.0 / 200 # FPS to low level control
sample_step_size = 1.0 / 50 * 3
cbf_step_size = 1.0 / 40 # FPS to low level control


# td = 0 # time-delay 
if td == 0:
    DelayCompMode = False
else:
    # k1 = 0.3 # form [-1, 1]
    k1 = 0.6 # form [-1, 1]
    # k2 = (math.acos(-k1) * math.sqrt(1 - k1**2)) / td 
    k2 = 2

td_max = td + 10 # maximum delay 

dobpara['av'] = 10.
dobpara['ata'] = 100.

ctrlpara = {}

ctrlpara['l1cbf'] = 3.0 # good?
ctrlpara['kblf'] = 1/control_step_size
ctrlpara['l2cbf'] = 5.0 # good?

violation_list = [] # safety distance

p_dob = 0. # initial state of EDOB
hdxyvta = 0.

eps = 0.09
weight1 = 1/10/10
weight2 = 1/4/4
bias = 5.0

ro = 1
r_precpt = 20 
f0 = ro * ro # Safety score threshold.

f0warning = 2 # Safety score threshold.

Vp_seq = [0. for i in range(td+1)]
Tp_seq = [0. for i in range(td+1)]

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('data/sta_lstm_10272020.tar'))
if args['use_cuda']:
    net = net.cuda()


if WarningSystemMode:
    dcp = read_csv('data/training_data/CP_data.csv', header=None)
    CP_data = dcp.values
    CP_data_f0 = [tup for tup in CP_data if tup[0]<f0warning]

# sc = lambda x,y: np.linalg.norm(x-y)
sc = lambda x,y,Q,Rmat,bias: np.dot((x-y) @ Rmat.T - np.array([bias, 0.]), np.dot(Q, (x-y) @ Rmat.T - np.array([bias, 0.])))
sc_warning = lambda x,y,Q,Rmat: np.dot((x-y) @ Rmat.T, np.dot(Q, (x-y) @ Rmat.T))


Q = np.array([[weight1, 0.],[0.,weight2]])
Qwarning = np.array([[1/20/20, 0.],[0.,1.]])

warningsignal = 0



chrono.SetChronoDataPath(chrono.GetChronoDataPath())
veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

import sys,os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Initial vehicle location and orientation
initLoc = chrono.ChVector3d(-200, -61, 0.3)
initRot = chrono.QuatFromAngleZ(-0.8)

# initRot = chrono.ChQuaterniond(1, 0, 0, 0)

# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
vis_type = veh.VisualizationType_MESH
# Collision type for chassis (PRIMITIVES, MESH, or NONE)
chassis_collision_type = veh.CollisionType_NONE

# Type of tire model (RIGID, TMEASY)
tire_model = veh.TireModelType_TMEASY

# Rigid terrain
# terrain_model = veh.RigidTerrain.BOX
terrainHeight = 0      # terrain height
terrainLength = 1200.0  # size in X direction
terrainWidth = 1200.0   # size in Y direction

# Poon chassis tracked by the camera
trackPoint = chrono.ChVector3d(-5.0, 0.0, 1.8)

# Contact method
contact_method = chrono.ChContactMethod_NSC
contact_vis = False

# Simulation step sizes
step_size = 1e-3
tire_step_size = step_size

# =============================================================================

# --------------
# Create systems
# --------------
# Create the sedan vehicle, set parameters, and initialize
sedan = veh.BMW_E90()
sedan.SetContactMethod(contact_method)
sedan.SetChassisCollisionType(chassis_collision_type)
sedan.SetChassisFixed(False)
sedan.SetInitPosition(chrono.ChCoordsysd(initLoc, initRot))
sedan.SetTireType(tire_model)
sedan.SetTireStepSize(tire_step_size)


sedan.Initialize()

sedan.SetChassisVisualizationType(vis_type)
sedan.SetSuspensionVisualizationType(vis_type)
sedan.SetSteeringVisualizationType(vis_type)
sedan.SetWheelVisualizationType(vis_type)
sedan.SetTireVisualizationType(vis_type)
with open(f'data/training_data/{scene_prefix}_data_warning_test_F{f0warning}_eps{eps}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
with open(f'data/training_data/{scene_prefix}_data_warning_test_eps{eps}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
# indicator:
# initialize the ball to represent the ball from right hand merging
# green_cyl = chrono.ChBodyEasyCylinder(chrono.ChAxis_X, 0.4, 0.1, 1000, True, False)
# green_cyl = chrono.ChBodyEasySphere(0.3, 10, True, False)
green_cyl = chrono.ChBodyEasyBox(0.5,0.5,0.5,10,True,False)
# red_cyl = chrono.ChBodyEasyCylinder(chrono.ChAxis_X, 0.4, 0.1, 1000, True, False)
# red_cyl = chrono.ChBodyEasySphere(0.3, 10, True, False)
red_cyl = chrono.ChBodyEasyBox(0.5,0.5,0.5,10,True,False)
# define colors
red_color =chrono.ChColor(1,0,0)
green_color = chrono.ChColor(0,1,0)


vis_mat_g = chrono.ChVisualMaterial()
vis_mat_g.SetDiffuseColor(green_color)
vis_mat_r = chrono.ChVisualMaterial()
vis_mat_r.SetDiffuseColor(red_color)

green_shape = green_cyl.GetVisualModel().GetShape(0)
green_shape.AddMaterial(vis_mat_g)

red_shape = red_cyl.GetVisualModel().GetShape(0)
red_shape.AddMaterial(vis_mat_r)

if WarningSystemMode:
    green_cyl.SetPos(chrono.ChVector3d(0,0,10000))
    green_cyl.SetFixed(False)
    red_cyl.SetPos(chrono.ChVector3d(0,0,10000))
    red_cyl.SetFixed(False)

    sedan.GetSystem().Add(green_cyl)
    sedan.GetSystem().Add(red_cyl)
    def indicator(state='safe'):
        if state == 'safe':
            green_cyl.SetPos(sedan.GetChassis().GetPos() + chrono.ChVector3d(-0.5,0,1.5))
        else:
            red_cyl.SetPos(sedan.GetChassis().GetPos() + chrono.ChVector3d(-0.5,0,1.5))


#### indicator_true:
# initialize the ball to represent the ball from right hand merging
# green_cylt = chrono.ChBodyEasyCylinder(chrono.ChAxis_X, 0.4, 0.1, 1000, True, False)
# red_cylt = chrono.ChBodyEasyCylinder(chrono.ChAxis_X, 0.4, 0.1, 1000, True, False)
green_cylt = chrono.ChBodyEasySphere(0.3, 10, True, False)
red_cylt = chrono.ChBodyEasySphere(0.3, 10, True, False)
# define colors
red_color =chrono.ChColor(1,0,0)
green_color = chrono.ChColor(0,1,0)


vis_mat_gt = chrono.ChVisualMaterial()
vis_mat_gt.SetDiffuseColor(green_color)
vis_mat_rt = chrono.ChVisualMaterial()
vis_mat_rt.SetDiffuseColor(red_color)

green_shapet = green_cylt.GetVisualModel().GetShape(0)
green_shapet.AddMaterial(vis_mat_gt)

red_shapet = red_cylt.GetVisualModel().GetShape(0)
red_shapet.AddMaterial(vis_mat_r)

green_cylt.SetPos(chrono.ChVector3d(0,0,10000))
green_cylt.SetFixed(False)
red_cylt.SetPos(chrono.ChVector3d(0,0,10000))
red_cylt.SetFixed(False)

sedan.GetSystem().Add(green_cylt)
sedan.GetSystem().Add(red_cylt)

def indicator_true(state='safe'):
    if state == 'safe':
        green_cylt.SetPos(sedan.GetChassis().GetPos() + chrono.ChVector3d(-0.5,0,3.5))
    else:
        red_cylt.SetPos(sedan.GetChassis().GetPos() + chrono.ChVector3d(-0.5,0,3.5))
#sedan.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

# Create the terrain

patch_mat = chrono.ChContactMaterialNSC()
patch_mat.SetFriction(0.9)
patch_mat.SetRestitution(0.01)
terrain = veh.RigidTerrain(sedan.GetSystem())
patch = terrain.AddPatch(patch_mat, 
    chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0), chrono.QUNIT), 
    terrainLength, terrainWidth)

patch.SetTexture(veh.GetDataFile("terrain/textures/Concrete002_2K-JPG/Concrete002_2K_Color.jpg"), 100, 100)
patch.SetColor(chrono.ChColor(0.8, 0.8, 0.5))
terrain.Initialize()

# adding visualization for the reference trajectory
# Create visualization material for the path
vis_mat_path = chrono.ChVisualMaterial()
vis_mat_path.SetDiffuseColor(chrono.ChColor(0.0, 1.0, 0.0))
road_vis_mat = chrono.ChVisualMaterial()
road_vis_mat.SetDiffuseColor(chrono.ChColor(1.0, 1.0, 1.0))
# Create ChBodyEasyBox objects at specified positions
# reference centerline of the sedan
# reference_trajectory = np.loadtxt('./data/reference_traj/trajectory.csv', delimiter=',')
reference_trajectory = np.loadtxt(project_root+'/CCTA-highwayControl/data/reference_traj/trajectory_complex_4.csv', delimiter=',')
# reference_trajectory = np.loadtxt(project_root+'/CCTA-highwayControl/data/reference_traj/trajectory_straight.csv', delimiter=',')
# only show sparse points not all points
visual_reference_trajectory = reference_trajectory[::50]
for pos in visual_reference_trajectory:
    center_x,center_y,center_heading = pos
    
    right_lane_x = center_x + 1.85 * np.cos(center_heading+np.pi/2)
    right_lane_y = center_y + 1.85 * np.sin(center_heading+np.pi/2)
    left_lane_x = center_x + 1.85 * np.cos(center_heading-np.pi/2)
    left_lane_y = center_y + 1.85 * np.sin(center_heading-np.pi/2)
    lane_rot = chrono.QuatFromAngleZ(center_heading)
    # add right lane and left lane
    right_lane_body = chrono.ChBodyEasyBox(1.5, 0.1, 0.05, 1000, True, False)
    right_lane_body.SetPos(chrono.ChVector3d(right_lane_x, right_lane_y, 0.5))
    right_lane_body.SetRot(lane_rot)
    right_lane_body.SetFixed(True)
    right_lane_shape = right_lane_body.GetVisualModel().GetShape(0)
    right_lane_shape.AddMaterial(road_vis_mat)
    left_lane_body = chrono.ChBodyEasyBox(1.5, 0.1, 0.05, 1000, True, False)
    left_lane_body.SetPos(chrono.ChVector3d(left_lane_x, left_lane_y, 0.5))
    left_lane_body.SetRot(lane_rot)
    left_lane_body.SetFixed(True)
    left_lane_shape = left_lane_body.GetVisualModel().GetShape(0)
    left_lane_shape.AddMaterial(road_vis_mat)
    sedan.GetSystem().Add(right_lane_body)
    sedan.GetSystem().Add(left_lane_body)
    
    #sedan.GetSystem().Add(box_body)

# manager = sens.ChSensorManager(sedan.GetSystem())

intensity = 1.0
# manager.scene.AddPointLight(chrono.ChVector3f(2, 2.5, 100), chrono.ChColor(intensity, intensity, intensity), 500.0)

# Update rate in Hz
update_rate = 15
# Image width and height
image_width = 1280
image_height = 720
# Camera's horizontal field of view
fov = 2.8
# Lag (in seconds) between sensing and when data becomes accessible
lag = 0
# Exposure (in seconds) of each image
exposure_time = 0
offset_pose = chrono.ChFramed(
        chrono.ChVector3d(-1.35, -0.25, 0.8), chrono.QuatFromAngleAxis(-0.1, chrono.ChVector3d(0, 1, 0)))
# cam = sens.ChCameraSensor(
#         sedan.GetChassisBody(),              # body camera is attached to
#         update_rate,            # update rate in Hz
#         offset_pose,            # offset pose
#         image_width,            # image width
#         image_height,           # image height
#         fov)                    # camera's horizontal field of view
# cam.SetName("Camera Sensor")
# cam.SetLag(lag)
# cam.SetCollectionWindow(exposure_time)
# cam.PushFilter(sens.ChFilterVisualize(image_width, image_height, "Before Grayscale Filter"))
# # add sensor to manager
# manager.AddSensor(cam)


# initialize the virtual vehicles
ini_state = [1000000,1000000,0,0]
sur_veh1 = simplifiedVehModel(sedan.GetSystem(),ini_state,[0,0],sur_control_step_size)
sur_veh2 = simplifiedVehModel(sedan.GetSystem(),ini_state,[0,0],sur_control_step_size)
sur_veh3 = simplifiedVehModel(sedan.GetSystem(),ini_state,[0,0],sur_control_step_size)
sur_veh4 = simplifiedVehModel(sedan.GetSystem(),ini_state,[0,0],sur_control_step_size)
sur_veh5 = simplifiedVehModel(sedan.GetSystem(),ini_state,[0,0],sur_control_step_size)

# virtual vehicle for powertrain conversion
# vir_veh = simplifiedVehModel(sedan.GetSystem(),[0,0,0,0],[0,0],control_step_size,False)
# -------------------------------------
# Create the vehicle Irrlicht interface
# Create the driver system
# -------------------------------------

vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
vis.SetWindowTitle('sedan vehicle simulation')
vis.SetWindowSize(1280, 1024)
vis.SetChaseCamera(trackPoint, 6.0, 0.5)
vis.Initialize()
vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
vis.AddLightDirectional()
# vis.AddTypicalLights()
vis.AddSkyBox()
vis.AttachVehicle(sedan.GetVehicle())


# Create the driver system
driver = veh.ChInteractiveDriverIRR(vis)
driver.SetJoystickConfigFile(project_root+'/CCTA-highwayControl/data/driving_wheel.json')# Set the time response for steering and throttle keyboard inputs.
steering_time = 1.0  # time to go from 0 to +1 (or from 0 to -1)
throttle_time = 1.0  # time to go from 0 to +1
braking_time = 0.3   # time to go from 0 to +1
driver.SetSteeringDelta(render_step_size / steering_time)
driver.SetThrottleDelta(render_step_size / throttle_time)
driver.SetBrakingDelta(render_step_size / braking_time)

driver.Initialize()
# ---------------
# Simulation loop
# ---------------

# Number of simulation steps between miscellaneous events
render_steps = math.ceil(render_step_size / step_size)
control_steps = math.ceil(control_step_size / step_size)
low_control_step =  math.ceil(low_control_step_size / step_size)
sur_control_step =  math.ceil(sur_control_step_size / step_size)
sample_steps = math.ceil(sample_step_size / step_size)
cbf_steps = math.ceil(cbf_step_size / step_size)
mpc_control_step = math.ceil(mpc_step_size / step_size)

# Initialize simulation frame counter s
realtime_timer = chrono.ChRealtimeStepTimer()
step_number = 0
control_number = 0
render_frame = 0
is_merging = False
is_merging_left = False
merge_trajectory = []
merge_trajectory_left = []
merge_traj_ind = 0
merge_traj_ind_left = 0

is_parallel = False
is_parallel_left = False
parallel_trajectory = []
parallel_trajectory_left = []
parallel_traj_ind = 0
parallel_traj_ind_left = 0

is_serial = False
serial_trajectory = []
serial_traj_ind = 0

# Constants
merge_start_time = 5.0
min_pause = 3.0  # Minimum pause between trajectories
max_pause = 4.0  # Maximum pause between trajectories

# Variables to track last trajectory generation time
last_trajectory_time = 0
next_trajectory_time = merge_start_time

# Variables to track the last side used for each scenario type
last_merge_side = None
last_parallel_side = None
NMPC = mpc_wpts_solver_sedan()

ind_s1,ind_s2,ind_s3,ind_s4,ind_s5 = 0,0,0,0,0
log_s1,log_s2,log_s3,log_s4,log_s5 = [],[],[],[],[]
train_data = {}
center_veh_log = []
rmax = 20
all_trajj = []
vir_veh_1 = [0,0]
vir_veh_2 = [0,0]
vir_veh_3 = [0,0]
vir_veh_4 = [0,0]
vir_veh_5 = [0,0]
SV_pos = np.array([vir_veh_1, vir_veh_2, vir_veh_3, vir_veh_4, vir_veh_5]).T
v_SV_pos = np.array([[0.,0,],[0.,0.],[0.,0.],[0.,0.],[0.,0.]]).T

ct_sim = 0
veh_pos = sedan.GetChassisBody().GetPos()
veh_heading = sedan.GetChassisBody().GetRot().GetCardanAnglesZYX().z
veh_x = veh_pos.x
veh_y = veh_pos.y
veh_heading = veh_heading


real_vehicle_state = []
vir_vehicle_state = []


# Initialize pygame and joystick

if enable_joystick:
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

while vis.Run() :
    time = sedan.GetSystem().GetChTime()

    # manager.Update()

    # control sedan vehicle 
    # driver_sedan = straight() # lanechange
    # get trailer and tractor position
    veh_pos = sedan.GetChassisBody().GetPos()
    veh_heading = sedan.GetChassisBody().GetRot().GetCardanAnglesZYX().z
    veh_x = veh_pos.x
    veh_y = veh_pos.y
    veh_heading = veh_heading
    veh_back_x = veh_x - 5.74 * np.cos(veh_heading)
    veh_back_y = veh_y - 5.74 * np.sin(veh_heading)
    
    # listen to the joystick button 23
    if enable_joystick:
        for event in pygame.event.get():
            # Button press event
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 23:  # Button 23 pressed
                    is_manual = not is_manual  # Toggle the value
                    print(f"is_manual is now: {is_manual}")

    if time > next_trajectory_time:
    # if time > 30:

        # Randomly choose between tractor and trailer position
        if np.random.rand() < 0.5:
            sedan_x, sedan_y = veh_x, veh_y
            position_used = "parallel"
        else:
            sedan_x, sedan_y = veh_back_x, veh_back_y
            position_used = "behind"

        # Randomly choose between merge, parallel, and serial scenarios
        scenario_choice = np.random.rand()
        
        if scenario_choice > 0.7 and (not flagcbf):  # 40% chance for merge
            # Merge scenario
            from_side = 'left' if last_merge_side != 'left' else 'right'
            driver_vel = np.random.uniform(40, 45)
            merge_time = np.random.uniform(2.5, 4.5)
            
            new_trajectory = merging_scenario(reference_trajectory, sedan_x, sedan_y, 
                                              from_left_right=from_side,
                                              vel=driver_vel,
                                              merge_time=merge_time,
                                              freq=1/render_step_size)
            print(f"Merging from {from_side}, velocity: {driver_vel:.2f}, merge time: {merge_time:.2f}, using {position_used} position")
            print(new_trajectory.shape)
            
            if from_side == 'left':
                merge_trajectory_left = new_trajectory
                is_merging_left = True
                merge_traj_ind_left = 0
                sur_veh2.set_state([merge_trajectory_left[0][0],merge_trajectory_left[0][1],merge_trajectory_left[0][2],driver_vel])
            else:
                merge_trajectory = new_trajectory
                is_merging = True
                merge_traj_ind = 0
                # add virtual vehicle
                sur_veh1.set_state([merge_trajectory[0][0],merge_trajectory[0][1],merge_trajectory[0][2],driver_vel])
            
            last_merge_side = from_side
        elif scenario_choice < 0.3 and (not flagcbf):  # 40% chance for parallel
            # Parallel scenario
            from_side = 'left' if last_parallel_side != 'left' else 'right'
            driver_vel = np.random.uniform(30, 40)
            run_time = np.random.uniform(3.5, 5.5)
            
            new_trajectory = parallel_scenario(reference_trajectory, sedan_x, sedan_y, 
                                               from_left_right=from_side,
                                               vel=driver_vel,
                                               run_time=run_time,
                                               freq=1/render_step_size)
            print(f"Parallel from {from_side}, velocity: {driver_vel:.2f}, run time: {run_time:.2f}, using {position_used} position")
            print(new_trajectory.shape)
            
            if from_side == 'left':
                parallel_trajectory_left = new_trajectory
                is_parallel_left = True
                parallel_traj_ind_left = 0
                sur_veh3.set_state([parallel_trajectory_left[0][0],parallel_trajectory_left[0][1],parallel_trajectory_left[0][2],driver_vel])
            else:
                parallel_trajectory = new_trajectory
                is_parallel = True
                parallel_traj_ind = 0
                sur_veh4.set_state([parallel_trajectory[0][0],parallel_trajectory[0][1],parallel_trajectory[0][2],driver_vel])
            
            last_parallel_side = from_side
        elif (0.3<scenario_choice < 0.7) and (not flagcbf):  # 20% chance for serial, but only if no merging is active
            if not is_merging and not is_merging_left:
                # Serial scenario
                driver_vel = np.random.uniform(15, 20)
                # run_time = np.random.uniform(3.5, 5.5)
                # driver_vel = 12.0
                run_time = np.random.uniform(23.5, 25.5)
                
                serial_trajectory = serial_scenario(reference_trajectory, sedan_x, sedan_y,
                                                    deviation_distance=32,
                                                    vel=driver_vel,
                                                    run_time=run_time,
                                                    freq=1/render_step_size)
                print(f"Serial scenario, velocity: {driver_vel:.2f}, run time: {run_time:.2f}, using {position_used} position")
                print(serial_trajectory.shape)
                
                is_serial = True
                serial_traj_ind = 0
                sur_veh5.set_state([serial_trajectory[0][0],serial_trajectory[0][1],serial_trajectory[0][2],driver_vel])
            else:
                print("Serial scenario skipped due to active merging")

        # Set the next trajectory generation time
        last_trajectory_time = time
        next_trajectory_time = time + np.random.uniform(min_pause, max_pause)
####################################### Controller design ######################################
####################################### Controller design ######################################
####################################### Controller design ######################################
####################################### Controller design ######################################
####################################### Controller design ######################################
####################################### Controller design ######################################
####################################### Controller design ######################################
####################################### Controller design ######################################
####################################### Controller design ######################################
    # driver_inputs = driver.GetInputs()
    if (step_number % control_steps == 0):
        if is_manual:
            driver_inputs = driver.GetInputs()
            #print(f"throttle and braking: {driver_inputs.m_throttle}, {driver_inputs.m_braking}")
            # throttle = driver_inputs.m_throttle, steering = driver_inputs.m_steering, braking = driver_inputs.m_braking
            if len(Vcontrol_seq) >= td_max:
                Vcontrol_seq.pop(0)  # Remove the first entry
            Vcontrol_seq.append(driver_inputs.m_throttle - driver_inputs.m_braking)  # Add the new value to the end
            if len(Tcontrol_seq) >= td_max:
                Tcontrol_seq.pop(0)  # Remove the first entry
            Tcontrol_seq.append(driver_inputs.m_steering)  # Add the new value to the end

            if len(Vcontrol_seq) > td:
                # ref_vel = Vcontrol_seq[-td-1]
                ref_throttle = Vcontrol_seq[-td-1]
                ref_steering = Tcontrol_seq[-td-1]
            else:
                # ref_vel = 0.
                ref_throttle = 0.
                ref_steering = 0.
            if DelayCompMode:
                if len(Vcontrol_seq) > td:

                    vtd = Vcontrol_seq[-td-1]
                    dvtd = (Vcontrol_seq[-td-1+1] - Vcontrol_seq[-td-1]) / (control_step_size)
                    vptd = Vp_seq[-td-1]
                    dvptd = (Vp_seq[-td-1+1] - Vp_seq[-td-1]) / control_step_size
                    
                    vp_current = Vp_seq[-1] + control_step_size * (dvtd + k1 * (dvtd - dvptd) + k2 * (vtd - vptd))
                    Vp_seq.pop(0)
                    Vp_seq.append(vp_current)

                    Ttd = Tcontrol_seq[-td-1]
                    dTtd = (Tcontrol_seq[-td-1+1] - Tcontrol_seq[-td-1]) / (control_step_size)
                    Tptd = Tp_seq[-td-1]
                    dTptd = (Tp_seq[-td-1+1] - Tp_seq[-td-1]) / control_step_size
                    
                    Tp_current = Tp_seq[-1] + control_step_size * (dTtd + k1 * (dTtd - dTptd) + k2 * (Ttd - Tptd))
                    Tp_seq.pop(0)
                    Tp_seq.append(Tp_current)
                    # if vp_current > 1:
                    #     Kp = 1.5 / 20
                    #     current_vel = sedan.GetVehicle().GetSpeed()
                    #     vel_error = vp_current - current_vel
                    #     ref_throttle = Kp * vel_error
                    # else:
                    ref_throttle = vp_current
                    steering = Tp_current
                    ref_steering = Tp_current



            ref_throttle = np.clip(ref_throttle,-1.0,1.0)
            ref_steering = np.clip(ref_steering,-1.0,1.0)

            throttle = ref_throttle
            steering = ref_steering

            ref_beta = ref_steering * delta
            # beta = ref_steering * delta
            # print(f'driver throttle = {driver_inputs.m_throttle}, ref throttle: {ref_throttle}')
            
            ###### CBF DOING THE WORK ####
            veh_state = [veh_x,veh_y,veh_heading,sedan.GetVehicle().GetSpeed(), beta]
            # if veh_state[3] > 0.1:
            #     weight1 = 1.0 / veh_state[3] / veh_state[3]

            # Q = np.array([[weight1, 0.],[0.,weight2]])
            D = 0.05 * delta
            ref_dbeta = -ctrlpara['kblf'] * (beta - ref_beta) # low-level tracking
            # print(f'{beta=},{ref_beta=},{ref_throttle=},{ref_dbeta=}')

            ud = (ref_throttle, ref_dbeta)
            upre = (alpha, dbeta)
            if CBFMode:
                (alpha, dbeta), flagcbf = DOBCBF_dbeta(veh_state, ud, upre, p_dob, ctrlpara, dobpara,
                                                        weight1, weight2, SV_pos, v_SV_pos, ro, r_precpt, bias, dob_mode, cbfflavor) # DOB-CBF-QP
            else:
                alpha = ref_throttle
                dbeta = ref_dbeta
            # print(f'{alpha=},{dbeta=}')
            # alpha, beta = (ref_throttle, ref_steering) # uncomment if no CBF
            beta = beta + dbeta * low_control_step_size
            # ref_throttle = alpha
            # ref_dbeta = dbeta

            # print(f'{ud=},(a,b)={(alpha,beta)}')
            alpha = np.clip(alpha,-1.0,1.0)
            # beta = np.clip(beta,-1.0,1.0)
            alpha_list.append([time, driver_inputs.m_throttle - driver_inputs.m_braking, ref_throttle, alpha])
            beta_list.append([time, driver_inputs.m_steering, ref_steering, np.clip(beta/delta,-1.0,1.0)])
            flagcbf_list.append([time, flagcbf])
        else:
            if (step_number % mpc_control_step == 0):
                driver_inputs = driver.GetInputs()

                ref_traj = reference_trajectory
                lookahead = 5.0
                veh_state=[veh_x,veh_y,veh_heading]
                N_mpc = 10
                reff_vel = 20
                ref_sampling_step = 10 # good

                index = FindMinDistInd(veh_state=veh_state, ref_traj=ref_traj, lookahead=lookahead)
                ref_state_current = list(ref_traj[index,:])
                xref = reference_trajectory[index+ref_sampling_step:index+ref_sampling_step*(N_mpc+2):ref_sampling_step,:].T
                
                # xref = np.array([ref_state_current]).T * np.ones((3, N_mpc+1))
                uref = np.vstack((reff_vel * np.ones((1, xref.shape[1]-1)), np.zeros((1, xref.shape[1]-1))))
                
                start_time = tm.time()
                ctrl, _ = mpc_wpts_solver_sedan3(SV_pos, veh_state, xref, uref,mpc_step_size, N_mpc) # the last argument is penalty
                
                ctrl = np.array(ctrl).squeeze()
                print('---------------------------------------------')
                print('solving time: %s' % (tm.time() - start_time))

                ref_vel = ctrl[0]
                ref_steering = ctrl[1]
                
                current_vel = sedan.GetVehicle().GetSpeed()
                vel_error = ref_vel - current_vel

                # # print(f'ref_vel = {ref_vel}, ref_steering = {steering}, currrent_vel = {current_vel}')
                Kp = 1.5 / 30
                if vel_error >= 0:
                    ref_throttle = Kp * vel_error
                    braking = 0.
                else:
                    braking = -Kp * vel_error
                    ref_throttle = Kp * vel_error
                # ct_sim += 1

                # braking = np.clip(braking,0.0,1.0)
                ref_throttle = np.clip(ref_throttle,-1.0,1.0)
                ref_steering = np.clip(ref_steering,-1.0,1.0)

                alpha = ref_throttle
                beta = ref_steering * delta

            # ref_beta = ref_steering * delta
            # beta = ref_steering * delta
            # print(f'throttle = {throttle}, braking = {braking}, steering = {steering}')

    if (step_number % low_control_step == 0):
        veh_state = [veh_x,veh_y,veh_heading,sedan.GetVehicle().GetSpeed(), beta]

        scc = 10000.
        for i in range(SV_pos.shape[1]):
            Rmatt = np.array([[np.cos(veh_heading), np.sin(veh_heading)],[-np.sin(veh_heading), np.cos(veh_heading)]])
            diffx = np.array([SV_pos[0,i]- veh_x, SV_pos[1,i]- veh_y])
            re_diffx = Rmatt @ diffx
            # if re_diffx[0] >= -0.1 * r_precpt:
                # scc = np.min((scc, sc(SV_pos[:,i],np.array([veh_x, veh_y]),Q,Rmatt,bias)))
            scc = np.min((scc, sc(SV_pos[:,i],np.array([veh_x, veh_y]),Q,Rmatt,bias)))
        hvalue_list.append([time,scc])
        # alpha = np.clip(alpha,-1.0,1.0)
        # beta = np.clip(beta,-1.0,1.0)

        dp, hdxyvta = xdot_dob_dbeta(veh_state, p_dob, (alpha, dbeta), dobpara)
        # print(f'scc = {scc}')
        # dp, hdxyvta = xdot_dob(veh_state, p_dob, (alpha, beta), dobpara)


        p_dob = p_dob + low_control_step_size * dp

        if alpha > 0:
            throttle = alpha
            braking = 0.
        else:
            braking = -alpha
            throttle = 0.
        steering = np.clip(beta/delta,-1.0,1.0)

        beta = steering * delta
        
        aEst = -16.576565271233825 + 1.235469*veh_state[3] + (-2.39238426e-02)*veh_state[3]**2 + \
               2.15863500e+01*alpha + (-1.34372610)*veh_state[3]*alpha + (2.17613620e-02)*alpha*veh_state[3]**2 + 1.59919731e-04*alpha*veh_state[3]**3 + hdxyvta
        a_est_list.append(aEst)
        a_model_list.append(aEst -hdxyvta)

        
            
        driver.SetSteering(steering)
        driver.SetThrottle(throttle)
        driver.SetBraking(braking)
        # Get driver inputs
        driver_inputs = driver.GetInputs()

    if (step_number % sur_control_step == 0):
        if is_merging and merge_traj_ind < len(merge_trajectory):
            control1 = vir_veh_controller(veh_state=[sur_veh1.x,sur_veh1.y, sur_veh1.theta,sur_veh1.v],ref_traj=merge_trajectory,lookahead=1.0)
            sur_veh1.update(control1)
            # print("updating vir veh 1")
        
        if is_merging_left and merge_traj_ind_left < len(merge_trajectory_left):
            control2 = vir_veh_controller(veh_state=[sur_veh2.x,sur_veh2.y, sur_veh2.theta,sur_veh2.v],ref_traj=merge_trajectory_left,lookahead=1.0)
            sur_veh2.update(control2)
        
        if is_parallel and parallel_traj_ind < len(parallel_trajectory):
            control3 = vir_veh_controller(veh_state=[sur_veh4.x,sur_veh4.y, sur_veh4.theta,sur_veh4.v],ref_traj=parallel_trajectory,lookahead=1.0)
            sur_veh4.update(control3)
        
        if is_parallel_left and parallel_traj_ind_left < len(parallel_trajectory_left):
            control4 = vir_veh_controller(veh_state=[sur_veh3.x,sur_veh3.y, sur_veh3.theta,sur_veh3.v],ref_traj=parallel_trajectory_left,lookahead=1.0)
            sur_veh3.update(control4)
        
        if is_serial and serial_traj_ind < len(serial_trajectory):
            control5 = vir_veh_controller(veh_state=[sur_veh5.x,sur_veh5.y, sur_veh5.theta,sur_veh5.v],ref_traj=serial_trajectory,lookahead=1.0)
            sur_veh5.update(control5)
            
   
    
    # Render scene and output POV-Ray data
    if (step_number % render_steps == 0) :
        # print('sur veh 3:',sur_veh3.get_state())
        center_veh_log.append([time,veh_x,veh_y,veh_heading])
        train_data['center_vehicle'] = center_veh_log
        # print vir veh 1 
        if is_merging and merge_traj_ind < len(merge_trajectory):
            dis_2_end_1 = np.sqrt((merge_trajectory[-1][0]-sur_veh1.x)**2 + (merge_trajectory[-1][1]-sur_veh1.y)**2)
            if dis_2_end_1 < 1:
                is_merging = False
                merge_traj_ind = 0
                train_data['s1_'+str(ind_s1)] = log_s1
                log_s1 = []
                ind_s1 += 1
                # print("vehicle 1 reached the end, turn off the visualization")
                sur_veh1.pause_visualization()
            else:
                # print('vehicle 1:',sur_veh1.get_state())
                log_s1.append([time,sur_veh1.x,sur_veh1.y,sur_veh1.theta])
                merge_traj_ind += 1

        elif is_merging:
            is_merging = False
            merge_traj_ind = 0
            train_data['s1_'+str(ind_s1)] = log_s1
            log_s1 = []
            ind_s1 += 1
            # print("vehicle 1 reached the end at 2nd condition")
        vir_veh_1 = [sur_veh1.x, sur_veh1.y] if is_merging else [0,0]  
        speed_veh_1 = [sur_veh1.v * np.cos(sur_veh1.theta), sur_veh1.v * np.sin(sur_veh1.theta)]  if is_merging else [0,0]  
            
        
        # Left merge
        if is_merging_left and merge_traj_ind_left < len(merge_trajectory_left):
            dis_2_end_2 = np.sqrt((merge_trajectory_left[-1][0]-sur_veh2.x)**2 + (merge_trajectory_left[-1][1]-sur_veh2.y)**2)
            if dis_2_end_2 < 1:
                is_merging_left = False
                merge_traj_ind_left = 0
                train_data['s2_'+str(ind_s2)] = log_s2
                log_s2 = []
                ind_s2 += 1
                sur_veh2.pause_visualization()
            else:
                log_s2.append([time,sur_veh2.x,sur_veh2.y,sur_veh2.theta])
                merge_traj_ind_left += 1

        elif is_merging_left:
            is_merging_left = False
            merge_traj_ind_left = 0
            train_data['s2_'+str(ind_s2)] = log_s2
            log_s2 = []
            ind_s2 += 1
        vir_veh_2 = [sur_veh2.x, sur_veh2.y] if is_merging_left else [0,0]  
        speed_veh_2 = [sur_veh2.v * np.cos(sur_veh2.theta), sur_veh2.v * np.sin(sur_veh2.theta)]  if is_merging_left else [0,0]  
            
        # Right parallel
        if is_parallel and parallel_traj_ind < len(parallel_trajectory):
            dis_2_end_3 = np.sqrt((parallel_trajectory[-1][0]-sur_veh4.x)**2 + (parallel_trajectory[-1][1]-sur_veh4.y)**2)
            if dis_2_end_3 < 3:
                is_parallel = False
                parallel_traj_ind = 0
                train_data['s3_'+str(ind_s3)] = log_s3
                log_s3 = []
                ind_s3 += 1
                sur_veh4.pause_visualization()
                # print("vehicle 3 reached the end, exit at condition 1")
            else:
                log_s3.append([time,sur_veh4.x,sur_veh4.y,sur_veh4.theta])
                parallel_traj_ind += 1
            
        elif is_parallel:
            is_parallel = False
            parallel_traj_ind = 0
            train_data['s3_'+str(ind_s3)] = log_s3
            log_s3 = []
            ind_s3 += 1
            sur_veh4.pause_visualization()
            # print("vehicle 3 reached the end, exit at condition 2")
        vir_veh_4 = [sur_veh4.x, sur_veh4.y] if is_parallel else [0,0]  
        speed_veh_4 = [sur_veh4.v * np.cos(sur_veh4.theta), sur_veh4.v * np.sin(sur_veh4.theta)]  if is_parallel else [0,0]  


        # Left parallel
        if is_parallel_left and parallel_traj_ind_left < len(parallel_trajectory_left): 
            dis_2_end_4 = np.sqrt((parallel_trajectory_left[-1][0]-sur_veh3.x)**2 + (parallel_trajectory_left[-1][1]-sur_veh3.y)**2)
            if dis_2_end_4 < 3:
                is_parallel_left = False
                parallel_traj_ind_left = 0
                train_data['s4_'+str(ind_s4)] = log_s4
                log_s4 = []
                ind_s4 += 1
                sur_veh3.pause_visualization()
            else:
                log_s4.append([time,sur_veh3.x,sur_veh3.y,sur_veh3.theta])
                parallel_traj_ind_left += 1

        elif is_parallel_left:
            is_parallel_left = False
            parallel_traj_ind_left = 0
            train_data['s4_'+str(ind_s4)] = log_s4
            log_s4 = []
            ind_s4 += 1
            sur_veh3.pause_visualization()

        vir_veh_3 = [sur_veh3.x, sur_veh3.y] if is_parallel_left else [0,0]  
        speed_veh_3 = [sur_veh3.v * np.cos(sur_veh3.theta), sur_veh3.v * np.sin(sur_veh3.theta)]  if is_parallel_left else [0,0]  

        # Serial
        if is_serial and serial_traj_ind < len(serial_trajectory):
            dis_2_end_5 = np.sqrt((serial_trajectory[-1][0]-sur_veh5.x)**2 + (serial_trajectory[-1][1]-sur_veh5.y)**2)
            if dis_2_end_5 < 1:
                is_serial = False
                serial_traj_ind = 0
                train_data['s5_'+str(ind_s5)] = log_s5
                log_s5 = []
                ind_s5 += 1
                sur_veh5.pause_visualization()
            else:
                log_s5.append([time,sur_veh5.x,sur_veh5.y,sur_veh5.theta])

                serial_traj_ind += 1

        elif is_serial:
            is_serial = False
            serial_traj_ind = 0
            train_data['s5_'+str(ind_s5)] = log_s5
            log_s5 = []
            ind_s5 += 1
            sur_veh5.pause_visualization()
        vir_veh_5 = [sur_veh5.x, sur_veh5.y] if is_serial else [0,0]  
        speed_veh_5 = [sur_veh5.v * np.cos(sur_veh5.theta), sur_veh5.v * np.sin(sur_veh5.theta)]  if is_serial else [0,0]  
        
        SV_pos = np.array([vir_veh_1, vir_veh_2, vir_veh_4, vir_veh_3, vir_veh_5]).T
        v_SV_pos = np.array([speed_veh_1, speed_veh_2, speed_veh_4, speed_veh_3, speed_veh_5]).T
        if WarningSystemMode:
            if (step_number % sample_steps == 0) :
                SV_pos = np.array([vir_veh_1, vir_veh_2, vir_veh_4, vir_veh_3, vir_veh_5]).T
                all_trajj.append([veh_x, veh_y, veh_heading, \
                                vir_veh_1[0], vir_veh_1[1], \
                                vir_veh_2[0], vir_veh_2[1], \
                                vir_veh_3[0], vir_veh_3[1], vir_veh_4[0], vir_veh_4[1], vir_veh_5[0], vir_veh_5[1]])
                if len(all_trajj) > args['in_length']:
                    sc_pred = 10000.
                    all_trajj = all_trajj[-args['in_length']:]

                    all_traj = np.array(all_trajj)
                    ref_pos = all_traj[-1,:2]
                    ego_heading = all_traj[-1,2]
                    Rmat = np.array([[np.cos(ego_heading), np.sin(ego_heading)],[-np.sin(ego_heading), np.cos(ego_heading)]])
                    hist, nbrs, mask = FindNbrs(all_traj, args)
                    hist = hist[:,:,-1::-1]
                    nbrs = nbrs[:,:,-1::-1]
                    if args['use_cuda']:
                        hist = torch.tensor(hist.copy(), dtype=torch.float32).cuda()
                        nbrs = torch.tensor(nbrs.copy(), dtype=torch.float32).cuda()
                        mask = torch.tensor(mask.copy()).cuda()
                    fut_pred, weight_ts_center, weight_ts_nbr, weight_ha= net(hist, nbrs, mask, 0, 0)
                    fut_pred = fut_pred[:,0,:].detach().cpu().numpy()
                    fut_pred = fut_pred[:,-1::-1] @ Rmat + ref_pos

                    warningsignal, sc_pred = CPWarningSystem(sc=sc_warning, Q=Qwarning, Rmat=Rmat, CP_data_f0=CP_data_f0, pred=fut_pred, SV_pos=SV_pos, eps=eps)
                    # print(warningsignal)
                    # print(sc_pred)
                    with open(f'data/training_data/{scene_prefix}_data_warning_test_F{f0warning}_eps{eps}.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([time, veh_x, veh_y, veh_heading, sc_pred, \
                                        vir_veh_1[0], vir_veh_1[1], \
                                        vir_veh_2[0], vir_veh_2[1], \
                                        vir_veh_3[0], vir_veh_3[1], \
                                        vir_veh_4[0], vir_veh_4[1], \
                                        vir_veh_5[0], vir_veh_5[1]])
                    with open(f'data/training_data/{scene_prefix}_data_warning_test_eps{eps}.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([time, veh_x, veh_y, veh_heading, warningsignal, \
                                        vir_veh_1[0], vir_veh_1[1], \
                                        vir_veh_2[0], vir_veh_2[1], \
                                        vir_veh_3[0], vir_veh_3[1], \
                                        vir_veh_4[0], vir_veh_4[1], \
                                        vir_veh_5[0], vir_veh_5[1]])
        if WarningSystemMode:
            if warningsignal:
                indicator('danger')
            else:
                indicator('safe')
        SV_poss = np.array([vir_veh_1, vir_veh_2, vir_veh_3, vir_veh_4, vir_veh_5]).T
        Rmatt = np.array([[np.cos(veh_heading), np.sin(veh_heading)],[-np.sin(veh_heading), np.cos(veh_heading)]])
        gY = 10000.
        for i in range(SV_poss.shape[1]):
            gY = np.min((gY, sc(SV_poss[:,i],np.array([veh_x, veh_y]),Q,Rmatt,bias)))
            diffx = np.array([SV_poss[0,i]- veh_x, SV_poss[1,i]- veh_y])
            re_diffx = Rmatt @ diffx

        if gY <  f0 and (re_diffx[0] >= -0.1 * r_precpt):
            indicator_true('danger')
        else:
            indicator_true('safe')
                # for i in range(SV_pos.shape[1]):
                #     dd = (veh_x - SV_pos[0,i])**2 + (veh_y - SV_pos[1,i])**2
                #     if dd <= 10 * rmax*rmax:
                #         indicator('danger')
                #     else:
                #         indicator('safe')
        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        # filename = 'figs/img_' + str(render_frame) +'.jpg' 
        # vis.WriteImageToFile(filename)
        # render_frame += 1


        

    
    # Update modules (process inputs from other modules)
    driver.Synchronize(time)
    terrain.Synchronize(time)
    sedan.Synchronize(time, driver_inputs, terrain)
    vis.Synchronize(time, driver_inputs)

    # Advance simulation for one timestep for all modules
    driver.Advance(step_size)
    terrain.Advance(step_size)
    sedan.Advance(step_size)
    vis.Advance(step_size)
    
    if (step_number % low_control_step == 0):
        a_actual_list.append(sedan.GetVehicle().GetPointAcceleration(chrono.ChVector3d(0,0,0)).x)
    # Increment frame number
    step_number += 1

    # Spin in place for real time to catch up
    # realtime_timer.Spin(step_size)

else:
    list_of_traindata = list(train_data.values())
    print('number of element: ',len(list_of_traindata))
    # this is the center vehicle
    print('number of steps: ', len(list_of_traindata[0]))
    # np.save(project_root+'/CCTA-highwayControl/data/training_data/sedan_train_data.npy', np.array(list_of_traindata, dtype=object))
    # with open(project_root+'/CCTA-highwayControl/data/training_data/sedan_train_data.txt', 'w') as f:
    #     for item in list_of_traindata:
    #         np.savetxt(f, item, fmt='%f')
    #         f.write('\n')
    if IfSave:
        np.savetxt('data/' + dir_acc_value, np.array([a_actual_list, a_est_list, a_model_list]).T,delimiter=',',fmt='%f')
        np.savetxt('data/' + dir_hvalue, np.array(hvalue_list),delimiter=',',fmt='%f')
        np.savetxt('data/' + dir_alpha_value, np.array(alpha_list),delimiter=',',fmt='%f')
        np.savetxt('data/' + dir_beta_value, np.array(beta_list),delimiter=',',fmt='%f')
        np.savetxt('data/' + dir_flagcbf, np.array(flagcbf_list),delimiter=',',fmt='%f')
        # np.save(project_root+'/CCTA-highwayControl/data/training_data/sedan_train_data.npy', np.array(list_of_traindata, dtype=object))
        np.save('data/'+ dir_veh_state, np.array(list_of_traindata, dtype=object))


    loaded_data = list(np.load(project_root+'/CCTA-highwayControl/data/training_data/sedan_train_data.npy', allow_pickle=True))
    print('number of element',len(loaded_data))
    print('number of steps: ', len(loaded_data[0])) # center vehicle information

    print(len(a_actual_list))
    print(len(a_est_list))
    print(len(a_model_list))
    plt.figure(dpi=150)
    dt = 5
    plt.plot(np.array(a_actual_list)[::dt],'r',label='Real Acc')
    plt.plot(np.array(a_est_list)[::dt],'b',label='Corrected Acc')
    plt.plot(np.array(a_model_list)[::dt],'g',label='Acc model output')

    # Add labels and title
    plt.xlabel('t')
    plt.ylabel('m/s^2')
    # plt.title('2D Scatter Plot of Points')

    # Add grid and legend
    plt.grid(True)
    plt.legend()
    plt.tight_layout

    # Show the plot
    plt.show()

