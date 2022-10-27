# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =============================================================================
# Calibration of the SCM parameters using Bayesian Optimization
# Author: Wei Hu
# =============================================================================
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import ctypes
from numpy.ctypeslib import ndpointer
import pymc3 as pm
import arviz as az
import theano
import theano.tensor as tt
import pychrono.core as chrono
import pychrono.vehicle as veh
import time
import math
import random

# =================================================================================
# Plate pushing down to material, radius = 0.2m and 0.3m, pushing speed = 0.5~2cm/s
# First column: sinkage (m); Second column: 0.2m plate force (N) and 0.3m plate force (N)
# Data is in this file: 01_plot_plate_sinkage/plate_selected_force_sinkage_points_crm.txt
obsData = np.array([
    [2.500000000000000139e-02, 3.799593781980423955e+03],
    [5.000000000000000278e-02, 7.071550304958572269e+03],
    [7.500000000000001110e-02, 1.027906211598210211e+04],
    [1.000000000000000056e-01, 1.368405981537898515e+04],
    [1.250000000000000000e-01, 1.719653666033052650e+04],
    [1.500000000000000222e-01, 2.114784820937306358e+04],
    [1.750000000000000167e-01, 2.521768334237091040e+04],
    [2.000000000000000111e-01, 3.001722437581105623e+04],
    [2.500000000000000139e-02, 7.699234792362178268e+03],
    [5.000000000000000278e-02, 1.517459053253363527e+04],
    [7.500000000000001110e-02, 2.334488886762092079e+04],
    [1.000000000000000056e-01, 3.177416260828689701e+04],
    [1.250000000000000000e-01, 4.146838429193985212e+04],
    [1.500000000000000222e-01, 5.226071349810941319e+04],
    [1.750000000000000167e-01, 6.379186789361234696e+04],
    [2.000000000000000111e-01, 7.750203355261370598e+04],
])
# obsData = np.array([
#     [ 0.025 ,  3905.0 ],
#     [ 0.050 ,  7102.0 ],
#     [ 0.075 , 10278.0 ],
#     [ 0.100 , 13693.0 ],
#     [ 0.125 , 17203.0 ],
#     [ 0.150 , 21148.0 ],
#     [ 0.175 , 25221.0 ],
#     [ 0.200 , 30012.0 ],
#     [ 0.025 ,  7799.0 ],
#     [ 0.050 , 15232.0 ],
#     [ 0.075 , 23329.0 ],
#     [ 0.100 , 31788.0 ],
#     [ 0.125 , 41513.0 ],
#     [ 0.150 , 52356.0 ],
#     [ 0.175 , 63854.0 ],
#     [ 0.200 , 77640.0 ]
# ])

# Number of measured data
nSims = obsData.shape[0]

# Rescale the measured data
data_rescale = [10.0, 0.0001]
for i in range(nSims):
    obsData[i][0] = obsData[i][0] * data_rescale[0]
    obsData[i][1] = obsData[i][1] * data_rescale[1]

# Load data into the calibration system
input_data = obsData[:,0].flatten()
data_measured = obsData[:,1].flatten()

# Calibration system setup
sigma = 0.01
nchains = 4
nsteps = 1000000

# Bounds of the paramater and initial guess
lower = [-1.0e7, 0.0,   0.0]
upper = [ 1.0e7, 1.0e7, 3.0]
iniGuess = [1.0e4, 1.0e4, 1.0]

# Size of the plate
plate_rad = 0.2
plate_len = 0.05
plate_area = math.pi * plate_rad**2

# Name of the parameter (Bekker Wong)
K_c = 'K_c'
K_phi = 'K_phi'
n_exp = 'n'
x_rescale = [1.0, 1.0, 1.0]

# Output setup
out_dir = "DEMO_OUTPUT/Figure/python_scripts/nc_file/"
direction = "new_3_para_Normal_"
num_chains = str(nchains) + "_chains_"
num_steps = str(nsteps) + "_steps_"
sigma_val = "sigma_" + str(sigma) 
NC_File = out_dir + direction + num_chains + num_steps + sigma_val + ".nc"
Trace = out_dir + direction + "Trace_" + num_chains + num_steps + sigma_val + ".png"
Post = out_dir + direction + "Post_" + num_chains + num_steps + sigma_val + ".png"
fig_size1 = [10, 7.5]
fig_size2 = [15, 5]
text_size = 14.0

def F1(x, nSims):
    func_out = []
    for i in range(nSims):
        # read the sinkage
        z = input_data[i] / data_rescale[0]

        # get the k_c
        k_c = x[0] * x_rescale[0]

        # get the k_phi
        k_phi = x[1] * x_rescale[1]

        # get the n exponent
        n = x[2] * x_rescale[2]

        # calculate the pressure
        nr = plate_rad
        if i > 7:
            nr = plate_rad * 1.5
        P = (k_c / nr + k_phi) * z**n

        # calculate the force
        pa = plate_area
        if i > 7:
            pa = plate_area * 2.25
        F = P * pa * data_rescale[1]

        func_out.append(F)
        # print("[", x[0], ",", x[1], " ", z, ",", P, ",", F, ",", obsData[i][4], "]")

    return np.array(func_out)

def likelihood_func(cali_param, data, sigma):
    # t = time.time()
    # model_output = F(cali_param, nSims)
    model_output = F1(cali_param, nSims)
    # elapsed = time.time() - t
    # print(f"One sim time: {elapsed:f}")
    return_value = - (0.5 / sigma**2) * np.sum((model_output - data[:nSims])**2) / nSims
    print("cali_param and return_value = ", cali_param[0], cali_param[1], cali_param[2], return_value)
    return return_value

def main():
    print(f"Running on PyMC3 v{pm.__version__}") 
    print(f"There are {nSims:d} data points")
    
    logl = LogLike(likelihood_func, data_measured, sigma)
    basic_model = pm.Model()
    
    with basic_model:
        # x = pm.Uniform('x', lower=[0.0, 0.0], upper=[10.0, 5.0],  shape = 2) # x is calibration params
        x1 = pm.Uniform(K_c,   lower=lower[0], upper=upper[0])
        x2 = pm.Uniform(K_phi, lower=lower[1], upper=upper[1])
        x3 = pm.Uniform(n_exp, lower=lower[2], upper=upper[2])

        #sigma = pm.HalfNormal("sigma", sigma=1) 
        
        # theta = tt.as_tensor_variable(x)
        theta = tt.as_tensor_variable([x1,x2,x3])
        
        #pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        pm.Potential("likelihood",logl(theta))

    # map_est = pm.find_MAP(model=basic_model)
    # print(map_est)

    with basic_model:
        # step = pm.Metropolis(vars=x)
        step = pm.Metropolis(vars=[x1,x2,x3])

        tic = time.perf_counter()
        trace = pm.sample(nsteps, tune=nsteps//(nchains+1), chains=nchains, step=step, 
                          discard_tuned_samples=True, return_inferencedata=True,
                          start={K_c:iniGuess[0], K_phi:iniGuess[1], n_exp:iniGuess[2]})
                          # start={'x':[0.8, 0.9]})
        toc = time.perf_counter()
        print("===========================================================")
        print(f"Simulation finished in {toc - tic:0.4f} seconds")
        print("===========================================================")

        # save results into file for later post processing
        trace.to_netcdf(NC_File)
        
        # save results into pngs
        print(pm.summary(trace).to_string())
        fig1 = az.plot_trace(trace,figsize = (fig_size1[0], fig_size1[1]))
        fig1.ravel()[0].figure.savefig(Trace)

        fig2 = az.plot_posterior(trace,figsize = (fig_size2[0], fig_size2[1]), textsize = text_size)
        fig2.ravel()[0].figure.savefig(Post)


# Define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)
    
    def __init__(self, loglike, data, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl = self.likelihood(theta, self.data, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood


if __name__ == "__main__":
    main()

# =============================================================================
# def F(x, nSims):
#     # sample_no = random.randint(0, nSims-1)
#     # t = time.time()
#     func_out = [] 
#     for i in range(1):
#         # Create the mechanical system
#         mysystem = chrono.ChSystemSMC()

#         # The path to the Chrono data directory containing various assets
#         chrono.SetChronoDataPath('/srv/home/whu59/research/sbel/d_chrono_fsi_granular/chrono_3001/chrono_build/data/')

#         # Parameters for plate
#         plate_rad = 0.208
#         plate_len = 0.56
#         plate_center = chrono.ChVectorD(0, 0.001 + plate_len / 2.0, -1.5)

#         total_mass = 10.0
#         part_mass = total_mass / 2.0

#         lin_vel = 0.01 
#         ang_vel = 1.0 * math.pi / 180.0

#         # Create the ground
#         ground = chrono.ChBody()
#         ground.SetBodyFixed(True)
#         mysystem.Add(ground)

#         # Create the rigid plate with contact mesh
#         plate = chrono.ChBody()
#         mysystem.Add(plate)
#         plate.SetMass(total_mass - part_mass)
#         plate.SetInertiaXX(chrono.ChVectorD(10.0, 10.0, 10.0))
#         plate.SetPos(plate_center)
#         plate.SetPos_dt(chrono.ChVectorD(0.0, -2.0, 0.0))

#         # Load mesh
#         mesh = chrono.ChTriangleMeshConnected()
#         mesh.LoadWavefrontMesh(chrono.GetChronoDataFile('robot/viper/obj/viper_cylwheel.obj'))
#         # mesh.LoadWavefrontMesh('wheel_10mm_grouser.obj')
#         # mesh.LoadWavefrontMesh('hmmwv_tire_coarse_closed.obj')
#         mesh.Transform(chrono.ChVectorD(0, 0, 0), chrono.ChMatrix33D(chrono.Q_from_AngZ(0)))  

#         # Set visualization assets
#         # vis_shape = chrono.ChTriangleMeshShape()
#         # vis_shape.SetMesh(mesh)
#         # plate.AddAsset(vis_shape)
#         # plate.AddAsset(chrono.ChColorAsset(0.3, 0.3, 0.3))

#         # Set collision shape
#         material = chrono.ChMaterialSurfaceSMC()

#         plate.GetCollisionModel().ClearModel()
#         plate.GetCollisionModel().AddTriangleMesh(material,                # contact material
#                                                   mesh,                    # the mesh 
#                                                   False,                   # is it static?
#                                                   False,                   # is it convex?
#                                                   chrono.ChVectorD(0,0,0), # position on plate
#                                                   chrono.ChMatrix33D(1),   # orientation on plate 
#                                                   0.01)                    # "thickness" for increased robustness
#         plate.GetCollisionModel().BuildModel()
#         plate.SetCollide(True)

#         # Create chassis
#         chassis = chrono.ChBody()
#         mysystem.Add(chassis)
#         chassis.SetMass(part_mass)
#         chassis.SetInertiaXX(chrono.ChVectorD(10, 10, 10))
#         chassis.SetPos(plate_center)
#         chassis.SetCollide(False)

#         # Create axle
#         axle = chrono.ChBody()
#         mysystem.Add(axle)
#         axle.SetMass(part_mass)
#         axle.SetInertiaXX(chrono.ChVectorD(10, 10, 10))
#         axle.SetPos(plate_center)
#         axle.SetCollide(False)

#         # Create motor, actuator, and links
#         motor = chrono.ChLinkMotorRotationAngle()
#         # motor.SetSpindleConstraint(chrono.ChLinkMotorRotation.SpindleConstraint_OLDHAM)
#         motor.SetAngleFunction(chrono.ChFunction_Ramp(0, ang_vel))
#         motor.Initialize(plate, axle, chrono.ChFrameD(plate_center, chrono.Q_from_AngY(math.pi/2)))
#         mysystem.Add(motor)

#         actuator1 = chrono.ChLinkLinActuator()
#         actuator1.SetActuatorFunction(chrono.ChFunction_Ramp(0, 0))
#         actuator1.SetDistanceOffset(1)
#         actuator1.Initialize(ground, chassis, False, chrono.ChCoordsysD(plate_center, chrono.Q_from_AngY(math.pi/2)), 
#             chrono.ChCoordsysD(plate_center + chrono.ChVectorD(0, 0, 1.0), chrono.Q_from_AngY(math.pi/2)) )
#         mysystem.AddLink(actuator1)

#         actuator2 = chrono.ChLinkLinActuator()
#         actuator2.SetActuatorFunction(chrono.ChFunction_Ramp(0, lin_vel))
#         actuator2.SetDistanceOffset(1)
#         actuator2.Initialize(axle, chassis, False, chrono.ChCoordsysD(plate_center, chrono.Q_from_AngZ(math.pi/2)), 
#             chrono.ChCoordsysD(plate_center + chrono.ChVectorD(0, 1.0, 0), chrono.Q_from_AngZ(math.pi/2)) )
#         mysystem.AddLink(actuator2)

#         prismatic1 = chrono.ChLinkLockPrismatic()
#         prismatic1.Initialize(ground, chassis, chrono.ChCoordsysD(plate_center, chrono.QUNIT))
#         mysystem.AddLink(prismatic1)

#         prismatic2 = chrono.ChLinkLockPrismatic()
#         prismatic2.Initialize(axle, chassis, chrono.ChCoordsysD(plate_center, chrono.Q_from_AngX(-math.pi/2)))
#         mysystem.AddLink(prismatic2)

#         # Create SCM terrain patch
#         # Note that SCMDeformableTerrain uses a default ISO reference frame (Z up). 
#         # Since the mechanism is modeled here in a Y-up global frame, we rotate
#         # the terrain plane by -90 degrees about the X axis.
#         terrain = veh.SCMDeformableTerrain(mysystem)
#         terrain.SetPlane(chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngX(-math.pi/2)))
#         terrain.Initialize(1.0, 4.0, 0.01)

#         # Constant soil properties
#         terrain.SetSoilParameters( 9 * 2e5,     # Bekker Kphi 0.2e6
#                                    0,       # Bekker Kc 0
#                                    1.0 * 1.0,     # Bekker n exponent 1.1
#                                    0,       # Mohr cohesive limit (Pa) 0
#                                    1 * 30.0,    # Mohr friction limit (degrees) 30
#                                    1 * 0.01,    # Janosi shear coefficient (m) 0.01
#                                    4e7,     # Elastic stiffness (Pa/m), before plastic yield, must be > Kphi 4e7
#                                    3e4      # Damping (Pa s/m), proportional to negative vertical speed (optional) 3e4
#         )

#         # Run the simulation
#         ni = 0
#         F_final = 0.0
#         T_final = 0.0
#         S_final = 0.0
#         delta_t = 0.0
#         # total_mass = 200.0
#         while(mysystem.GetChTime() < ):    
#             mysystem.DoStepDynamics(0.01)
#             # if delta_t > 0.299999:
#             #     Force   = - motor.Get_react_force().x
#             #     Torque  = - motor.Get_react_torque().z
#             #     Sinkage = (tire_rad - wheel.GetPos().y) * 10.0
#                 # func_out.append(Sinkage)
#                 # print("[", mysystem.GetChTime(), ",", Force, ",", Torque, ",", Sinkage, ",", total_mass, "],")

#                 # tire_slip = tire_slip + 0.2
#                 # ang_vel = 0.175
#                 # lin_vel = ang_vel * (1.0 - tire_slip) * tire_rad
#                 # actuator.Set_dist_funct(chrono.ChFunction_Ramp(0, lin_vel))
#                 # motor.SetAngleFunction(chrono.ChFunction_Ramp(0, ang_vel))
#                 # total_mass = total_mass + 200.0
#                 # part_mass = total_mass / 2.0
#                 # wheel.SetMass(total_mass - part_mass)
#                 # axle.SetMass(part_mass)
#                 # delta_t = 0.0
#             # delta_t = delta_t + 0.01
#         #     if mysystem.GetChTime() > 2.8:
#         #         F_final = F_final - motor.Get_react_force().x
#         #         T_final = T_final - motor.Get_react_torque().z
#         #         S_final = S_final + (tire_rad - wheel.GetPos().y) * 1000.0
#         #         ni = ni + 1
#         # F_final = F_final / ni * 0.005
#         # T_final = T_final / ni * 1
#         # S_final = S_final / ni * 1
#         S_final = (plate_rad - wheel.GetPos().y) * 100.0
#         func_out.append(S_final)

#         del mysystem
#         print("[", obsData[i][0], ",", F_final, ",", T_final, ",", S_final, ",", total_mass, "]")

#     return np.array(func_out)
#     # return sinkage_sim, sample_no