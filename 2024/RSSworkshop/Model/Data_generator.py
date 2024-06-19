#This file contains all the data generators for the tests case used for this project
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Model.Integrator import *
from Model.force_fun import *
from Model.utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def simulate_slider_crank_dynamic_fe(theta_0, omega_initial, t_values, r_constant):
    theta_values = np.zeros_like(t_values)
    omega_values = np.zeros_like(t_values)
    x_values = np.zeros_like(t_values)
    dxdt_values = np.zeros_like(t_values)
    theta_values[0] = theta_0
    omega_values[0] = omega_initial
    r = 1.0  # length of crank
    l = 4.0  # length of connecting rod
    dt=t_values[1]-t_values[0]
    x_values[0] = r + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_0) ** 2)
    dxdt_values[0] = omega_initial * r * np.sin(theta_0)
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        # Dynamic angular acceleration
        alpha = r_constant * np.sin(r_constant * t)

        #convert the theta_values to the range of [0,2*pi]
        theta_values[i] = theta_values[i - 1] + omega_values[i-1] * dt
        #convert the theta_values to the range of [0,2*pi]
        theta_values[i]=theta_values[i]%(2*np.pi)
        omega_values[i] = omega_values[i - 1] + alpha * dt
        # Compute the slider's position and velocity
        x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
        dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
    return theta_values, omega_values, x_values, -dxdt_values
def simulate_slider_crank_dynamic_rk4(theta_0, omega_initial, t_values, r_constant):
    theta_values = np.zeros_like(t_values)
    omega_values = np.zeros_like(t_values)
    x_values = np.zeros_like(t_values)
    dxdt_values = np.zeros_like(t_values)
    theta_values[0] = theta_0
    omega_values[0] = omega_initial
    r = 1.0  # length of crank
    l = 4.0  # length of connecting rod
    dt = t_values[1] - t_values[0]
    x_values[0] = r + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_0) ** 2)
    dxdt_values[0] = omega_initial * r * np.sin(theta_0)
    def alpha_function(t):
        return r_constant * np.sin(r_constant * t)
    for i in range(1, len(t_values)):
        #convert the theta_values to the range of [0,2*pi]
        t = t_values[i - 1]
        # First set of evaluations
        k1_theta = omega_values[i - 1]
        k1_omega = alpha_function(t)
        # Second set of evaluations (midpoint)
        k2_theta = omega_values[i - 1] + 0.5 * k1_omega * dt
        k2_omega = alpha_function(t + 0.5 * dt)
        # Third set of evaluations (midpoint)
        k3_theta = omega_values[i - 1] + 0.5 * k2_omega * dt
        k3_omega = alpha_function(t + 0.5 * dt)
        # Fourth set of evaluations
        k4_theta = omega_values[i - 1] + k3_omega * dt
        k4_omega = alpha_function(t + dt)
        # Weighted sum of evaluations to estimate the function's value at the next time step
        theta_values[i] = theta_values[i - 1] + (dt/6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        #convert the theta_values to the range of [0,2*pi]
        theta_values[i]=theta_values[i]%(2*np.pi)
        omega_values[i] = omega_values[i - 1] + (dt/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        # Compute the slider's position and velocity
        x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
        dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
    return theta_values, omega_values, x_values, dxdt_values
def generate_training_data(test_case,numerical_methods,dt,num_steps,if_noise=False,if_external_force=False,external_force_function=None,aug=False,model=None):
    print("Generating training data for test case: "+test_case)
    #This function generates training data for the test cases
    #test_case is a string that indicates which test case we are using
    #num_bodys is the number of bodys we want to simulate
    #T is the time we want to simulate
    #dt is the time step we want to use
    #n_samples is the number of samples we want to generate
    #if_noise is a boolean variable that indicates whether we want to add noise to the data
    #if_external_force is a boolean variable that indicates whether we want to add external force to the data
    #external_force_function is a function that takes in time and returns the external force
    #numerical_methods is a string that indicates which numerical methods we want to use
    #The following code is used to generate the training data for the test cases
    #The first step is to generate the initial conditions for the test cases
    #The initial conditions are generated as follows:
    #For the single mass spring system, we generate the initial conditions from a normal distribution with mean 0 and variance 1
    if test_case=="Single_Mass_Spring":
        print("Generating training data for single mass spring system")
        bodys = np.array([[1,0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            training_set = forward_euler_multiple_body(bodys_tensor,force_sms,num_steps,dt,if_final_state=False,model=model)
        if numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            training_set = runge_kutta_four_multiple_body(bodys_tensor,force_sms,num_steps,dt,if_final_state=False,model=model)
        if numerical_methods=="hf":
            print("Using analytical solution to generate the training data")
            training_set = torch.zeros((num_steps,1,2),dtype=torch.float32,device=device)
            training_set[0,:,:]=bodys_tensor
            for i in range(num_steps-1):
                training_set[i+1,:,:]=analytic_sms(training_set[i,:,:],dt,model=model)
        if numerical_methods not in ["fe","rk4","hf"]:
            raise ValueError("The numerical method for single mass spring system has to be fe, rk4 or hf")
        if if_noise:
            training_set+=torch.randn_like(training_set)*0.01
        return training_set
    if test_case=="Single_Mass_Spring_Damper":
        print("Generating training data for single mass spring system with damper")
        bodys = np.array([[1,0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            training_set = forward_euler_multiple_body(bodys_tensor,force_smsd,num_steps,dt,if_final_state=False,model=model)
        if numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            training_set = runge_kutta_four_multiple_body(bodys_tensor,force_smsd,num_steps,dt,if_final_state=False,model=model)
        if if_noise:
            training_set+=torch.randn_like(training_set)*0.003
        return training_set
    if test_case=="Triple_Mass_Spring_Damper":
        print("Generating training data for triple mass spring system with damper")
        bodys = np.array([[1,0],[2,0],[3,0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            training_set = forward_euler_multiple_body(bodys_tensor,force_tmsd,num_steps,dt,if_final_state=False,model=model)
        if numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            training_set = runge_kutta_four_multiple_body(bodys_tensor,force_tmsd,num_steps,dt,if_final_state=False,model=model)
        if if_noise:
            training_set+=torch.randn_like(training_set)*0.003
        return training_set
    if test_case=="Single_Mass_Spring_Symplectic":
        print("Generating training data for single mass spring system using symplectic integrator")
        #The generalized coordinates, and the generalized momentum for the single mass spring system with symplectic integrator
        bodys = np.array([[1,0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="sep_sv":
            print("Using Seperable Störmer-Verlet method to generate the training data")
            training_set = sep_stormer_verlet_multiple_body(bodys_tensor,dH_dq_smp,dH_dp_smp,num_steps,dt,if_final_state=False)
        elif numerical_methods=="yoshida4":
            print("Using Yoshida4 method to generate the training data")
            training_set = yoshida4_multiple_body(bodys_tensor,dH_dq_smp,dH_dp_smp,num_steps,dt,if_final_state=False)
        elif numerical_methods=="fukushima6":
            print("Using Fukushima6 method to generate the training data")
            training_set = fukushima6_multiple_body(bodys_tensor,dH_dq_smp,dH_dp_smp,num_steps,dt,if_final_state=False)
        else:
            raise ValueError("The numerical method for single mass spring system has to be sep_sv, yoshida4 or fukushima6")
        if if_noise:
            training_set+=torch.randn_like(training_set)*0.001
        return training_set
    if test_case=="Double_Pendulum":
        print("Generating training data for double pendulum system")
        #The polar coordinates for the double pendulum system
        bodys = np.array([[np.pi/2,0,0,0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            th1 = 2 / 4 * 180
            w1 = 0.0
            th2 = 1 / 4 * 180
            w2 = 0.0
            # initial state
            state = np.radians([th1, w1, th2, w2])
            dt = dt
            t_stop = dt*num_steps  # how many seconds to simulate
            t = np.linspace(0, t_stop, num_steps)
            y = scipy.integrate.solve_ivp(double_pendulum_derivs, t[[0, -1]], state, t_eval=t, rtol=1e-10, atol=1e-10,method='RK45').y.T
            print("y shape is "+str(y.shape)
            )
            #rearrange the y to the form of bodys tensor
            training_set = torch.zeros((len(t),2,2),dtype=torch.float32,device=device)
            training_set[:,0,:]=torch.tensor(y[:,0:2],dtype=torch.float32,device=device)
            training_set[:,1,:]=torch.tensor(y[:,2:4],dtype=torch.float32,device=device)
        else:
            raise ValueError("The numerical method for double pendulum system has to be rk4")
        if if_noise:
            training_set+=torch.randn_like(training_set)*0.001
        return training_set
    if test_case=="Slider_Crank":
        print("Generating training data for slider crank system")
        theta_0 = 0
        omega_0 = 1 * np.pi  # 1 rotation per second
        r = 1.0  # length of crank
        l = 4.0  # length of connecting rod
        r_constant = 4.0  # Example value for r
        x_0 = r_constant * np.cos(theta_0) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_0) ** 2)
        dx_0 = omega_0 * r * np.sin(theta_0)
        t_0 = 0
        t_values = np.arange(0, num_steps * dt, dt)
        initial = np.array([[theta_0,omega_0,t_0],[x_0,dx_0,t_0]])
        #bodys_tensor = torch.zeros((num_steps,2,3),dtype=torch.float32,device=device)
        bodys_tensor = torch.zeros((num_steps, 2, 2), dtype=torch.float32, device=device)
        print("bodys_tensor shape is "+str(bodys_tensor.shape))
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            theta_values, omega_values, x_values, dxdt_values = simulate_slider_crank_dynamic_fe(theta_0, omega_0,t_values, r_constant)
            #convert the theta_values to the range of [0,2*pi]
            theta_values=theta_values%(2*np.pi)
            #using the theta_values and x_values to generate the bodys_tensor
            bodys_tensor[:,0,0]=torch.tensor(theta_values,dtype=torch.float32,device=device)
            bodys_tensor[:,0,1]=torch.tensor(omega_values,dtype=torch.float32,device=device)
            #bodys_tensor[:,0,2]=torch.tensor(t_values,dtype=torch.float32,device=device)
            bodys_tensor[:,1,0]=torch.tensor(x_values,dtype=torch.float32,device=device)
            bodys_tensor[:,1,1]=torch.tensor(dxdt_values,dtype=torch.float32,device=device)
            #bodys_tensor[:,1,2]=torch.tensor(t_values,dtype=torch.float32,device=device)
        elif numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            theta_values, omega_values, x_values, dxdt_values = simulate_slider_crank_dynamic_rk4(theta_0, omega_0,t_values, r_constant)
            #using the theta_values and x_values to generate the bodys_tensor
            bodys_tensor[:,0,0]=torch.tensor(theta_values,dtype=torch.float32,device=device)
            bodys_tensor[:,0,1]=torch.tensor(omega_values,dtype=torch.float32,device=device)
            #bodys_tensor[:,0,2]=torch.tensor(t_values,dtype=torch.float32,device=device)
            bodys_tensor[:,1,0]=torch.tensor(x_values,dtype=torch.float32,device=device)
            bodys_tensor[:,1,1]=torch.tensor(dxdt_values,dtype=torch.float32,device=device)
            #bodys_tensor[:,1,2]=torch.tensor(t_values,dtype=torch.float32,device=device)
        return bodys_tensor
    if test_case=="Single_Pendulum":
        print("Generating training data for single pendulum system")
        theta_0 = 0
        omega_0 = 1 * np.pi
        t_0 = 0
        t_values = np.arange(0, num_steps * dt, dt)
        #initial should be a tensor with shape (1,2)
        initial = torch.tensor([[theta_0,omega_0]])
        bodys_tensor = torch.zeros((num_steps, 1, 2), dtype=torch.float32, device=device)
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            data_set = forward_euler_multiple_body(initial,force_sp,num_steps,dt,if_final_state=False,model=model)
        elif numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            data_set = runge_kutta_four_multiple_body(initial,force_sp,num_steps,dt,if_final_state=False,model=model)
        elif numerical_methods=="midpoint":
            print("Using midpoint method to generate the training data")
            data_set = midpoint_method_multiple_body(initial,force_sp,num_steps,dt,if_final_state=False,model=model)
        else:
            raise ValueError("The numerical method for single pendulum system has to be fe or rk4")
        if if_noise:
            data_set+=torch.randn_like(data_set)*0.001
        return data_set
    if test_case=="Cartpole":
        print("Generating training data for cartpole system")
        #The initial conditions for the cartpole system
        #bodys = np.array([[0,0,0,0]])
        bodys=np.array([[1, 0], [1 / 6 * np.pi, 0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            training_set = forward_euler_multiple_body(bodys_tensor,force_cp,num_steps,dt,if_final_state=False,model=model)
        if numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            training_set = runge_kutta_four_multiple_body(bodys_tensor,force_cp,num_steps,dt,if_final_state=False,model=model)
        if numerical_methods=="midpoint":
            print("Using midpoint method to generate the training data")
            training_set = midpoint_method_multiple_body(bodys_tensor,force_cp,num_steps,dt,if_final_state=False,model=model)

        if if_noise:
            training_set+=torch.randn_like(training_set)*0.001
        return training_set
    if test_case=="Slider_Crank_D":
        print("Generating training data for slider crank system with dynamic force")
        theta_0 = 0
        omega_0 = 1 * np.pi

def generate_sampling_training_data(test_case,numerical_methods,dt,num_steps,if_noise=False,if_external_force=False,external_force_function=None,model=None):
    print("Generating sampling training data for test case: "+test_case)
    #This function generates training data for the test cases
    #It returns two tensors, one is the inputs as the initial conditions, the other is the output for the next time step, the shape of the two tensors are (num_samples,num_bodys,2)
    #test_case is a string that indicates which test case we are using
    #num_bodys is the number of bodys we want to simulate
    #T is the time we want to simulate
    #dt is the time step we want to use
    #n_samples is the number of samples we want to generate
    #if_noise is a boolean variable that indicates whether we want to add noise to the data
    #if_external_force is a boolean variable that indicates whether we want to add external force to the data
    #external_force_function is a function that takes in time and returns the external force
    #numerical_methods is a string that indicates which numerical methods we want to use
    #The following code is used to generate the training data for the test cases
    #The first step is to generate the initial conditions for the test cases
    #The initial conditions are generated as follows:
    #For the single mass spring system, we generate the initial conditions from a normal distribution with mean 0 and variance 1
    if test_case=="Single_Mass_Spring":
        print("Generating training data for single mass spring system")
        #random uniformly generate the initial conditions
        bodys = np.random.uniform(-1,1,(num_steps,1,2))
        num_body=bodys.shape[1]
        #normalize the initial conditions to reasonable range, create the normalized bodys tensor
        normalization_constant = np.array([[1,2]])
        #Normalize the bodys tensor for the body_num dimension(each body), position and velocity dimension
        for i in range(num_body):
            bodys[:,i,:]=bodys[:,i,:]*normalization_constant[0,:]#
        print("bodys shape is "+str(bodys.shape))
        training_input = torch.tensor(bodys[0:-1,:],dtype=torch.float32,device=device)
        print("training input shape is "+str(training_input.shape))
        #training_output should have the same shape as training_input and it will be calculated using the numerical methods
        training_output = torch.zeros_like(training_input)
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            #Using forward Euler method to generate the training_output from training_input
            for i in range(num_steps-1):
                training_output[i,:] = forward_euler_multiple_body(training_input[i,:],force_sms,2,dt,if_final_state=True,model=model)
        elif numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            for i in range(num_steps-1):
                training_output[i,:] = runge_kutta_four_multiple_body(training_input[i,:],force_sms,2,dt,if_final_state=True,model=model)
        elif numerical_methods=="hf":
            print("Using analytical solution to generate the training data")
            #Using analytical solution to generate the training_output from training_input
            for i in range(num_steps-1):
                training_output[i,:] = analytic_sms(training_input[i,:],dt,model=model)
        else:
            raise ValueError("The numerical method for single mass spring system has to be fe, rk4 or hf")
        if if_noise:
            training_output+=torch.randn_like(training_output)*0.01
        return training_input,training_output
    if test_case=="Single_Mass_Spring_Damper":
        print("Generating training data for single mass spring system with damper")
        bodys = np.random.uniform(-1, 1, (num_steps, 1, 2))
        num_body = bodys.shape[1]
        # normalize the initial conditions to reasonable range, create the normalized bodys tensor
        normalization_constant = np.array([[1, 2]])
        # Normalize the bodys tensor for the body_num dimension(each body), position and velocity dimension
        for i in range(num_body):
            bodys[:, i, :] = bodys[:, i, :] * normalization_constant[0, :]  #
        print("bodys shape is " + str(bodys.shape))
        training_input = torch.tensor(bodys[0:-1, :], dtype=torch.float32, device=device)
        print("training input shape is " + str(training_input.shape))
        # training_output should have the same shape as training_input and it will be calculated using the numerical methods
        training_output = torch.zeros_like(training_input)
        if numerical_methods == "fe":
            print("Using forward Euler method to generate the training data")
            # Using forward Euler method to generate the training_output from training_input
            for i in range(num_steps - 1):
                training_output[i, :] = forward_euler_multiple_body(training_input[i, :], force_smsd, 2, dt,
                                                                    if_final_state=True, model=model)
        elif numerical_methods == "rk4":
            print("Using RK4 method to generate the training data")
            for i in range(num_steps - 1):
                training_output[i, :] = runge_kutta_four_multiple_body(training_input[i, :], force_smsd, 2, dt,
                                                                       if_final_state=True, model=model)
        else:
            raise ValueError("The numerical method for single mass spring system has to be fe, rk4")
        if if_noise:
            training_output += torch.randn_like(training_output) * 0.01
        return training_input,training_output
    if test_case=="Triple_Mass_Spring_Damper":
            print("Generating training data for triple mass spring system with damper")
            bodys = np.random.uniform(-1, 1, (num_steps, 3, 2))
            num_body = bodys.shape[1]
            # normalize the initial conditions to reasonable range, create the normalized bodys tensor
            normalization_constant = np.array([[1, 1],[2,3],[2.5,5.5]])
            bias_constant = np.array([[0, 0], [-0.5, 0], [1, -2.2]])
            # Normalize the bodys tensor for the body_num dimension(each body), position and velocity dimension
            for i in range(num_body):
                bodys[:, i, :] = (bodys[:, i, :] *normalization_constant[i, :] )+bias_constant [i,:]#
            print("bodys shape is " + str(bodys.shape))
            training_input = torch.tensor(bodys[0:-1, :], dtype=torch.float32, device=device)
            # training_output should have the same shape as training_input and it will be calculated using the numerical methods
            training_output = torch.zeros_like(training_input)
            if numerical_methods == "fe":
                print("Using forward Euler method to generate the training data")
                # Using forward Euler method to generate the training_output from training_input
                for i in range(num_steps - 1):
                    training_output[i, :] = forward_euler_multiple_body(training_input[i, :], force_smsd, 2, dt,
                                                                        if_final_state=True, model=model)
            elif numerical_methods == "rk4":
                print("Using RK4 method to generate the training data")
                for i in range(num_steps - 1):
                    training_output[i, :] = runge_kutta_four_multiple_body(training_input[i, :], force_smsd, 2, dt,
                                                                           if_final_state=True, model=model)
            else:
                raise ValueError("The numerical method for single mass spring system has to be fe, rk4")
            if if_noise:
                training_output += torch.randn_like(training_output) * 0.01
            return training_input, training_output
    if test_case=="Single_Mass_Spring_Symplectic":
        print("Generating training data for single mass spring system using symplectic integrator")
        #The generalized coordinates, and the generalized momentum for the single mass spring system with symplectic integrator
        bodys = np.array([[1,0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="sep_sv":
            print("Using Seperable Störmer-Verlet method to generate the training data")
            training_set = sep_stormer_verlet_multiple_body(bodys_tensor,dH_dq_smp,dH_dp_smp,num_steps,dt,if_final_state=False)
            print("training set shape is "+str(training_set.shape))
        if if_noise:
            training_set+=torch.randn_like(training_set)*0.001
        return training_set
    if test_case=="Double_Pendulum":
        print("Generating training data for double pendulum system")
        #The polar coordinates for the double pendulum system
        bodys = np.array([[np.pi/2,0,0,0]])
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        if numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            th1 = 3 / 7 * 180
            w1 = 0.0
            th2 = 3 / 4 * 180
            w2 = 0.0
            # initial state
            state = np.radians([th1, w1, th2, w2])
            dt = dt
            t_stop = dt*num_steps  # how many seconds to simulate
            t = np.linspace(0, t_stop, num_steps)
            y = scipy.integrate.solve_ivp(double_pendulum_derivs, t[[0, -1]], state, t_eval=t, rtol=1e-10, atol=1e-10,method='RK45').y.T
            print("y shape is "+str(y.shape)
            )
            #rearrange the y to the form of bodys tensor
            training_set = torch.zeros((len(t),2,2),dtype=torch.float32,device=device)
            training_set[:,0,:]=torch.tensor(y[:,0:2],dtype=torch.float32,device=device)
            training_set[:,1,:]=torch.tensor(y[:,2:4],dtype=torch.float32,device=device)
        else:
            raise ValueError("The numerical method for double pendulum system has to be rk4")
        if if_noise:
            training_set+=torch.randn_like(training_set)*0.001
        return training_set
    if test_case=="Cartpole":
        print("Generating training data for cartpole system")
        #The initial conditions for the cartpole system
        #bodys = np.array([[0,0,0,0]])
        #sampling range for the displacement and velocity of the cartpole
        x_min = -1.5
        x_max = 1.5
        x_dot_min = -4
        x_dot_max = 4
        #sampling range for the angle and angular velocity of the cartpole
        theta_min = 0
        theta_max = 2*np.pi
        theta_dot_min = -8
        theta_dot_max = 8
        #sampling for the external force
        F_min = -15
        F_max = 30
        bodys = np.random.uniform([[x_min, x_dot_min],[ theta_min, theta_dot_min]],[[x_max, x_dot_max],[ theta_max, theta_dot_max]],(num_steps,2,2))
        u=np.random.uniform(F_min,F_max,(num_steps,1))
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        u_tensor = torch.tensor(u,dtype=torch.float32,device=device)
        #call force_cp_ext to generate the acceleration for each time step
        accel_tensor = force_cp_ext_sequence(bodys_tensor,u_tensor)
        return bodys_tensor,u_tensor,accel_tensor
    if test_case=="Slider_Crank_D":
        print("Generating training data for slider crank system")
        theta_min = 0
        theta_max = 2*np.pi
        theta_dot_min=-3
        theta_dot_max=3
        F_min = -10
        F_max = 10
        Tao_min = -10
        Tao_max = 10
        bodys = np.random.uniform([[theta_min,theta_dot_min]],[[theta_max,theta_dot_max]],(num_steps,1,2))
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        u=np.random.uniform([[F_min,Tao_min]], [[F_max, Tao_max]], (num_steps, 2))
        #call the force function to generate the acceleration for each time step
        accel = force_scd_sequence(bodys,u)
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
        bodys_tensor = torch.tensor(bodys, dtype=torch.float32, device=device)
        accel_tensor = torch.tensor(accel, dtype=torch.float32, device=device)
        return bodys_tensor,u_tensor,accel_tensor
    if test_case=="Single_Pendulum_P":
        print("Generating training data for single pendulum system with different length")
        theta_min = -np.pi
        theta_max = np.pi
        theta_dot_min=-4
        theta_dot_max=4
        l_min = 0.1
        l_max = 10
        bodys = np.random.uniform([[theta_min,theta_dot_min]],[[theta_max,theta_dot_max]],(num_steps,1,2))
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        p=np.random.uniform(l_min,l_max,(num_steps,1))
        #call the force function to generate the acceleration for each time step

        p_tensor = torch.tensor(p, dtype=torch.float32, device=device)
        accel = force_sc_P_sequence(bodys_tensor, p_tensor)
        bodys_tensor = torch.tensor(bodys, dtype=torch.float32, device=device)
        accel_tensor = torch.tensor(accel, dtype=torch.float32, device=device)
        return bodys_tensor,p_tensor,accel_tensor
    if test_case=="Slider_Crank_P":
        print("Generating training data for slider crank system")
        theta_min = 0
        theta_max = 2*np.pi
        theta_dot_min=-3
        theta_dot_max=3
        m1_min = 5
        m1_max = 10
        m2_min = 5
        m2_max = 10
        m3_min = 5
        m3_max = 10
        I1_min = 5
        I1_max = 10
        I2_min = 5
        I2_max = 10
        I3_min = 5
        I3_max = 10
        l1_min=4
        l1_max=6
        l2_min=9
        l2_max=15
        bodys = np.random.uniform([[theta_min,theta_dot_min]],[[theta_max,theta_dot_max]],(num_steps,1,2))
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        #u=np.random.uniform([[F_min,Tao_min]], [[F_max, Tao_max]], (num_steps, 2))
        p=np.random.uniform([[m1_min,m2_min,m3_min,I1_min,I2_min,I3_min,l1_min,l2_min]],[[m1_max,m2_max,m3_max,I1_max,I2_max,I3_max,l1_max,l2_max]],(num_steps, 8))
        #call the force function to generate the acceleration for each time step
        accel = force_sc_P_sequence(bodys,p)
        u_tensor = torch.tensor(p, dtype=torch.float32, device=device)
        bodys_tensor = torch.tensor(bodys, dtype=torch.float32, device=device)
        accel_tensor = torch.tensor(accel, dtype=torch.float32, device=device)
        return bodys_tensor,u_tensor,accel_tensor
    if test_case=="Single_Pendulum_P1":
        print("Generating training data for single pendulum system with different length")
        theta_min = -np.pi
        theta_max = np.pi
        theta_dot_min=-4
        theta_dot_max=4
        l_min = 0.1
        l_max = 10
        bodys = np.random.uniform([[theta_min,theta_dot_min]],[[theta_max,theta_dot_max]],(num_steps,1,2))
        bodys_tensor = torch.tensor(bodys,dtype=torch.float32,device=device)
        p=np.random.uniform(l_min,l_max,(num_steps,1))
        #call the force function to generate the acceleration for each time step

        p_tensor = torch.tensor(p, dtype=torch.float32, device=device)
        accel = force_sc_P_sequence(bodys_tensor, p_tensor)
        bodys_tensor = torch.tensor(bodys, dtype=torch.float32, device=device)
        accel_tensor = torch.tensor(accel, dtype=torch.float32, device=device)
        return bodys_tensor,p_tensor,accel_tensor


def generate_ground_truth_initial(test_case,initial,numerical_methods,dt,num_steps,if_noise=False,if_external_force=False,external_force_function=None,aug=False,model=None):
    if test_case=="Single_Pendulum":
        print("Generating training data for single pendulum system")
        theta_0 = 0
        omega_0 = 1 * np.pi
        t_0 = 0
        t_values = np.arange(0, num_steps * dt, dt)
        #initial should be a tensor with shape (1,2)
        #initial = torch.tensor([[theta_0,omega_0]])
        bodys_tensor = torch.zeros((num_steps, 1, 2), dtype=torch.float32, device=device)
        if numerical_methods=="fe":
            print("Using forward Euler method to generate the training data")
            data_set = forward_euler_multiple_body(initial,force_sp,num_steps,dt,if_final_state=False,model=model)
        elif numerical_methods=="rk4":
            print("Using RK4 method to generate the training data")
            data_set = runge_kutta_four_multiple_body(initial,force_sp,num_steps,dt,if_final_state=False,model=model)
        elif numerical_methods=="midpoint":
            print("Using midpoint method to generate the training data")
            data_set = midpoint_method_multiple_body(initial,force_sp,num_steps,dt,if_final_state=False,model=model)
        else:
            raise ValueError("The numerical method for single pendulum system has to be fe or rk4")
        if if_noise:
            data_set+=torch.randn_like(data_set)*0.001
        return data_set
