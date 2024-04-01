import numpy as np
from numpy import cos, sin
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import fsolve
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import scipy
import random
from Model.force_fun import *
from Model.Integrator import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    return total_params
def visualize_training_pair(training_pair_input,training_pair_output):
    training_pair_input_np = training_pair_input.clone().cpu().detach().numpy()
    training_pair_output_np = training_pair_output.clone().cpu().detach().numpy()
    plt.figure()
    for i in range(training_pair_input_np.shape[1]):
        plt.scatter(training_pair_input_np[:, i, 0], training_pair_input_np[:, i, 1], label=f"body {i}_input")
        plt.scatter(training_pair_output_np[:, i, 0], training_pair_output_np[:, i, 1], label=f"body {i}_output")
    plt.show()
def set_seed(seed_value):
    random.seed(seed_value)           # Set seed for python's random module
    np.random.seed(seed_value)        # Set seed for numpy
    torch.manual_seed(seed_value)     # Set seed for pytorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Now, use the function to set the seed
#The visualization function used to visualize the training and testing results
def plot_energy_trajectory(test_case, numerical_methods, dt, data,test=False):
    plt.rcParams.update({'font.size': 12})
    #plt.rcParams["font.weight"] = "bold"
    #plt.rcParams["axes.labelweight"] = "bold"
    #Also set the font size of the axis tick labels
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    # Plot the trajectories
    print("Plotting the energy_trajectories for the test case: " + test_case)
    data = data.clone().cpu().detach().numpy()
    num_data, num_bodys, _ = data.shape
    # Set the color map
    cmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=num_data)
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    # Plot the trajectories
    fig, ax = plt.subplots()
    #use latex version for the font x,v for the axis
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v$')
    #ax.set_title('Trajectories')
    # Collect all x, y coordinates and colors
    all_x = data[:, :, 0].flatten()
    all_y = data[:, :, 1].flatten()
    colors1 = [scalarMap.to_rgba(i) for _ in range(num_bodys) for i in range(num_data)]
    # Plot all points in a single call
    ax.scatter(all_x, all_y, color=colors1, s=1)  # Reduced marker size for clarity
    if not os.path.exists("figures/" + test_case):
        os.makedirs("figures/" + test_case)
    if test:
        #add test to the file name
        plt.savefig("figures/" + test_case + "/Trajectories for " + test_case + " with integrator_" + numerical_methods + ' with dt=' + str(dt) + '_test.png')
    else:
        plt.savefig("figures/" + test_case + "/Trajectories for " + test_case + " with integrator_" + numerical_methods + ' with dt=' + str(dt) + '.png')
    # Plot the total energy
    # add subplots about the displacement and velocity vs time step for each body
    # start plotting
    fig, ax = plt.subplots(2, num_bodys,squeeze=False)
    #fig.suptitle('Displacement and Velocity vs. Time Step')
    for i in range(num_bodys):
        ax[0, i].set_xlabel('Time Step')
        ax[0, i].set_ylabel('$x$')
        ax[0, i].plot(range(num_data), data[:, i, 0])
        ax[0, i].set_title(f'Body {i} ')
        ax[1, i].set_xlabel('Time Step')
        ax[1, i].set_ylabel('$v$')
        ax[1, i].plot(range(num_data), data[:, i, 1])
    plt.tight_layout()
    if not os.path.exists("figures/" + test_case):
        os.makedirs("figures/" + test_case)
    if test:
        #add test to the file name
        plt.savefig("figures/" + test_case + "/Trajectories_XV for " + test_case + " with integrator_" + numerical_methods + ' with dt=' + str(dt) + '_test.png')
    else:
        plt.savefig("figures/" + test_case + "/Trajectories_XV for " + test_case + " with integrator_" + numerical_methods + ' with dt=' + str(dt) + '.png')

    fig, ax = plt.subplots()
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Energy')
    kinetic_energy = np.zeros(num_data)
    potential_energy = np.zeros(num_data)
    total_energy = np.zeros(num_data)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    # Check the test case once and then apply the relevant operations
    if test_case == "Single_Mass_Spring":
        kinetic_energy = 0.5 * np.sum(data[:, :, 1] ** 2 * 10, axis=1)
        potential_energy = 25 * np.sum(data[:, :, 0] ** 2, axis=1)

    elif test_case == "Single_Mass_Spring_Damper":
        kinetic_energy = 0.5 * np.sum(data[:, :, 1] ** 2 * 10, axis=1)
        potential_energy = 25 * np.sum(data[:, :, 0] ** 2, axis=1) + 0.05 * np.sum(data[:, :, 0] ** 4, axis=1)

    elif test_case == "Triple_Mass_Spring_Damper":
        for i in range(num_data):
            total_energy[i] = total_energy_tmsd(data_tensor[i])

    elif test_case == "Single_Mass_Spring_Symplectic":
        for i in range(num_data):
            total_energy[i] = total_energy_sms_symplectic(data_tensor[i])

    elif test_case == "Double_Pendulum":
        for i in range(num_data):
            total_energy[i] = double_pendulum_energy(data_tensor[i])

    # Summing up the energies
    total_energy += kinetic_energy + potential_energy

    # Plotting the data
    colors2 = [scalarMap.to_rgba(i) for i in range(num_data)]
    ax.scatter(range(num_data), total_energy, color=colors2)

    if not os.path.exists("figures/" + test_case):
        os.makedirs("figures/" + test_case)
    if test:
        #add test to the file name
        plt.savefig("figures/" + test_case + "/Energy for " + test_case + " with integrator_" + numerical_methods + ' with dt=' + str(dt) + '_test.png')
    else:
        plt.savefig("figures/" + test_case + "/Energy for " + test_case + " with integrator_" + numerical_methods + ' with dt=' + str(dt) + '.png')
    plt.show()

def compare_trajectories(test_case, numerical_methods,dt, data1, data2,num_training,if_symplectic=False,model_string="MBDPNODE"):
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    data1 = data1.clone().cpu().detach().numpy()
    data2 = data2.clone().cpu().detach().numpy()
    print(f"Plotting the trajectories for the test case: {test_case}")
    num_data = data1.shape[0]
    num_bodys = data1.shape[1]
    # for each body, plot the trajectory for data1 and data2 and compare them
    for i in range(num_bodys):
        #i=2
        fig, ax = plt.subplots()
        if not if_symplectic:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$v$')
        else:
            ax.set_xlabel('$q$')
            ax.set_ylabel('$p$')
        ax.set_title(f'Body {i}')
        # Use red color for the ground truth and blue color for the prediction
        # The ground truth is the data1 and the prediction is the data2
        # The ground truth should use solid line and the prediction should use dashed line
        # num_training is the number of points in the training range
        # other points are in the testing range and should be plotted with different colors
        ax.plot(data1[:num_training, i, 0], data1[:num_training, i, 1], color='blue', label='Ground Truth Training')
        ax.plot(data2[:num_training, i, 0], data2[:num_training, i, 1], '--', color='blue', label='Prediction Training')
        ax.plot(data1[num_training:, i, 0], data1[num_training:, i, 1], color='red',label='Ground Truth Testing')
        ax.plot(data2[num_training:, i, 0], data2[num_training:, i, 1], '--', color='red',label='Prediction Testing')
        ax.legend()
        # Save the figure
        save_path = os.path.join("figures", test_case)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path,
                                 f"Trajectories for {test_case} with model_{model_string}_integrator_{numerical_methods} with dt={dt}_body_{i}.png"))
        plt.show()

def compare_line_trajectories(test_case, numerical_methods, dt, data1, data2):
    """
    Plot and save the trajectories for the data using line plots.
    """
    data1 = data1.clone().cpu().detach().numpy()
    data2 = data2.clone().cpu().detach().numpy()
    print(f"Plotting the trajectories for the test case: {test_case}")
    num_data = data1.shape[0]
    num_bodys = data1.shape[1]
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectories')
    # Set the color map
    cmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=num_bodys)
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    # Plot the trajectories
    labels_added = {"Ground Truth": False, "Prediction": False}
    for j in range(num_bodys):
        #j=2
        colorVal = scalarMap.to_rgba(j * 100)
        label_gt = "Ground Truth" if not labels_added["Ground Truth"] else None
        label_pred = "Prediction" if not labels_added["Prediction"] else None

        ax.plot(data1[:, j, 0], data1[:, j, 1], color=colorVal, label=label_gt)
        ax.plot(data2[:, j, 0], data2[:, j, 1], '--', color=colorVal, label=label_pred)

        if label_gt: labels_added["Ground Truth"] = True
        if label_pred: labels_added["Prediction"] = True

    ax.legend()
    # Save the figure
    save_path = os.path.join("figures", test_case)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,
                             f"Compared_Trajectories for {test_case} with integrator_{numerical_methods} with dt={dt}.png"))
    plt.show()
def check_gradient(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, param.grad.norm())
        else:
            print(name, "No gradient")

def save_model(model,test_case,numerical_methods,model_string,dt,step_delay,training_size,num_epochs):
    save_path = os.path.join("saved_models", test_case)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path,
                                 f"{model_string} for {test_case} with integrator_{numerical_methods} with dt={dt}_step_delay={step_delay}_training_size={training_size}_num_epochs={num_epochs}.pt"))

# save the data for future plotting, with these labels: test_case, model string(predicted by which model or "ground truth" means ground truth) and training size and num_steps_test
def save_data(data,test_case,model_string,training_size,num_steps_test,dt):
    save_path = os.path.join("saved_data", test_case)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = data.clone().cpu().detach().numpy()
    print(f"Plotting the trajectories for the test case: {test_case}")
    np.save(os.path.join(save_path,
                                 f"{model_string} for {test_case} with training_size={training_size}_num_steps_test={num_steps_test}_dt={dt}.npy"),data)
def save_data_np(data,test_case,model_string,training_size,num_steps_test,dt):
    save_path = os.path.join("saved_data", test_case)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Plotting the trajectories for the test case: {test_case}")
    np.save(os.path.join(save_path,
                                 f"{model_string} for {test_case} with training_size={training_size}_num_steps_test={num_steps_test}_dt={dt}.npy"),data)
def mse(input, target):
    return ((input - target) ** 2).mean()
